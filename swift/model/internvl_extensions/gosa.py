# Copyright (c) ModelScope Contributors. All rights reserved.
"""GOSA/GOISE spatial encoding extension for a baseline InternVL model.

The pretrained model directory intentionally remains an upstream baseline.
This module attaches the optional local/global spatial encoders after the
baseline weights have been loaded, while preserving the historical checkpoint
key names ``pos_embed_2d`` and ``global_spatial_encoder``.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from types import MethodType

import torch
from torch import nn


class Learnable2DPositionalEncoding(nn.Module):

    def __init__(self, dim: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.half_dim = dim // 2
        self.row_embed = nn.Parameter(torch.zeros(max_h, self.half_dim))
        self.col_embed = nn.Parameter(torch.zeros(max_w, self.half_dim))
        nn.init.normal_(self.row_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.col_embed, mean=0.0, std=0.02)

    def forward(self, h, w, dtype, device):
        row = self.row_embed[:h]
        col = self.col_embed[:w]
        row_pos = row.unsqueeze(1).repeat(1, w, 1)
        col_pos = col.unsqueeze(0).repeat(h, 1, 1)
        return torch.cat([row_pos, col_pos], dim=-1).unsqueeze(0).to(dtype=dtype, device=device)


class GlobalTilePositionalEncoding(nn.Module):
    """Geometry-correct page coordinates for compressed visual tokens."""

    VALID_FUSIONS = {'add', 'scalar_gate', 'channel_gate'}
    VALID_THUMBNAIL_MODES = {'full', 'type_only', 'off'}
    VALID_COORD_MODES = {'real', 'uniform', 'shuffled', 'disabled', 'full_page'}

    def __init__(self,
                 dim,
                 num_bands=16,
                 fusion='add',
                 gate_init=0.1,
                 thumbnail_mode='full',
                 coord_mode='real',
                 coord_seed=0):
        super().__init__()
        if fusion not in self.VALID_FUSIONS:
            raise ValueError(f'Unsupported GLOBAL_PE_FUSION={fusion!r}; expected one of {sorted(self.VALID_FUSIONS)}')
        if thumbnail_mode not in self.VALID_THUMBNAIL_MODES:
            raise ValueError(
                f'Unsupported GLOBAL_PE_THUMBNAIL={thumbnail_mode!r}; '
                f'expected one of {sorted(self.VALID_THUMBNAIL_MODES)}')
        if coord_mode not in self.VALID_COORD_MODES:
            raise ValueError(
                f'Unsupported GLOBAL_PE_COORD_MODE={coord_mode!r}; expected one of {sorted(self.VALID_COORD_MODES)}')
        if not 0. < gate_init < 1.:
            raise ValueError(f'GLOBAL_PE_GATE_INIT must be in (0, 1), got {gate_init}')
        self.fusion = fusion
        self.thumbnail_mode = thumbnail_mode
        self.coord_mode = coord_mode
        self.coord_seed = int(coord_seed)
        self.register_buffer('freq_bands', torch.pi * (2.**torch.arange(num_bands)), persistent=False)
        fourier_dim = 2 + 4 * num_bands
        self.mlp = nn.Sequential(nn.Linear(fourier_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.tile_type_embedding = nn.Embedding(2, dim)
        nn.init.zeros_(self.tile_type_embedding.weight)
        gate_logit = math.log(gate_init / (1. - gate_init))
        if fusion == 'scalar_gate':
            self.gate_logit = nn.Parameter(torch.full((1,), gate_logit))
        elif fusion == 'channel_gate':
            self.gate_logit = nn.Parameter(torch.full((dim,), gate_logit))
        else:
            self.register_parameter('gate_logit', None)
        self._warned_missing_tile_types = False
        # The global branch is neutral before training.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _encode_positions(self, tile_boxes, h, w, dtype, device):
        boxes = tile_boxes.to(device=device, dtype=torch.float32)
        rows = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
        cols = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
        y, x = torch.meshgrid(rows, cols, indexing='ij')
        local_xy = torch.stack([x, y], dim=-1).unsqueeze(0)
        global_xy = boxes[:, None, None, :2] + local_xy * (boxes[:, None, None, 2:] - boxes[:, None, None, :2])
        angles = global_xy.unsqueeze(-1) * self.freq_bands.to(device=device)
        fourier = torch.cat(
            [global_xy, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)], dim=-1)
        encoding = self.mlp(fourier.to(dtype=self.mlp[0].weight.dtype))
        return encoding.to(dtype=dtype)

    @staticmethod
    def _uniform_boxes(count, *, dtype, device):
        """Lay ``count`` tiles on a page-aspect-independent near-square grid."""
        if count == 0:
            return torch.empty((0, 4), dtype=dtype, device=device)
        columns = math.ceil(math.sqrt(count))
        rows = math.ceil(count / columns)
        index = torch.arange(count, dtype=dtype, device=device)
        col = torch.remainder(index, columns)
        row = torch.div(index, columns, rounding_mode='floor')
        return torch.stack([col / columns, row / rows, (col + 1) / columns, (row + 1) / rows], dim=-1)

    def _coordinate_ablation(self, tile_boxes, tile_types):
        """Return parameter-matched coordinates for the requested ablation."""
        if self.coord_mode == 'real':
            return tile_boxes
        if self.coord_mode == 'full_page':
            # Historical alias retained for reproducing the earlier ablation.
            return tile_boxes.new_tensor([0., 0., 1., 1.]).expand_as(tile_boxes)

        regular = tile_types == 0
        regular_boxes = tile_boxes[regular]
        if self.coord_mode == 'uniform':
            replacement = self._uniform_boxes(
                regular_boxes.shape[0], dtype=tile_boxes.dtype, device=tile_boxes.device)
        elif self.coord_mode == 'shuffled':
            # A local CPU generator makes the mapping independent of global RNG
            # state, worker count, and train/evaluation process order. Use a
            # derangement so ``shuffled`` never silently leaves crop-coordinate
            # pairs intact (except for the unavoidable one-crop case).
            generator = torch.Generator(device='cpu')
            generator.manual_seed(self.coord_seed)
            count = regular_boxes.shape[0]
            permutation = torch.arange(count)
            if count > 1:
                identity = permutation
                for _ in range(100):
                    candidate = torch.randperm(count, generator=generator)
                    if torch.all(candidate != identity):
                        permutation = candidate
                        break
                else:
                    permutation = torch.roll(identity, shifts=1)
            permutation = permutation.to(tile_boxes.device)
            replacement = regular_boxes[permutation]
        else:
            # ``disabled`` is handled after encoding so the complete module stays
            # in the graph and has exactly the same parameter count.
            replacement = regular_boxes
        transformed = tile_boxes.clone()
        transformed[regular] = replacement
        return transformed

    @staticmethod
    def _sanitize_gate_gradient(grad):
        return torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)

    def forward(self, tile_boxes, tile_types, h, w, dtype, device):
        if tile_types is None:
            if not self._warned_missing_tile_types:
                warnings.warn('GOSA tile_types were not supplied; treating every input as a regular tile.', stacklevel=2)
                self._warned_missing_tile_types = True
            tile_types = torch.zeros(tile_boxes.shape[0], dtype=torch.long, device=device)
        else:
            tile_types = tile_types.to(device=device, dtype=torch.long)
        if tile_types.ndim != 1 or tile_types.shape[0] != tile_boxes.shape[0]:
            raise ValueError(f'tile_types must have shape [{tile_boxes.shape[0]}], got {tuple(tile_types.shape)}')
        if not torch.all((tile_types == 0) | (tile_types == 1)):
            raise ValueError('tile_types only supports 0 (regular tile) and 1 (thumbnail)')

        encoded_boxes = self._coordinate_ablation(tile_boxes, tile_types)
        dense_encoding = self._encode_positions(encoded_boxes, h, w, dtype, device)

        type_encoding = self.tile_type_embedding(tile_types).to(dtype=dtype)[:, None, None, :]
        type_encoding = type_encoding.expand(-1, h, w, -1)
        thumbnail_mask = tile_types.bool()[:, None, None, None]
        if self.thumbnail_mode == 'full':
            delta = dense_encoding + type_encoding
        elif self.thumbnail_mode == 'type_only':
            delta = torch.where(thumbnail_mask, type_encoding, dense_encoding + type_encoding)
        else:
            delta = torch.where(thumbnail_mask, torch.zeros_like(dense_encoding), dense_encoding + type_encoding)
        if self.coord_mode == 'disabled':
            # Multiplication rather than an early return keeps all parameters in
            # the DDP graph while removing both coordinate and tile-type signals.
            delta = delta * 0.
        if self.gate_logit is not None:
            gate = torch.sigmoid(self.gate_logit).to(dtype=dtype)
            if gate.requires_grad:
                gate.register_hook(self._sanitize_gate_gradient)
            delta = delta * gate
        return delta


def _extract_feature_with_gosa(self, pixel_values, tile_coords=None, tile_types=None):
    if self.select_layer == -1:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values, output_hidden_states=False, return_dict=True).last_hidden_state
    else:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values, output_hidden_states=True, return_dict=True).hidden_states[self.select_layer]
    vit_embeds = vit_embeds[:, 1:, :]
    h = w = int(vit_embeds.shape[1]**0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
    current_h, current_w = vit_embeds.shape[1:3]
    if self.enable_local_pe:
        local_encoding = self.pos_embed_2d(current_h, current_w, vit_embeds.dtype, vit_embeds.device)
        vit_embeds = vit_embeds + local_encoding
    if self.enable_global_pe:
        if tile_coords is None:
            warnings.warn(
                'ENABLE_GLOBAL_PE=True but tile coordinates were not supplied; using full-page geometry.',
                stacklevel=2)
            tile_coords = vit_embeds.new_tensor([[0., 0., 1., 1.]]).expand(vit_embeds.shape[0], -1)
        global_encoding = self.global_spatial_encoder(
            tile_coords, tile_types, current_h, current_w, vit_embeds.dtype, vit_embeds.device)
        vit_embeds = vit_embeds + global_encoding
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    return self.mlp1(vit_embeds)


def _load_component_state(module: nn.Module, model_dir: str, component: str) -> bool:
    state = {}
    suffix = f'{component}.'
    for weight_file in sorted(Path(model_dir).glob('model*.safetensors')):
        from safetensors import safe_open
        with safe_open(str(weight_file), framework='pt', device='cpu') as reader:
            for key in reader.keys():
                marker = key.find(suffix)
                if marker >= 0:
                    state[key[marker + len(suffix):]] = reader.get_tensor(key)
    if not state:
        return False
    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f'Invalid {component} checkpoint state: {incompatible}')
    return True


def attach_gosa(model: nn.Module, model_dir: str | None = None) -> bool:
    """Attach GOSA to a loaded baseline model when an environment flag is set."""
    enable_local = os.environ.get('ENABLE_LOCAL_PE', 'False').lower() == 'true'
    enable_global = os.environ.get('ENABLE_GLOBAL_PE', 'False').lower() == 'true'
    model.enable_local_pe = enable_local
    model.enable_global_pe = enable_global
    if not (enable_local or enable_global):
        return False
    if hasattr(model, 'pos_embed_2d') or hasattr(model, 'global_spatial_encoder'):
        # Transitional compatibility for other local InternVL sizes that have
        # not yet been migrated to a clean baseline source.
        warnings.warn(
            'This pretrained InternVL implementation already contains GOSA; '
            'move that implementation to swift.model.internvl_extensions.',
            stacklevel=2)
        return False

    feature_dim = model.config.vision_config.hidden_size * int(1 / model.downsample_ratio)**2
    anchor = next(model.mlp1.parameters())
    if enable_local:
        model.pos_embed_2d = Learnable2DPositionalEncoding(feature_dim).to(
            device=anchor.device, dtype=anchor.dtype)
        if model_dir:
            _load_component_state(model.pos_embed_2d, model_dir, 'pos_embed_2d')
    if enable_global:
        model.global_spatial_encoder = GlobalTilePositionalEncoding(
            feature_dim,
            fusion=os.environ.get('GLOBAL_PE_FUSION', 'add').lower(),
            gate_init=float(os.environ.get('GLOBAL_PE_GATE_INIT', '0.1')),
            thumbnail_mode=os.environ.get('GLOBAL_PE_THUMBNAIL', 'full').lower(),
            coord_mode=os.environ.get('GLOBAL_PE_COORD_MODE', 'real').lower(),
            coord_seed=int(os.environ.get('GLOBAL_PE_COORD_SEED', '0')),
        ).to(device=anchor.device, dtype=anchor.dtype)
        if model_dir:
            _load_component_state(model.global_spatial_encoder, model_dir, 'global_spatial_encoder')
    model.extract_feature = MethodType(_extract_feature_with_gosa, model)
    model.config.enable_local_pe = enable_local
    model.config.enable_global_pe = enable_global
    model.config.global_pe_coord_mode = os.environ.get('GLOBAL_PE_COORD_MODE', 'real').lower()
    model.config.global_pe_coord_seed = int(os.environ.get('GLOBAL_PE_COORD_SEED', '0'))
    return True
