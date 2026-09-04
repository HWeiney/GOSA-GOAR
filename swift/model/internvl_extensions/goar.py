"""GOSA-guided OCR anchor refinement for InternVL.

GOAR aligns OCR-prompt line embeddings with GOSA visual tokens, predicts a
visual box and an OCR-anchor residual box, and fuses them with learned
heteroscedastic precision. The fused code is injected back into the prompt so
the normal autoregressive ``<ref>...<box>...`` interface is preserved.
"""

from __future__ import annotations

import math
import os
from types import MethodType
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, str(default)).strip().lower()
    if value in ('true', '1', 'yes'):
        return True
    if value in ('false', '0', 'no'):
        return False
    raise ValueError(f'{name} must be a boolean, got: {value}')


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    xy1, xy2 = boxes[..., :2], boxes[..., 2:]
    return torch.cat([(xy1 + xy2) * 0.5, (xy2 - xy1).clamp_min(1e-4)], dim=-1)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center, size = boxes[..., :2], boxes[..., 2:].clamp_min(1e-4)
    return torch.cat([center - size * 0.5, center + size * 0.5], dim=-1).clamp(0., 1.)


def _generalized_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection = (rb - lt).clamp_min(0.).prod(dim=-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(0.).prod(dim=-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(0.).prod(dim=-1)
    union = area1 + area2 - intersection
    iou = intersection / union.clamp_min(1e-6)
    cover_lt = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    cover_rb = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    cover = (cover_rb - cover_lt).clamp_min(0.).prod(dim=-1)
    return iou - (cover - union) / cover.clamp_min(1e-6)


def _pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (rb - lt).clamp_min(0.).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0.).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0.).prod(dim=-1)
    return intersection / (area1[:, None] + area2[None, :] - intersection).clamp_min(1e-6)


class GOSAOCRAnchorRefinement(nn.Module):
    """End-to-end OCR anchor refinement and prompt-space fusion."""

    INFERENCE_BOX_MODES = ('fusion', 'visual', 'ocr')
    SOURCE_FUSION_MODES = ('uncertainty', 'mean')

    def __init__(self, hidden_size: int, bottleneck: int = 256, max_ocr_lines: int = 128,
                 pointer_loss_weight: float = 0.2, branch_loss_weight: float = 0.25,
                 uncertainty_loss_weight: float = 0.1, loss_weight: float = 1.0,
                 inference_box_mode: str = 'fusion', use_roi_feature: bool = True,
                 source_fusion_mode: str = 'uncertainty', refine_ocr_anchor: bool = True):
        super().__init__()
        self.max_ocr_lines = max_ocr_lines
        self.pointer_loss_weight = pointer_loss_weight
        self.branch_loss_weight = branch_loss_weight
        self.uncertainty_loss_weight = uncertainty_loss_weight
        self.loss_weight = loss_weight
        self.inference_box_mode = self._validate_inference_box_mode(inference_box_mode)
        self.use_roi_feature = use_roi_feature
        self.source_fusion_mode = self._validate_source_fusion_mode(source_fusion_mode)
        self.refine_ocr_anchor = bool(refine_ocr_anchor)
        self.text_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, bottleneck))
        self.visual_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, bottleneck))
        self.question_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, bottleneck))
        self.geometry_proj = nn.Sequential(nn.Linear(5, bottleneck), nn.GELU(), nn.Linear(bottleneck, bottleneck))
        self.fusion = nn.Sequential(
            nn.LayerNorm(bottleneck * 3), nn.Linear(bottleneck * 3, bottleneck), nn.GELU())
        self.pointer_key = nn.Linear(bottleneck, bottleneck, bias=False)
        self.pointer_query = nn.Linear(bottleneck, bottleneck, bias=False)
        self.visual_box_head = nn.Sequential(nn.Linear(bottleneck * 2, bottleneck), nn.GELU(), nn.Linear(bottleneck, 4))
        self.anchor_delta_head = nn.Sequential(nn.Linear(bottleneck * 2, bottleneck), nn.GELU(), nn.Linear(bottleneck, 4))
        self.precision_head = nn.Sequential(nn.Linear(bottleneck * 2, bottleneck), nn.GELU(), nn.Linear(bottleneck, 2))
        self.spatial_out = nn.Sequential(nn.Linear(4, bottleneck), nn.GELU(), nn.Linear(bottleneck, hidden_size))
        self.evidence_out = nn.Linear(bottleneck, hidden_size)
        self.injection_gate_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        # Exact no-op at initialization keeps the first step baseline-compatible.
        nn.init.zeros_(self.spatial_out[-1].weight)
        nn.init.zeros_(self.spatial_out[-1].bias)
        nn.init.zeros_(self.evidence_out.weight)
        nn.init.zeros_(self.evidence_out.bias)

    @classmethod
    def _validate_inference_box_mode(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in cls.INFERENCE_BOX_MODES:
            choices = ', '.join(cls.INFERENCE_BOX_MODES)
            raise ValueError(f'GOAR_INFERENCE_BOX_MODE must be one of {choices}, got: {value}')
        return value

    @classmethod
    def _validate_source_fusion_mode(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in cls.SOURCE_FUSION_MODES:
            choices = ', '.join(cls.SOURCE_FUSION_MODES)
            raise ValueError(f'GOAR_SOURCE_FUSION_MODE must be one of {choices}, got: {value}')
        return value

    @staticmethod
    def _token_centers(tile_boxes: torch.Tensor, tokens_per_tile: int) -> torch.Tensor:
        side = int(math.sqrt(tokens_per_tile))
        if side * side != tokens_per_tile:
            raise ValueError(f'GOAR requires square visual grids, got {tokens_per_tile} tokens per tile')
        axis = (torch.arange(side, device=tile_boxes.device, dtype=torch.float32) + 0.5) / side
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        local = torch.stack([x, y], dim=-1).reshape(1, tokens_per_tile, 2)
        return (tile_boxes[:, None, :2] + local * (tile_boxes[:, None, 2:] - tile_boxes[:, None, :2])).reshape(-1, 2)

    @classmethod
    def _roi_pool(cls, visual: torch.Tensor, tile_boxes: torch.Tensor, line_boxes: torch.Tensor) -> torch.Tensor:
        flat_visual = visual.reshape(-1, visual.shape[-1])
        centers = cls._token_centers(tile_boxes, visual.shape[1])
        inside = ((centers[None] >= line_boxes[:, None, :2]) &
                  (centers[None] <= line_boxes[:, None, 2:])).all(dim=-1)
        weights = inside.to(dtype=flat_visual.dtype)
        missing = weights.sum(dim=-1) == 0
        if missing.any():
            line_centers = (line_boxes[missing, :2] + line_boxes[missing, 2:]) * 0.5
            distance = (line_centers[:, None] - centers[None]).square().sum(dim=-1)
            nearest = distance.topk(k=min(4, centers.shape[0]), largest=False).indices
            fallback = torch.zeros_like(weights[missing])
            fallback.scatter_(1, nearest, 1.)
            weights[missing] = fallback
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.)
        return weights @ flat_visual

    @staticmethod
    def _valid_box(values: Sequence, device: torch.device) -> torch.Tensor | None:
        try:
            box = torch.as_tensor(values, dtype=torch.float32, device=device).reshape(4) / 1000.
        except (TypeError, ValueError, RuntimeError):
            return None
        if not torch.isfinite(box).all() or (box[2:] <= box[:2]).any():
            return None
        return box.clamp(0., 1.)

    @staticmethod
    def _prompt_state(embeds: torch.Tensor, labels: torch.Tensor | None, excluded: torch.Tensor):
        mask = ~excluded if labels is None else (labels == -100) & ~excluded
        indices = mask.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            indices = (~excluded).nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            indices = torch.arange(embeds.shape[0], device=embeds.device)
        return embeds[indices].mean(dim=0), int(indices[-1])

    def _sample_forward(self, embeds: torch.Tensor, labels: torch.Tensor | None, data: Dict[str, Any],
                        visual: torch.Tensor, tile_boxes: torch.Tensor, visual_mask: torch.Tensor | None = None):
        record, spans = data.get('ocr', {}), data.get('line_spans', [])
        bboxes, confidences = record.get('bboxes', []), record.get('confidences', [])
        offset = int(data.get('padding_offset', 0))
        lines = []
        for i, span in enumerate(spans[:self.max_ocr_lines]):
            if i >= len(bboxes) or len(span) != 2:
                break
            box = self._valid_box(bboxes[i], embeds.device)
            start, end = int(span[0]) + offset, int(span[1]) + offset
            if box is None or start < 0 or end <= start or end > embeds.shape[0]:
                continue
            confidence = float(confidences[i]) if i < len(confidences) else 0.
            lines.append((start, end, box, max(0., min(1., confidence))))
        if visual.numel() == 0:
            return embeds, None, None

        # A genuinely empty OCR cache is a supported input condition, not a
        # span-matching failure.  GOAR still has an OCR-independent visual box
        # branch, so use it as the structural fallback and supervise it with
        # the same final/branch objectives that it receives in the fused path.
        # Conversely, a non-empty OCR record with no usable lines returns no
        # box so the template's integrity guard can expose malformed/truncated
        # spans instead of silently degrading to visual-only execution.
        if not lines:
            has_ocr_rows = any(record.get(key) for key in ('words', 'bboxes', 'confidences'))
            if has_ocr_rows:
                return embeds, None, None
            excluded = torch.zeros(embeds.shape[0], dtype=torch.bool, device=embeds.device)
            if visual_mask is not None:
                excluded |= visual_mask
            question, injection_index = self._prompt_state(embeds, labels, excluded)
            question_code = self.question_proj(question)
            global_visual = self.visual_proj(visual.mean(dim=(0, 1)))
            visual_params = self.visual_box_head(torch.cat([question_code, global_visual]))
            visual_cxcywh = torch.cat([visual_params[:2].sigmoid(), visual_params[2:].sigmoid()])
            visual_box = _cxcywh_to_xyxy(visual_cxcywh)

            gate = self.injection_gate_logit.sigmoid().to(dtype=embeds.dtype)
            spatial_delta = self.spatial_out(
                visual_box.to(dtype=self.spatial_out[0].weight.dtype)).to(dtype=embeds.dtype)
            output = embeds.clone()
            output[injection_index] += gate * spatial_delta

            target = self._valid_box(data.get('target', []), embeds.device)
            if target is None:
                return output, None, visual_box.detach()
            visual_loss = F.l1_loss(visual_box, target) + 1. - _generalized_iou(visual_box, target)
            loss = visual_loss * (1. + self.branch_loss_weight) * self.loss_weight
            return output, loss, visual_box.detach()

        excluded = torch.zeros(embeds.shape[0], dtype=torch.bool, device=embeds.device)
        if visual_mask is not None:
            excluded |= visual_mask
        text_states, boxes, conf = [], [], []
        for start, end, box, confidence in lines:
            excluded[start:end] = True
            text_states.append(embeds[start:end].mean(dim=0))
            boxes.append(box)
            conf.append(confidence)
        boxes = torch.stack(boxes)
        conf = boxes.new_tensor(conf).unsqueeze(-1)
        question, injection_index = self._prompt_state(embeds, labels, excluded)
        text_code = self.text_proj(torch.stack(text_states))
        visual_code = self.visual_proj(self._roi_pool(visual, tile_boxes, boxes))
        if not self.use_roi_feature:
            # Preserve the architecture and parameter budget while removing
            # only the per-line GOSA ROI observation for the structural ablation.
            visual_code = torch.zeros_like(visual_code)
        geometry = torch.cat([boxes, conf], dim=-1).to(dtype=self.geometry_proj[0].weight.dtype)
        geometry_code = self.geometry_proj(geometry)
        line_code = self.fusion(torch.cat([text_code, visual_code, geometry_code], dim=-1))
        question_code = self.question_proj(question)
        scores = self.pointer_key(line_code) @ self.pointer_query(question_code) / math.sqrt(line_code.shape[-1])
        pointer = scores.softmax(dim=0)
        evidence = (pointer[:, None] * line_code).sum(dim=0)
        global_visual = self.visual_proj(visual.mean(dim=(0, 1)))

        visual_params = self.visual_box_head(torch.cat([question_code, global_visual]))
        visual_cxcywh = torch.cat([visual_params[:2].sigmoid(), visual_params[2:].sigmoid()])
        anchor_cxcywh = (pointer[:, None] * _xyxy_to_cxcywh(boxes)).sum(dim=0)
        delta = self.anchor_delta_head(torch.cat([question_code, evidence]))
        corrected_cxcywh = torch.cat([
            anchor_cxcywh[:2] + 0.1 * delta[:2].tanh(),
            anchor_cxcywh[2:] * torch.exp(0.5 * delta[2:].tanh()),
        ]).clamp(1e-4, 1.)
        if self.refine_ocr_anchor:
            refined_cxcywh = corrected_cxcywh
        else:
            # Keep the residual head in the graph for parameter-matched DDP,
            # while injecting and supervising the selected raw OCR anchor.
            refined_cxcywh = anchor_cxcywh + 0. * corrected_cxcywh
        log_precision = self.precision_head(torch.cat([question_code, evidence])).clamp(-5., 5.)
        source_weights = log_precision.softmax(dim=0)
        if self.source_fusion_mode == 'mean':
            # Preserve the precision branch and its auxiliary supervision while
            # replacing only learned source weighting with a fixed 50/50 mean.
            source_weights = 0. * source_weights + .5
        final_cxcywh = source_weights[0] * visual_cxcywh + source_weights[1] * refined_cxcywh
        visual_box, refined_box = _cxcywh_to_xyxy(visual_cxcywh), _cxcywh_to_xyxy(refined_cxcywh)
        final_box = _cxcywh_to_xyxy(final_cxcywh)

        if self.training and self.inference_box_mode != 'fusion':
            raise ValueError('GOAR visual/ocr box modes are inference-only; train with fusion mode')
        injection_box = {
            'fusion': final_box,
            'visual': visual_box,
            'ocr': refined_box,
        }[self.inference_box_mode]

        gate = self.injection_gate_logit.sigmoid().to(dtype=embeds.dtype)
        spatial_delta = self.spatial_out(
            injection_box.to(dtype=self.spatial_out[0].weight.dtype)).to(dtype=embeds.dtype)
        evidence_delta = self.evidence_out(evidence).to(dtype=embeds.dtype)
        output = embeds.clone()
        output[injection_index] += gate * (spatial_delta + evidence_delta)
        for weight, (start, end, _, _) in zip(pointer, lines):
            output[start:end] += gate * weight.to(dtype=embeds.dtype) * evidence_delta

        target = self._valid_box(data.get('target', []), embeds.device)
        if target is None:
            return output, None, injection_box.detach()
        final_loss = F.l1_loss(final_box, target) + 1. - _generalized_iou(final_box, target)
        branch_loss = (F.l1_loss(visual_box, target) + 1. - _generalized_iou(visual_box, target)
                       + F.l1_loss(refined_box, target) + 1. - _generalized_iou(refined_box, target))
        target_ious = _pairwise_iou(boxes, target[None]).squeeze(-1)
        if target_ious.max() >= 0.05:
            pointer_loss = F.cross_entropy(scores[None], target_ious.argmax()[None])
        else:
            # Do not force a wrong OCR row when OCR entirely missed the target.
            pointer_loss = scores.sum() * 0.
        errors = torch.stack([(visual_box - target).abs().mean(), (refined_box - target).abs().mean()])
        uncertainty_loss = (errors * log_precision.exp() - log_precision).mean()
        loss = (final_loss + self.branch_loss_weight * branch_loss + self.pointer_loss_weight * pointer_loss
                + self.uncertainty_loss_weight * uncertainty_loss) * self.loss_weight
        return output, loss, injection_box.detach()

    def forward(self, inputs_embeds: torch.Tensor, goar_data: List[Dict[str, Any]], visual_features: torch.Tensor,
                tile_coords: torch.Tensor, labels: torch.Tensor | None = None,
                visual_mask: torch.Tensor | None = None):
        outputs, losses, boxes, tile_cursor = [], [], [], 0
        for batch_idx, data in enumerate(goar_data):
            num_tiles = int(data.get('num_tiles', 0))
            sample_labels = labels[batch_idx] if labels is not None else None
            output, loss, box = self._sample_forward(
                inputs_embeds[batch_idx], sample_labels, data,
                visual_features[tile_cursor:tile_cursor + num_tiles],
                tile_coords[tile_cursor:tile_cursor + num_tiles],
                visual_mask[batch_idx] if visual_mask is not None else None)
            tile_cursor += num_tiles
            outputs.append(output)
            if loss is not None:
                losses.append(loss)
            boxes.append(box)
        return torch.stack(outputs), (torch.stack(losses).mean() if losses else None), boxes


def find_goar_host(model: nn.Module) -> nn.Module | None:
    visited, queue = set(), [model]
    while queue:
        current = queue.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))
        # PEFT wrappers delegate unknown attributes to their inner model. Using
        # hasattr() here can therefore return the outer PeftModel even though
        # goar_adapter and forward_with_goar live on the inner InternVL model.
        # Aux loss would then be written to one object and read from another.
        if 'goar_adapter' in current._modules:
            return current
        for name in ('model', 'base_model'):
            child = current._modules.get(name)
            if isinstance(child, nn.Module):
                queue.append(child)
    return None


def get_active_goar_adapter(host: nn.Module) -> GOSAOCRAnchorRefinement:
    """Return PEFT's trainable modules_to_save copy instead of its frozen original."""
    adapter = host._modules['goar_adapter']
    modules_to_save = getattr(adapter, 'modules_to_save', None)
    active_adapters = getattr(adapter, 'active_adapters', None)
    if modules_to_save is None or not active_adapters:
        return adapter
    if len(active_adapters) != 1:
        raise RuntimeError(f'GOAR requires exactly one active PEFT adapter, got: {active_adapters}')
    active = active_adapters[0]
    if active not in modules_to_save:
        raise RuntimeError(f'Active PEFT adapter {active!r} has no trainable GOAR modules_to_save copy')
    return modules_to_save[active]


def attach_goar(model: nn.Module) -> bool:
    if os.environ.get('ENABLE_GOAR', 'False').lower() != 'true':
        return False
    if not (getattr(model, 'enable_global_pe', False) and os.environ.get('OCR_MODE', '').lower() == 'prompt'):
        raise ValueError('ENABLE_GOAR=True requires ENABLE_GLOBAL_PE=True and OCR_MODE=prompt')
    if hasattr(model, 'goar_adapter'):
        return False
    llm_config = getattr(model.config, 'llm_config', model.language_model.config)
    anchor = next(model.language_model.parameters())
    model.goar_adapter = GOSAOCRAnchorRefinement(
        hidden_size=int(llm_config.hidden_size), bottleneck=int(os.environ.get('GOAR_BOTTLENECK', '256')),
        max_ocr_lines=int(os.environ.get('GOAR_MAX_OCR_LINES', '128')),
        pointer_loss_weight=float(os.environ.get('GOAR_POINTER_LOSS_WEIGHT', '0.2')),
        branch_loss_weight=float(os.environ.get('GOAR_BRANCH_LOSS_WEIGHT', '0.25')),
        uncertainty_loss_weight=float(os.environ.get('GOAR_UNCERTAINTY_LOSS_WEIGHT', '0.1')),
        loss_weight=float(os.environ.get('GOAR_LOSS_WEIGHT', '1.0')),
        inference_box_mode=os.environ.get('GOAR_INFERENCE_BOX_MODE', 'fusion'),
        use_roi_feature=_env_bool('GOAR_USE_ROI_FEATURE', True),
        source_fusion_mode=os.environ.get('GOAR_SOURCE_FUSION_MODE', 'uncertainty'),
        refine_ocr_anchor=_env_bool('GOAR_REFINE_OCR_ANCHOR', True),
    ).to(device=anchor.device, dtype=anchor.dtype)
    model.config.enable_goar = True
    model._goar_forward_origin = model.forward

    def forward_with_goar(self, *args, **kwargs):
        return self._goar_forward_origin(*args, **kwargs)

    model.forward = MethodType(forward_with_goar, model)
    return True
