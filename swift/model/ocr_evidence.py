# Copyright (c) ModelScope Contributors. All rights reserved.
"""Lightweight OCR evidence encoder for InternVL.

The module intentionally consumes cached OCR predictions rather than images.  It
does not depend on, or modify, the implementation of LiLT shipped by
Transformers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import nn
from transformers import (AutoTokenizer, Blip2QFormerConfig, Blip2QFormerModel,
                          LiltModel)


EMPTY_OCR = {'words': [], 'bboxes': [], 'confidences': []}


def load_ocr_encoder_state(encoder: nn.Module, model_dir: str) -> bool:
    """Restore OCR-only keys that HF saw before the branch was attached.

    Full-training checkpoints are normally safetensors. Reading keys one by
    one avoids materializing the InternVL weights a second time.
    """
    prefix = 'ocr_evidence_encoder.'
    state = {}
    for weight_file in sorted(Path(model_dir).glob('model*.safetensors')):
        from safetensors import safe_open
        with safe_open(str(weight_file), framework='pt', device='cpu') as reader:
            for key in reader.keys():
                if key.startswith(prefix):
                    state[key[len(prefix):]] = reader.get_tensor(key)
    if not state:
        return False
    incompatible = encoder.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f'Unexpected OCR checkpoint keys: {incompatible.unexpected_keys}')
    # Missing LiLT keys are valid for adapter-like OCR-only checkpoints, but all
    # trainable heads must be present when any OCR state was found.
    required_prefixes = ('confidence_mlp.', 'query_tokens.', 'qformer.', 'projector.')
    for required in required_prefixes:
        if not any(key.startswith(required) for key in state):
            raise ValueError(f'Incomplete OCR checkpoint: no key starts with {required!r}.')
    return True


class OCREvidenceEncoder(nn.Module):
    """LiLT + BLIP-2 Q-Former OCR evidence branch.

    Args:
        lilt_path: Local LiLT-InfoXLM checkpoint. Network model identifiers are
            deliberately rejected by using ``local_files_only=True``.
        llm_hidden_size: Read from the owning InternVL config by the loader.
    """

    def __init__(self,
                 lilt_path: str,
                 llm_hidden_size: int,
                 num_queries: int = 64,
                 max_ocr_length: int = 512,
                 unfreeze_lilt_top_layers: int = 0):
        super().__init__()
        self.hidden_size = 768
        self.num_queries = num_queries
        self.max_ocr_length = max_ocr_length
        self.tokenizer = AutoTokenizer.from_pretrained(lilt_path, use_fast=True, local_files_only=True)
        if not self.tokenizer.is_fast:
            raise ValueError('OCREvidenceEncoder requires a fast tokenizer to align words and subwords.')
        self.lilt = LiltModel.from_pretrained(lilt_path, local_files_only=True)
        if self.lilt.config.hidden_size != self.hidden_size:
            raise ValueError(f'Expected LiLT hidden_size=768, got {self.lilt.config.hidden_size}.')

        # The confidence branch is deliberately small; its output is added to
        # every aligned LiLT subword state.
        self.confidence_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, self.hidden_size),
        )
        qformer_config = Blip2QFormerConfig(
            hidden_size=self.hidden_size,
            encoder_hidden_size=self.hidden_size,
            num_hidden_layers=4,
            num_attention_heads=12,
            cross_attention_frequency=1,
        )
        # Config-only construction: no BLIP-2 checkpoint is downloaded.
        self.qformer = Blip2QFormerModel(qformer_config)
        self.query_tokens = nn.Embedding(num_queries, self.hidden_size)
        nn.init.normal_(self.query_tokens.weight, mean=0.0, std=qformer_config.initializer_range)
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, int(llm_hidden_size)),
        )
        self.set_lilt_trainable_layers(unfreeze_lilt_top_layers)

    def set_lilt_trainable_layers(self, top_layers: int = 0) -> None:
        """Freeze LiLT, optionally unfreezing its highest encoder layers."""
        self.lilt.requires_grad_(False)
        layers = self.lilt.encoder.layer
        if top_layers < 0 or top_layers > len(layers):
            raise ValueError(f'unfreeze_lilt_top_layers must be in [0, {len(layers)}], got {top_layers}')
        if top_layers:
            for layer in layers[-top_layers:]:
                layer.requires_grad_(True)

    @staticmethod
    def normalize_bbox(bbox: Sequence[float], width: float, height: float) -> List[int]:
        """Normalize an xyxy pixel box to LiLT's inclusive [0, 1000] space."""
        if width <= 0 or height <= 0 or len(bbox) != 4:
            raise ValueError(f'Invalid bbox/image size: bbox={bbox}, width={width}, height={height}')
        x1, y1, x2, y2 = bbox
        normalized = [round(1000 * x1 / width), round(1000 * y1 / height),
                      round(1000 * x2 / width), round(1000 * y2 / height)]
        normalized = [max(0, min(1000, int(v))) for v in normalized]
        normalized[0], normalized[2] = sorted((normalized[0], normalized[2]))
        normalized[1], normalized[3] = sorted((normalized[1], normalized[3]))
        return normalized

    @staticmethod
    def _validate_record(record: Dict) -> Dict:
        record = record or EMPTY_OCR
        words = [str(word) for word in record.get('words', [])]
        bboxes = record.get('bboxes', [])
        confidences = record.get('confidences', [])
        if not (len(words) == len(bboxes) == len(confidences)):
            raise ValueError('OCR words, bboxes and confidences must have equal lengths.')
        clean_words, clean_boxes, clean_confidences = [], [], []
        for word, bbox, confidence in zip(words, bboxes, confidences):
            if not word.strip():
                continue
            if len(bbox) != 4:
                raise ValueError(f'OCR bbox must be xyxy with four values, got {bbox}')
            bbox = [max(0, min(1000, int(round(value)))) for value in bbox]
            bbox[0], bbox[2] = sorted((bbox[0], bbox[2]))
            bbox[1], bbox[3] = sorted((bbox[1], bbox[3]))
            clean_words.append(word)
            clean_boxes.append(bbox)
            clean_confidences.append(max(0., min(1., float(confidence))))
        return {'words': clean_words, 'bboxes': clean_boxes, 'confidences': clean_confidences}

    def prepare_batch(self, records: Sequence[Dict], device: torch.device) -> Dict[str, torch.Tensor]:
        """Tokenize words and replicate each word's bbox/confidence to its subwords."""
        records = [self._validate_record(record) for record in records]
        # Fast tokenizers cannot encode an empty split-word list reliably. A
        # harmless dummy word is used internally and removed via ``has_ocr``.
        token_words = [record['words'] or [self.tokenizer.unk_token or '[UNK]'] for record in records]
        token_boxes = [record['bboxes'] or [[0, 0, 0, 0]] for record in records]
        encoded = self.tokenizer(
            token_words,
            boxes=token_boxes,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_ocr_length,
            return_tensors='pt',
        )
        batch_size, seq_len = encoded['input_ids'].shape
        # LayoutXLMTokenizerFast (used by LiLT-InfoXLM) performs the bbox
        # replication itself. Keep the word-id pass below for confidence.
        bbox = encoded.pop('bbox').to(dtype=torch.long)
        confidence = torch.zeros((batch_size, seq_len, 1), dtype=torch.float32)
        has_ocr = torch.tensor([bool(record['words']) for record in records], dtype=torch.bool)
        for batch_idx, record in enumerate(records):
            if not record['words']:
                continue
            for token_idx, word_idx in enumerate(encoded.word_ids(batch_index=batch_idx)):
                if word_idx is None:
                    continue
                confidence[batch_idx, token_idx, 0] = record['confidences'][word_idx]
        return {
            'input_ids': encoded['input_ids'].to(device),
            'attention_mask': encoded['attention_mask'].to(device),
            'bbox': bbox.to(device),
            'confidence': confidence.to(device),
            'has_ocr': has_ocr.to(device),
        }

    def forward(self, records: Sequence[Dict]) -> torch.Tensor:
        # PEFT wraps modules listed in ``modules_to_save`` with a
        # ModulesToSaveWrapper during stage-2 LoRA training. Treat every
        # component as a generic nn.Module instead of indexing Sequential or
        # reading Embedding.weight through the wrapper.
        query_parameter = next(self.query_tokens.parameters())
        device = query_parameter.device
        batch = self.prepare_batch(records, device)
        lilt_outputs = self.lilt(
            input_ids=batch['input_ids'],
            bbox=batch['bbox'],
            attention_mask=batch['attention_mask'],
            return_dict=True,
        )
        encoder_states = lilt_outputs.last_hidden_state
        confidence_dtype = next(self.confidence_mlp.parameters()).dtype
        confidence_delta = self.confidence_mlp(batch['confidence'].to(confidence_dtype))
        encoder_states = encoder_states + confidence_delta.to(encoder_states.dtype)
        query_ids = torch.arange(self.num_queries, device=device)
        query_vectors = self.query_tokens(query_ids)
        queries = query_vectors.unsqueeze(0).expand(len(records), -1, -1).to(encoder_states.dtype)
        # Pass query embeddings positionally: PEFT's auxiliary wrapper reserves
        # the first positional argument as ``x`` before forwarding to Q-Former.
        query_states = self.qformer(
            queries,
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=batch['attention_mask'],
            return_dict=True,
        ).last_hidden_state
        projector_dtype = next(self.projector.parameters()).dtype
        projected = self.projector(query_states.to(projector_dtype))
        # No OCR is a valid input. Its evidence slots contain no signal and do
        # not accidentally act as 64 learned soft-prompt tokens.
        projected = projected * batch['has_ocr'][:, None, None].to(projected.dtype)
        return projected
