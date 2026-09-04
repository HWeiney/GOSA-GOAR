# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import hashlib
import os
import random
import re
import torch
from functools import lru_cache
from functools import partial
from torch import nn
from typing import Any, Dict, List, Literal

from swift.utils import get_env_args, is_deepspeed_enabled
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, findall
from ..vision_utils import load_video_internvl, transform_image
from .llm import GptOssTemplateMeta, GptTemplate
from .microsoft import Phi3TemplateMeta
from .utils import ChatmlTemplateMeta


# Internal embedding placeholder. It is replaced before the language model is
# called, so no tokenizer vocabulary or frozen LLM embedding has to be changed.
OCR_DUMMY_TOKEN_INDEX = -101


@lru_cache(maxsize=8192)
def _read_ocr_cache(path: str) -> Dict[str, Any]:
    with open(path, encoding='utf-8') as f:
        record = json.load(f)
    return {
        'words': record.get('words', []),
        'bboxes': record.get('bboxes', []),
        'confidences': record.get('confidences', []),
    }


OCR_PERTURB_MODES = {'clean', 'box', 'confidence', 'joint'}


def _record_rng(record: Dict[str, Any], seed: int, stream: str) -> random.Random:
    """Return a process/order-independent RNG for one OCR page and stream."""
    payload = json.dumps({
        'words': record.get('words', []),
        'bboxes': record.get('bboxes', []),
        'confidences': record.get('confidences', []),
    }, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    digest = hashlib.blake2b(f'{seed}:{stream}:{payload}'.encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, byteorder='big', signed=False))


def _noisy_interval(left: float, right: float, rng: random.Random, std: float) -> tuple[float, float]:
    left, right = sorted((left + rng.gauss(0., std), right + rng.gauss(0., std)))
    left, right = max(0., min(1000., left)), max(0., min(1000., right))
    if right - left < 1.:
        center = (left + right) / 2.
        left, right = max(0., center - .5), min(1000., center + .5)
        if right - left < 1.:
            left, right = (0., 1.) if center < 500. else (999., 1000.)
    return left, right


def perturb_ocr_record(record: Dict[str, Any], mode: str, box_std: float, confidence_std: float,
                       seed: int) -> Dict[str, Any]:
    """Apply deterministic, page-clustered OCR anchor corruption."""
    if mode not in OCR_PERTURB_MODES:
        raise ValueError(f'Unsupported OCR_PERTURB_MODE={mode!r}; expected one of {sorted(OCR_PERTURB_MODES)}')
    if box_std < 0. or confidence_std < 0.:
        raise ValueError('OCR perturbation standard deviations must be non-negative.')
    words = list(record.get('words', []))
    boxes = [list(box) for box in record.get('bboxes', [])]
    confidences = [float(value) for value in record.get('confidences', [])]
    if not (len(words) == len(boxes) == len(confidences)):
        raise ValueError('OCR words, bboxes and confidences must have equal lengths before perturbation.')
    if mode in {'box', 'joint'} and box_std > 0.:
        rng = _record_rng(record, seed, 'box')
        perturbed_boxes = []
        for box in boxes:
            if len(box) != 4:
                raise ValueError(f'OCR bbox must have four coordinates, got: {box!r}')
            x1, x2 = _noisy_interval(float(box[0]), float(box[2]), rng, box_std)
            y1, y2 = _noisy_interval(float(box[1]), float(box[3]), rng, box_std)
            perturbed_boxes.append([x1, y1, x2, y2])
        boxes = perturbed_boxes
    if mode in {'confidence', 'joint'} and confidence_std > 0.:
        rng = _record_rng(record, seed, 'confidence')
        confidences = [max(0., min(1., value + rng.gauss(0., confidence_std))) for value in confidences]
    return {'words': words, 'bboxes': boxes, 'confidences': confidences}


class InternvlTemplate(Template):
    skip_prompt = False
    num_image_token = None
    placeholder_tokens = ['<IMG_CONTEXT>']
    support_padding_free = True

    def init_env_args(self):
        super().init_env_args()
        self.input_size = get_env_args('input_size', int, 448)
        self.max_num = get_env_args('max_num', int, 12)
        self.enable_goise = get_env_args('ENABLE_GLOBAL_PE', bool, False)
        self.enable_grounding_token_weight = get_env_args('ENABLE_GROUNDING_TOKEN_WEIGHT', bool, False)
        self.grounding_box_loss_weight = get_env_args('GROUNDING_BOX_LOSS_WEIGHT', float, 1.5)
        self.grounding_ref_loss_weight = get_env_args('GROUNDING_REF_LOSS_WEIGHT', float, 1.0)
        self.use_ocr = get_env_args('USE_OCR', bool, False)
        self.ocr_mode = os.environ.get('OCR_MODE', 'lilt_qformer' if self.use_ocr else 'none').strip().lower()
        if self.ocr_mode not in {'none', 'prompt', 'lilt_qformer'}:
            raise ValueError(f'Unsupported OCR_MODE={self.ocr_mode!r}; use none, prompt, or lilt_qformer.')
        self.ocr_num_queries = get_env_args('OCR_NUM_QUERIES', int, 64)
        self.ocr_perturb_mode = os.environ.get('OCR_PERTURB_MODE', 'clean').strip().lower()
        self.ocr_box_noise_std = get_env_args('OCR_BOX_NOISE_STD', float, 50.)
        self.ocr_confidence_noise_std = get_env_args('OCR_CONFIDENCE_NOISE_STD', float, .15)
        self.ocr_perturb_seed = get_env_args('OCR_PERTURB_SEED', int, 0)
        if self.ocr_perturb_mode not in OCR_PERTURB_MODES:
            raise ValueError(
                f'Unsupported OCR_PERTURB_MODE={self.ocr_perturb_mode!r}; '
                f'expected one of {sorted(OCR_PERTURB_MODES)}')
        if self.ocr_box_noise_std < 0. or self.ocr_confidence_noise_std < 0.:
            raise ValueError('OCR perturbation standard deviations must be non-negative.')
        self.enable_goar = get_env_args('ENABLE_GOAR', bool, False)
        if self.enable_goar and self.ocr_mode != 'prompt':
            raise ValueError('ENABLE_GOAR=True requires OCR_MODE=prompt.')

    @staticmethod
    def _empty_ocr_record() -> Dict[str, List]:
        return {'words': [], 'bboxes': [], 'confidences': []}

    def _get_ocr_record(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        has_ocr_source = 'ocr' in inputs.extra_kwargs or 'ocr_cache' in inputs.extra_kwargs
        if self.ocr_mode != 'none' and self.is_training and not has_ocr_source:
            raise ValueError(
                'OCR mode is enabled but neither `ocr` nor `ocr_cache` reached the template. '
                'Keep custom dataset columns with `--remove_unused_columns false`.')
        record = inputs.extra_kwargs.get('ocr')
        cache_path = inputs.extra_kwargs.get('ocr_cache')
        if record is None and cache_path:
            try:
                record = _read_ocr_cache(os.path.abspath(os.path.expanduser(cache_path)))
            except FileNotFoundError:
                # Missing/no OCR is explicitly supported; malformed existing
                # caches still fail loudly instead of being silently ignored.
                record = None
        if not record:
            return self._empty_ocr_record()
        clean_record = {
            'words': list(record.get('words', [])),
            'bboxes': list(record.get('bboxes', [])),
            'confidences': list(record.get('confidences', [])),
        }
        return perturb_ocr_record(
            clean_record, getattr(self, 'ocr_perturb_mode', 'clean'),
            getattr(self, 'ocr_box_noise_std', 50.),
            getattr(self, 'ocr_confidence_noise_std', .15),
            getattr(self, 'ocr_perturb_seed', 0))

    @staticmethod
    def _ocr_prompt(record: Dict[str, Any]) -> str:
        lines = []
        for word, bbox, confidence in zip(
                record['words'], record['bboxes'], record['confidences']):
            lines.append(f'{word} <box>[[{",".join(str(int(v)) for v in bbox)}]]</box> ({float(confidence):.3f})')
        body = '\n'.join(lines) if lines else '(empty)'
        return f'\n<ocr_start>\n{body}\n<ocr_end>\n'

    @staticmethod
    def _find_subsequence(values: List[int], pattern: List[int], start: int = 0) -> tuple[int, int] | None:
        if not pattern:
            return None
        for index in range(start, len(values) - len(pattern) + 1):
            if values[index:index + len(pattern)] == pattern:
                return index, index + len(pattern)
        return None

    def _find_ocr_line_spans(self, input_ids: List[int], record: Dict[str, Any]) -> List[List[int]]:
        spans, cursor = [], 0
        for word, bbox, confidence in zip(record['words'], record['bboxes'], record['confidences']):
            line = f'{word} <box>[[{",".join(str(int(v)) for v in bbox)}]]</box> ({float(confidence):.3f})'
            # Match the exact prompt context. Qwen's tokenizer merges the final
            # ``)`` with the following newline, so encoding the bare line makes
            # every otherwise valid OCR row fail subsequence matching.
            tokens = self.processor.encode(line + '\n', add_special_tokens=False)
            span = self._find_subsequence(input_ids, tokens, cursor)
            if span is None:
                # Preserve OCR-row alignment: later entries must not silently
                # inherit the wrong bbox when a line cannot be located.
                spans.append([-1, -1])
            else:
                spans.append([span[0], span[1]])
                cursor = span[1]
        return spans

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            image_context = ['<image>\n']
        else:
            image_context = ['<img>', [-100], '</img>\n']
        return image_context

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        idx_list = findall(input_ids, -100)
        pixel_values = None
        images = inputs.images
        if images:
            labels = encoded.get('labels')
            if self.num_image_token is None:
                self.num_image_token = int((self.input_size // 14)**2 * (0.5**2))
            pixel_values_images = [transform_image(image, self.input_size, self.max_num) for image in images]
            pixel_values = torch.cat(pixel_values_images, dim=0).to(self.model_info.torch_dtype)
            image_bs = pixel_values.shape[0]

            idx, idx2 = idx_list[0], idx_list[-1]  # remove [-100, -100]
            img_tokens: List[int] = self.processor.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * self.num_image_token * image_bs
            input_ids = input_ids[:idx] + img_tokens + input_ids[idx2 + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * len(img_tokens) + labels[idx2 + 1:]
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
        encoded['pixel_values'] = pixel_values
        return encoded

    def forward_context(self, model, inputs):
        model_name = model.language_model.__class__.__name__.lower()
        if self.padding_free and 'internlm2' in model_name:
            position_ids = inputs['position_ids']
            modeling_module = model.language_model.model.layers[0].attention.__class__
            return self._patch_flash_attention_forward(modeling_module, position_ids, use_new_func=True)
        else:
            return super().forward_context(model, inputs)

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        tile_coords = inputs.pop('tile_coords', None)
        tile_types = inputs.pop('tile_types', None)
        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        input_ids = inputs['input_ids']
        inputs_embeds = embedding(input_ids).to(device=device)
        pixel_values = inputs.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device)
            if tile_coords is not None:
                vit_embeds = model.extract_feature(
                    pixel_values,
                    tile_coords=tile_coords.to(device=device),
                    tile_types=tile_types.to(device=device) if tile_types is not None else None).to(device=device)
            else:
                vit_embeds = model.extract_feature(pixel_values).to(device=device)
            selected = (input_ids == self.processor.encode('<IMG_CONTEXT>', add_special_tokens=False)[0])
            inputs_embeds[selected] = vit_embeds.reshape(-1, vit_embeds.shape[-1]).to(dtype=inputs_embeds.dtype)
        elif is_deepspeed_enabled():
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), device=device, dtype=inputs_embeds.dtype)
            vit_embeds = model.extract_feature(dummy_pixel_values).to(device=device)
            inputs_embeds += vit_embeds.mean() * 0.
        return {'inputs_embeds': inputs_embeds}


register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.internvl,
        default_system='You are an AI assistant whose name is InternLM (书生·浦语).',
        template_cls=InternvlTemplate,
        auto_add_bos=True))
register_template(
    Phi3TemplateMeta(
        MLLMTemplateType.internvl_phi3,
        default_system='You are an AI assistant whose name is Phi-3.',
        template_cls=InternvlTemplate,
        auto_add_bos=True))


class Internvl2Template(InternvlTemplate):
    VIDEO_SEGMENTS = 8

    def init_env_args(self):
        super().init_env_args()
        self.video_max_num = get_env_args('video_max_num', int, 1)
        self.video_segments = get_env_args('video_segments', int, self.VIDEO_SEGMENTS)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        image_context = super().replace_tag('image', index, inputs)
        if media_type == 'image':
            return image_context
        elif media_type == 'video':
            load_video = partial(load_video_internvl, num_segments=self.video_segments)
            return self.replace_video2image(load_video, inputs, lambda i: [f'Frame{i + 1}: '] + image_context)

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<ref>{ref}</ref>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<box>[{bbox}]</box>']

    @staticmethod
    def _find_supervised_token_span(input_ids: List[int], labels: List[int], tokens: List[int]) -> int | None:
        if not tokens:
            return None
        for i in range(len(input_ids) - len(tokens) + 1):
            if input_ids[i:i + len(tokens)] == tokens and all(label != -100 for label in labels[i:i + len(tokens)]):
                return i
        return None

    @staticmethod
    def _init_loss_scale_from_labels(labels: List[int]) -> List[float]:
        return [1.0 if label != -100 else 0.0 for label in labels]

    def _apply_grounding_token_weights(self, encoded: Dict[str, Any], ref_text: str, box_text: str) -> bool:
        input_ids, labels = encoded['input_ids'], encoded['labels']
        box_tokens = self.processor.encode(box_text, add_special_tokens=False)
        box_start = self._find_supervised_token_span(input_ids, labels, box_tokens)
        if box_start is None:
            return False
        loss_scale = encoded.get('loss_scale')
        if loss_scale is None:
            loss_scale = self._init_loss_scale_from_labels(labels)
        else:
            loss_scale = list(loss_scale)
        ref_tokens = self.processor.encode(ref_text, add_special_tokens=False)
        ref_start = self._find_supervised_token_span(input_ids, labels, ref_tokens)
        if ref_start is not None:
            for i in range(ref_start, ref_start + len(ref_tokens)):
                loss_scale[i] *= float(self.grounding_ref_loss_weight)
        for i in range(box_start, box_start + len(box_tokens)):
            loss_scale[i] *= float(self.grounding_box_loss_weight)
        encoded['loss_scale'] = loss_scale
        return True

    def _ensure_loss_scale(self, encoded: Dict[str, Any]) -> None:
        if encoded.get('loss_scale') is None:
            encoded['loss_scale'] = self._init_loss_scale_from_labels(encoded['labels'])

    def _add_grounding_target(self, encoded: Dict[str, Any], inputs: StdTemplateInputs):
        if encoded.get('labels') is None:
            return None
        if self.enable_grounding_token_weight:
            self._ensure_loss_scale(encoded)
        pattern = re.compile(r'<ref>(.*?)</ref>\s*<box>\s*\[\[([^\]]+)\]\]\s*</box>', re.DOTALL)
        pair = None
        for message in inputs.messages:
            if message['role'] == 'assistant' and isinstance(message['content'], str):
                match = pattern.search(message['content'])
                if match is not None:
                    pair = match
                    break
        if pair is None:
            return None
        try:
            coords = [float(value.strip()) / 1000. for value in pair.group(2).split(',')]
        except ValueError:
            return None
        if len(coords) != 4:
            return None
        ref_text = pair.group(1)
        box_text = pair.group(2)
        if self.enable_grounding_token_weight:
            self._apply_grounding_token_weights(encoded, ref_text, box_text)
        return [coordinate * 1000. for coordinate in coords]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        ocr_record = self._get_ocr_record(inputs) if self.ocr_mode != 'none' else self._empty_ocr_record()
        if self.ocr_mode == 'prompt':
            for message in inputs.messages:
                if message['role'] == 'user' and isinstance(message['content'], str):
                    image_end = message['content'].rfind('<image>')
                    insert_at = image_end + len('<image>') if image_end >= 0 else 0
                    message['content'] = (message['content'][:insert_at] + self._ocr_prompt(ocr_record)
                                          + message['content'][insert_at:])
                    break
        encoded = super(InternvlTemplate, self)._encode(inputs)
        input_ids = encoded['input_ids']
        idx_list = findall(input_ids, -100)
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        images = inputs.images
        if images:
            has_video = bool(inputs.videos)
            if self.num_image_token is None:
                self.num_image_token = int((self.input_size // 14)**2 * (0.5**2))
            max_num = self.max_num
            if has_video:
                max_num = self.video_max_num
            if self.enable_goise:
                image_outputs = [
                    transform_image(image, self.input_size, max_num, return_tile_coords=True) for image in images
                ]
                pixel_values = [output[0] for output in image_outputs]
                encoded['tile_coords'] = torch.cat([output[1] for output in image_outputs], dim=0)
                encoded['tile_types'] = torch.cat([output[2] for output in image_outputs], dim=0)
            else:
                pixel_values = [transform_image(image, self.input_size, max_num) for image in images]
            num_patches = [pv.shape[0] for pv in pixel_values]
            pixel_values = torch.cat(pixel_values).to(self.model_info.torch_dtype)
        else:
            pixel_values = None
            num_patches = []
        assert len(num_patches) == len(
            idx_list), f'len(num_patches): {len(num_patches)}, len(idx_list): {len(idx_list)}'

        def _get_new_tokens(i):
            img_tokens: List[int] = self.processor.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * self.num_image_token * num_patches[i]
            if self.ocr_mode == 'lilt_qformer' and i == len(num_patches) - 1:
                # Textual boundary markers make the evidence interval explicit;
                # only the 64 internal slots are replaced with learned features.
                img_tokens += self.processor.encode('<ocr_start>', add_special_tokens=False)
                img_tokens += [OCR_DUMMY_TOKEN_INDEX] * self.ocr_num_queries
                img_tokens += self.processor.encode('<ocr_end>', add_special_tokens=False)
            return img_tokens

        encoded['input_ids'], encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
            input_ids, labels, loss_scale, idx_list, _get_new_tokens)
        encoded['pixel_values'] = pixel_values
        if self.ocr_mode == 'lilt_qformer' and images:
            encoded['ocr_data'] = ocr_record
        grounding_target = self._add_grounding_target(encoded, inputs)
        if self.enable_goar:
            encoded['goar_data'] = {
                'ocr': ocr_record,
                'line_spans': self._find_ocr_line_spans(encoded['input_ids'], ocr_record),
                'target': grounding_target or [],
                'num_tiles': int(sum(num_patches)),
            }
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        for key in ('tile_coords', 'tile_types'):
            values = [item[key] for item in batch if item.get(key) is not None]
            if values:
                res[key] = torch.cat(values, dim=0)
        ocr_data = [item['ocr_data'] for item in batch if item.get('ocr_data') is not None]
        if ocr_data:
            res['ocr_data'] = ocr_data
        goar_data = [item['goar_data'] for item in batch if item.get('goar_data') is not None]
        if goar_data:
            # Training pads on the right. In left-padded inference, line spans
            # are shifted to their positions in the collated sequence.
            sequence_length = max(len(source['input_ids']) for source in batch)
            for item, source in zip(goar_data, batch):
                source_length = len(source['input_ids'])
                item['padding_offset'] = sequence_length - source_length if self.padding_side == 'left' else 0
            res['goar_data'] = goar_data
        return res

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        tile_coords = inputs.pop('tile_coords', None)
        tile_types = inputs.pop('tile_types', None)
        ocr_data = inputs.pop('ocr_data', None)
        goar_data = inputs.pop('goar_data', None)
        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        input_ids = inputs['input_ids'].to(device=device)
        ocr_mask = input_ids == OCR_DUMMY_TOKEN_INDEX
        safe_input_ids = input_ids.masked_fill(ocr_mask, self.tokenizer.pad_token_id)
        inputs_embeds = embedding(safe_input_ids).to(device=device)
        pixel_values = inputs.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device)
            if tile_coords is not None:
                vit_embeds = model.extract_feature(
                    pixel_values,
                    tile_coords=tile_coords.to(device=device),
                    tile_types=tile_types.to(device=device) if tile_types is not None else None).to(device=device)
            else:
                vit_embeds = model.extract_feature(pixel_values).to(device=device)
            visual_mask = input_ids == self.processor.encode('<IMG_CONTEXT>', add_special_tokens=False)[0]
            inputs_embeds[visual_mask] = vit_embeds.reshape(-1, vit_embeds.shape[-1]).to(dtype=inputs_embeds.dtype)
        elif is_deepspeed_enabled():
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), device=device, dtype=inputs_embeds.dtype)
            vit_embeds = model.extract_feature(dummy_pixel_values).to(device=device)
            inputs_embeds += vit_embeds.mean() * 0.

        if ocr_mask.any():
            if ocr_data is None:
                ocr_data = [self._empty_ocr_record()] * (int(ocr_mask.sum().item()) // self.ocr_num_queries)
            encoder = model.ocr_evidence_encoder
            ocr_embeds = encoder(ocr_data).to(device=device, dtype=inputs_embeds.dtype)
            expected = int(ocr_mask.sum().item())
            if ocr_embeds.numel() == 0 or ocr_embeds.shape[0] * ocr_embeds.shape[1] != expected:
                raise ValueError(
                    f'OCR placeholder/embedding mismatch: placeholders={expected}, embeddings={tuple(ocr_embeds.shape)}')
            inputs_embeds[ocr_mask] = ocr_embeds.reshape(-1, ocr_embeds.shape[-1])
        if goar_data is not None:
            if tile_coords is None:
                raise ValueError('GOAR requires GOSA tile coordinates.')
            from swift.model.internvl_extensions import find_goar_host, get_active_goar_adapter
            host = find_goar_host(model)
            if host is None:
                raise ValueError('GOAR data reached the template but the model has no GOAR adapter.')
            goar_adapter = get_active_goar_adapter(host)
            inputs_embeds, aux_loss, boxes = goar_adapter(
                inputs_embeds, goar_data, vit_embeds, tile_coords.to(device=device), inputs.get('labels'), visual_mask)
            failed_samples = [index for index, box in enumerate(boxes) if box is None]
            if failed_samples:
                raise RuntimeError(
                    'GOAR found no valid OCR line spans for non-empty OCR samples '
                    f'{failed_samples}; refusing silent visual-only execution')
            # The pre-hook is registered on the unwrapped PEFT model. Keep the
            # graph tensor there for Trainer.compute_loss instead of adding
            # arbitrary fields to ModelOutput, which DDP cannot reconstruct.
            model._goar_aux_loss = aux_loss
            model._goar_forward_applied = True
            host._last_goar_boxes = boxes
        return {'inputs_embeds': inputs_embeds}


_internvl2_system = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.internvl2,
        default_system=_internvl2_system,
        template_cls=Internvl2Template,
    ))

register_template(
    Phi3TemplateMeta(
        MLLMTemplateType.internvl2_phi3,
        default_system=_internvl2_system,
        template_cls=Internvl2Template,
    ))

register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.internvl2_5,
        template_cls=Internvl2Template,
        default_system='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'))

register_template(ChatmlTemplateMeta(MLLMTemplateType.internvl3_5, template_cls=Internvl2Template))


class Internvl3_5GPTTemplate(Internvl2Template, GptTemplate):
    pass


register_template(GptOssTemplateMeta(MLLMTemplateType.internvl3_5_gpt, template_cls=Internvl3_5GPTTemplate))


class InternvlhfTemplate(Internvl2Template):

    def init_env_args(self):
        Template.init_env_args(self)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type in ['image', 'video']
        if media_type == 'video':
            if self.mode == 'vllm':
                return Template.replace_tag(self, 'video', index, inputs)
            else:
                return [[-200]]
        else:
            if self.mode == 'vllm':
                return ['<IMG_CONTEXT>']
            else:
                return ['<img>', [-100], '</img>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        import numpy as np
        from transformers.image_utils import concatenate_list, make_flat_list_of_images
        from transformers.video_utils import make_batched_videos

        from swift.template.vision_utils import load_video_hf
        encoded = super(InternvlTemplate, self)._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        images = inputs.images
        videos = inputs.videos
        image_num_patches_indices = np.array([0])
        video_num_patches_indices = np.array([0])
        video_patch_indices = np.array([0])
        image_num_patches = []
        video_num_patches = []
        image_video_patches = []
        image_idx_list = []
        video_idx_list = []
        image_pixel_values = None
        video_pixel_values = None

        if images:
            # InternS1Processor
            image_idx_list = findall(input_ids, -100)
            images = make_flat_list_of_images(images)
            image_inputs = self.processor.image_processor(images=images, crop_to_patches=True, return_tensors='pt')
            image_num_patches = image_inputs.pop('num_patches')
            image_pixel_values = image_inputs.pop('pixel_values').to(self.model_info.torch_dtype)
            image_num_patches_indices = np.cumsum(image_num_patches)
        if videos:
            video_idx_list = findall(input_ids, -200)
            videos, _ = load_video_hf(videos)
            videos = make_batched_videos(videos)
            video_inputs = self.processor.video_processor(videos=videos, return_tensors='pt')
            video_pixel_values = video_inputs.pop('pixel_values_videos').to(self.model_info.torch_dtype)
            num_frames_per_video = [len(video) for video in video_pixel_values]
            video_num_patches = [1 for frames in num_frames_per_video for _ in range(frames)]
            video_patch_indices = np.cumsum(num_frames_per_video)
            video_num_patches_indices = np.cumsum(video_num_patches)
            video_pixel_values = video_pixel_values.flatten(0, 1)

        def merge_and_sort(image_idx_list: List[int], video_idx_list: List[int]) -> tuple:
            """Merge and sort image and video index lists while preserving their relative order."""
            merged = []
            is_image_list = []
            i, j = 0, 0

            while i < len(image_idx_list) and j < len(video_idx_list):
                if image_idx_list[i] < video_idx_list[j]:
                    merged.append(image_idx_list[i])
                    i += 1
                    is_image_list.append(True)
                else:
                    merged.append(video_idx_list[j])
                    j += 1
                    is_image_list.append(False)
            # Add remaining elements
            merged.extend(image_idx_list[i:])
            is_image_list.extend([True] * (len(image_idx_list) - i))
            merged.extend(video_idx_list[j:])
            is_image_list.extend([False] * (len(video_idx_list) - j))
            return merged, is_image_list

        # Merge and sort the index lists
        idx_list, is_image_list = merge_and_sort(image_idx_list, video_idx_list)

        # Validate the lengths
        if images and len(image_idx_list) > 0:
            assert len(image_num_patches_indices) == len(image_idx_list)
        if videos and len(video_idx_list) > 0:
            assert len(video_patch_indices) == len(video_idx_list)

        def _get_new_tokens(i):
            if is_image_list[i]:
                # Find the corresponding image index
                image_idx = sum(is_image_list[:i])
                start = image_num_patches_indices[image_idx - 1] if image_idx > 0 else 0
                end = image_num_patches_indices[image_idx]
                image_seq_length = self.processor.image_seq_length
                image_video_patches.append(image_pixel_values[start:end])
                img_tokens: List[int] = self.processor.encode(
                    '<IMG_CONTEXT>', add_special_tokens=False) * image_seq_length * image_num_patches[image_idx]
            else:
                # Find the corresponding video index
                video_idx = i - sum(is_image_list[:i])
                current_patch = video_patch_indices[video_idx - 1] if video_idx > 0 else 0
                end_patch = video_patch_indices[video_idx]

                start = video_num_patches_indices[current_patch] if video_idx > 0 else 0
                end = video_num_patches_indices[end_patch - 1]
                image_video_patches.append(video_pixel_values[start:end])
                image_seq_length = self.processor.image_seq_length
                num_patches = list(video_num_patches[current_patch:end_patch])
                video_prompt = ''.join(
                    f"Frame{i + 1}: <img>{'<IMG_CONTEXT>' * image_seq_length * num_patches[i]}</img>\n"
                    for i in range(len(num_patches)))
                img_tokens = self.processor.encode(video_prompt, add_special_tokens=False)
            return img_tokens

        encoded['input_ids'], encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
            input_ids, labels, loss_scale, idx_list, _get_new_tokens)
        if images or videos:
            encoded['pixel_values'] = concatenate_list(image_video_patches)
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        input_ids = inputs['input_ids']
        inputs_embeds = embedding(input_ids).to(device=device)
        pixel_values = inputs.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device)
            image_features = model.model.get_image_features(
                pixel_values,
                vision_feature_layer=self.config.vision_feature_layer,
                vision_feature_select_strategy=self.config.vision_feature_select_strategy,
            )
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            special_image_mask = input_ids == self.config.image_token_id
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        elif is_deepspeed_enabled():
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), device=device, dtype=inputs_embeds.dtype)
            image_features = model.model.get_image_features(
                dummy_pixel_values,
                vision_feature_layer=self.config.vision_feature_layer,
                vision_feature_select_strategy=self.config.vision_feature_select_strategy,
            )
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            inputs_embeds = inputs_embeds + image_features.mean() * 0.
        return {'inputs_embeds': inputs_embeds}


INTERNS1_DEFAULT_SYSTEM = ('You are an expert reasoner with extensive experience in all areas. '
                           'You approach problems through systematic thinking and rigorous reasoning. '
                           'Your response should reflect deep understanding and precise logical thinking, '
                           'making your solution path and reasoning clear to others. '
                           'Please put your thinking process within <think>...</think> tags.')

register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.interns1,
        template_cls=InternvlhfTemplate,
        default_system=INTERNS1_DEFAULT_SYSTEM,
        is_thinking=True,
        thinking_prefix='<think>',
    ))

register_template(ChatmlTemplateMeta(MLLMTemplateType.internvl_hf, template_cls=InternvlhfTemplate))
