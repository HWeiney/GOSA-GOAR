#!/usr/bin/env python3
"""Convert the Hugging Face parquet release of SROIE for InternVL/GOAR.

The release stores images, four KIE text values, and OCR text-region boxes in
two parquet files.  It does not directly associate a KIE value with a box, so
this converter derives an auditable proxy box from the provided OCR regions.
Every row records the selected OCR indices and alignment quality.

The default auxiliary split is frozen as 200 train, 100 validation, and the
complete 361-document extended test set.  Sampling is image-level and
deterministic; no field rows from the same receipt can cross splits.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import random
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pyarrow.parquet as pq
from PIL import Image


DEFAULT_SOURCE = Path('data/SROIE/source')
FIELDS = ('company', 'date', 'address', 'total')
FIELD_LABELS = {
    'company': 'Company',
    'date': 'Date',
    'address': 'Address',
    'total': 'Total',
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_bbox(box: Sequence[float], width: int, height: int) -> List[int]:
    if len(box) != 4:
        raise ValueError(f'Expected xyxy box, got {box!r}')
    result = [
        round(1000 * float(box[0]) / width),
        round(1000 * float(box[1]) / height),
        round(1000 * float(box[2]) / width),
        round(1000 * float(box[3]) / height),
    ]
    result = [max(0, min(1000, value)) for value in result]
    if result[0] > result[2] or result[1] > result[3]:
        raise ValueError(f'Inverted bbox after normalization: {box!r}')
    return result


def union_boxes(boxes: Sequence[Sequence[int]]) -> List[int]:
    if not boxes:
        raise ValueError('Cannot union an empty list of boxes.')
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def normalized_text(value: str) -> str:
    value = unicodedata.normalize('NFKC', str(value)).casefold()
    return ' '.join(value.split())


def alnum_text(value: str) -> str:
    return ''.join(re.findall(r'[a-z0-9]+', normalized_text(value)))


def tokens(value: str) -> List[str]:
    return re.findall(r'[a-z0-9]+', normalized_text(value))


def multiset_overlap(predicted: Sequence[str], target: Sequence[str]) -> Tuple[float, float, float]:
    predicted_count, target_count = Counter(predicted), Counter(target)
    overlap = sum((predicted_count & target_count).values())
    precision = overlap / max(sum(predicted_count.values()), 1)
    recall = overlap / max(sum(target_count.values()), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _span_score(text: str, target: str) -> Dict[str, float]:
    precision, recall, token_f1 = multiset_overlap(tokens(text), tokens(target))
    compact_text, compact_target = alnum_text(text), alnum_text(target)
    sequence = difflib.SequenceMatcher(None, compact_text, compact_target).ratio()
    containment = float(bool(compact_target) and bool(compact_text) and (
        compact_target in compact_text or compact_text in compact_target))
    score = 0.55 * token_f1 + 0.30 * sequence + 0.15 * containment
    return {
        'score': score,
        'token_precision': precision,
        'token_recall': recall,
        'token_f1': token_f1,
        'sequence_similarity': sequence,
        'containment': containment,
    }


def best_contiguous_span(words: Sequence[str], target: str, max_span: int) -> Tuple[List[int], Dict]:
    best_indices, best_metrics = [], {'score': -1.0}
    for start in range(len(words)):
        for end in range(start + 1, min(len(words), start + max_span) + 1):
            text = ' '.join(words[start:end])
            metrics = _span_score(text, target)
            # Prefer a tighter region when textual scores tie.
            metrics['score'] -= 0.002 * (end - start - 1)
            candidate = list(range(start, end))
            if (metrics['score'], -len(candidate), -start) > (
                    best_metrics['score'], -len(best_indices), -(best_indices[0] if best_indices else 0)):
                best_indices, best_metrics = candidate, metrics
    best_metrics = dict(best_metrics)
    best_metrics['method'] = 'contiguous-ocr-span'
    return best_indices, best_metrics


def best_address_span(words: Sequence[str], target: str, max_span: int = 8) -> Tuple[List[int], Dict]:
    """Favor coverage of a long address before lexical precision.

    Generic fuzzy matching often picks one perfectly matching house number and
    discards the remaining address lines.  Address grounding instead ranks
    contiguous spans by target-token recall, then F1 and sequence similarity.
    """
    best_indices, best_metrics = [], {'score': -1.0}
    best_rank = (-1.0, -1.0, -1.0, 0, 0)
    for start in range(len(words)):
        for end in range(start + 1, min(len(words), start + max_span) + 1):
            metrics = _span_score(' '.join(words[start:end]), target)
            candidate = list(range(start, end))
            rank = (
                metrics['token_recall'], metrics['token_f1'],
                metrics['sequence_similarity'], -len(candidate), -start)
            if rank > best_rank:
                best_indices, best_metrics, best_rank = candidate, metrics, rank
    best_metrics = dict(best_metrics)
    best_metrics['method'] = 'address-recall-first-contiguous-ocr-span'
    return best_indices, best_metrics


def date_digit_variants(value: str) -> set[str]:
    """Return common compact numeric renderings for a date annotation."""
    digits = ''.join(re.findall(r'\d+', normalized_text(value)))
    variants = {digits} if digits else set()
    if len(digits) == 8:
        # SROIE annotations mix YYYYMMDD with OCR-rendered DD/MM/YYYY.
        variants.update({digits[6:8] + digits[4:6] + digits[:4],
                         digits[4:6] + digits[6:8] + digits[:4]})
    return variants


def monetary_values(value: str) -> set[Decimal]:
    """Extract complete numeric amounts; never accept one-digit substrings."""
    result = set()
    for match in re.findall(r'(?<!\d)\d[\d,]*(?:\.\d{1,2})?(?!\d)', normalized_text(value)):
        try:
            result.add(Decimal(match.replace(',', '')))
        except InvalidOperation:
            pass
    return result


def align_date(words: Sequence[str], target: str) -> Tuple[List[int], Dict]:
    target_variants = date_digit_variants(target)
    exact = []
    for index, word in enumerate(words):
        line_digits = ''.join(re.findall(r'\d+', normalized_text(word)))
        if line_digits and any(variant in line_digits for variant in target_variants):
            exact.append(index)
    if exact:
        index = max(exact, key=lambda candidate: (_span_score(words[candidate], target)['score'],
                                                  -candidate))
        metrics = _span_score(words[index], target)
        metrics['method'] = 'date-equivalent-digit-order'
        return [index], metrics
    return best_contiguous_span(words, target, max_span=3)


def align_total(words: Sequence[str], target: str) -> Tuple[List[int], Dict]:
    """Disambiguate repeated amounts using the last standalone TOTAL anchor."""
    target_amounts = monetary_values(target)
    candidates = []
    for index, word in enumerate(words):
        if target_amounts & monetary_values(word):
            candidates.append(index)
    anchors = []
    excluded = {'exclude', 'inclusive', 'gst', 'subtotal', 'tax'}
    for index, word in enumerate(words):
        line_tokens = set(tokens(word))
        if 'total' not in line_tokens:
            continue
        preference = 2 if not (line_tokens & excluded) else 1
        if line_tokens <= {'total'}:
            preference = 3
        anchors.append((preference, index))
    if candidates and anchors:
        preferred_anchor = max(anchors)[1]
        index = min(candidates, key=lambda candidate: (abs(candidate - preferred_anchor), -candidate))
        metrics = _span_score(words[index], target)
        metrics.update({
            'method': 'amount-nearest-standalone-total-anchor',
            'total_anchor_index': preferred_anchor,
        })
        return [index], metrics
    return best_contiguous_span(words, target, max_span=2)


def align_entity_bbox(field_name: str, target: str, words: Sequence[str],
                      boxes: Sequence[Sequence[int]]) -> Tuple[List[int], Dict]:
    if len(words) != len(boxes):
        raise ValueError('SROIE words and bboxes must have equal lengths.')
    if not words:
        raise ValueError('Cannot align an entity without OCR regions.')
    if field_name == 'total':
        indices, metrics = align_total(words, target)
    elif field_name == 'date':
        indices, metrics = align_date(words, target)
    elif field_name == 'address':
        indices, metrics = best_address_span(words, target)
    else:
        indices, metrics = best_contiguous_span(words, target, max_span=3)
    if not indices:
        raise ValueError(f'No OCR alignment found for {field_name}={target!r}')
    metrics = dict(metrics)
    metrics.update({
        'ocr_indices': indices,
        'ocr_text': [str(words[index]) for index in indices],
        'proxy_grounding': True,
    })
    return union_boxes([boxes[index] for index in indices]), metrics


def freeze_split(train_keys: Sequence[str], test_keys: Sequence[str], *, seed: int,
                 train_size: int, valid_size: int) -> Dict[str, List[str]]:
    if train_size + valid_size > len(train_keys):
        raise ValueError('Requested train/validation sizes exceed the official train pool.')
    if set(train_keys) & set(test_keys):
        raise ValueError('Official train and test keys overlap.')
    shuffled = list(train_keys)
    random.Random(seed).shuffle(shuffled)
    return {
        'train': shuffled[:train_size],
        'valid': shuffled[train_size:train_size + valid_size],
        'unused_train_pool': shuffled[train_size + valid_size:],
        'test': list(test_keys),
    }


def read_parquet_rows(path: Path) -> List[Dict]:
    columns = ['image', 'key', 'image_size', 'entities', 'words', 'bboxes']
    return pq.read_table(path, columns=columns).to_pylist()


def image_extension(image_record: Dict) -> str:
    suffix = Path(str(image_record.get('path') or '')).suffix.lower()
    return suffix if suffix in {'.jpg', '.jpeg', '.png'} else '.jpg'


def extract_image(row: Dict, output_path: Path) -> Tuple[int, int]:
    image_bytes = row['image']['bytes']
    if not image_bytes:
        raise ValueError(f'{row["key"]}: embedded image bytes are missing.')
    expected = (int(row['image_size']['width']), int(row['image_size']['height']))
    with Image.open(io.BytesIO(image_bytes)) as image:
        actual = image.size
        image.verify()
    if actual != expected:
        raise ValueError(f'{row["key"]}: image size {actual} != metadata {expected}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.is_file() or output_path.stat().st_size != len(image_bytes):
        output_path.write_bytes(image_bytes)
    return expected


def write_ocr_cache(row: Dict, image_path: Path, cache_path: Path,
                    width: int, height: int) -> Dict:
    words = [str(word).strip() for word in row['words']]
    boxes_px = [[int(value) for value in box] for box in row['bboxes']]
    if len(words) != len(boxes_px):
        raise ValueError(f'{row["key"]}: OCR words/boxes length mismatch.')
    boxes = [normalize_bbox(box, width, height) for box in boxes_px]
    payload = {
        'schema_version': 1,
        'engine': 'SROIE-provided-OCR',
        'source': image_path.as_posix(),
        'image_size': [width, height],
        'bbox_space': '[0,1000]',
        'language': 'en',
        'confidence_policy': '1.0-imputed-because-provided-OCR-has-no-confidence',
        'words': words,
        'bboxes': boxes,
        'bboxes_px': boxes_px,
        'confidences': [1.0] * len(words),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return payload


def build_field_row(source: Dict, *, split: str, field_name: str, image_path: Path,
                    cache_path: Path, width: int, height: int,
                    words: Sequence[str], boxes: Sequence[Sequence[int]]) -> Dict | None:
    answer = str(source['entities'][field_name]).strip()
    if not answer:
        return None
    answer_bbox, alignment = align_entity_bbox(field_name, answer, words, boxes)
    target = f'<ref>{answer}</ref><box>[[{",".join(map(str, answer_bbox))}]]</box>'
    question = f'Locate and answer the field: {FIELD_LABELS[field_name]}'
    return {
        'id': f'{split}:{source["key"]}:{field_name}',
        'images': [image_path.as_posix()],
        'messages': [
            {'role': 'user', 'content': f'<image>\n{question}'},
            {'role': 'assistant', 'content': target},
        ],
        'language': 'en',
        'split': split,
        'source': 'ICDAR2019-SROIE',
        'dataset_name': 'SROIE',
        'key': source['key'],
        'entity_name': field_name,
        'question': question,
        'answer': answer,
        'answer_bbox': answer_bbox,
        'image_width': width,
        'image_height': height,
        'bbox_space': '[0,1000]',
        'bbox_supervision': 'proxy-from-SROIE-provided-OCR-regions',
        'bbox_alignment': alignment,
        'ocr_cache': cache_path.as_posix(),
        'ground_truth': target,
    }


def write_split(rows_by_key: Dict[str, Dict], keys: Sequence[str], *, split: str,
                output_dir: Path) -> Dict:
    output_path = output_dir / f'sroie_{split}.jsonl'
    image_dir, cache_dir = output_dir / 'images', output_dir / 'ocr_cache'
    stats = Counter()
    alignment_scores = {field: [] for field in FIELDS}
    with output_path.open('w', encoding='utf-8') as writer:
        for key in keys:
            source = rows_by_key[key]
            extension = image_extension(source['image'])
            image_path = image_dir / f'{key}{extension}'
            cache_path = cache_dir / f'{key}.json'
            width, height = extract_image(source, image_path)
            ocr = write_ocr_cache(source, image_path, cache_path, width, height)
            for field_name in FIELDS:
                row = build_field_row(
                    source, split=split, field_name=field_name,
                    image_path=image_path, cache_path=cache_path,
                    width=width, height=height,
                    words=ocr['words'], boxes=ocr['bboxes'])
                if row is None:
                    stats[f'missing_annotation:{field_name}'] += 1
                    continue
                writer.write(json.dumps(row, ensure_ascii=False) + '\n')
                score = float(row['bbox_alignment']['score'])
                alignment_scores[field_name].append(score)
                stats['qa_rows'] += 1
                stats[f'field:{field_name}'] += 1
                if score < 0.5:
                    stats[f'low_alignment:{field_name}'] += 1
            stats['documents'] += 1
    return {
        **dict(stats),
        'alignment_score': {
            field: {
                'min': min(scores),
                'mean': sum(scores) / len(scores),
                'max': max(scores),
            } for field, scores in alignment_scores.items()
        },
    }


def convert(args) -> Dict:
    source_dir = args.source
    train_path = source_dir / 'data' / 'train-00000-of-00001.parquet'
    test_path = source_dir / 'data' / 'test-00000-of-00001.parquet'
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_rows, test_rows = read_parquet_rows(train_path), read_parquet_rows(test_path)
    rows_by_key = {}
    for row in [*train_rows, *test_rows]:
        key = str(row['key'])
        if key in rows_by_key:
            raise ValueError(f'Duplicate SROIE key: {key}')
        rows_by_key[key] = row
    split = freeze_split(
        [str(row['key']) for row in train_rows],
        [str(row['key']) for row in test_rows],
        seed=args.seed, train_size=args.train_size, valid_size=args.valid_size)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for split_name in ('train', 'valid', 'test'):
        split_stats[split_name] = write_split(
            rows_by_key, split[split_name], split=split_name, output_dir=output_dir)
    report = {
        'schema_version': 1,
        'dataset': 'ICDAR2019-SROIE extended Hugging Face release',
        'source_dir': str(source_dir),
        'output_dir': str(output_dir),
        'split_policy': {
            'seed': args.seed,
            'train_size': args.train_size,
            'valid_size': args.valid_size,
            'test_size': len(split['test']),
            'unused_official_train_size': len(split['unused_train_pool']),
            'sampling_unit': 'receipt image key',
        },
        'split_keys': split,
        'bbox_supervision': {
            'kind': 'proxy',
            'source': 'SROIE-provided OCR text-region boxes aligned to KIE text values',
            'reporting_rule': 'mIoU/GA must be labeled proxy-grounding metrics, not official SROIE metrics',
        },
        'ocr': {
            'engine': 'SROIE-provided-OCR',
            'confidence_policy': '1.0 imputed',
        },
        'input_sha256': {
            train_path.name: file_sha256(train_path),
            test_path.name: file_sha256(test_path),
        },
        'stats': split_stats,
    }
    (output_dir / 'conversion_manifest.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/SROIE'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-size', type=int, default=200)
    parser.add_argument('--valid-size', type=int, default=100)
    args = parser.parse_args()
    if args.train_size <= 0 or args.valid_size <= 0:
        parser.error('--train-size and --valid-size must be positive.')
    report = convert(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
