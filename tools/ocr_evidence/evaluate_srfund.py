#!/usr/bin/env python3
"""Evaluate grounded SRFUND predictions and OCR/InternVL complementarity."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple


OUTPUT_RE = re.compile(
    r'^\s*<ref>(.*?)</ref>\s*<box>\s*\[\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*'
    r'([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*'
    r'([-+]?\d+(?:\.\d+)?)\s*\]\]\s*</box>\s*$', re.DOTALL)
EMPTY_THINK_RE = re.compile(r'^\s*<think>\s*</think>\s*', re.DOTALL)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', str(text)).casefold()
    return ' '.join(text.split())


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        for j, char_right in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char_left != char_right)))
        previous = current
    return previous[-1]


def anls(prediction: str, target: str) -> float:
    prediction, target = normalize_text(prediction), normalize_text(target)
    if not prediction and not target:
        return 1.
    distance = edit_distance(prediction, target) / max(len(prediction), len(target), 1)
    return 1. - distance if distance < .5 else 0.


def parse_output(value: str) -> Optional[Tuple[str, list]]:
    # Qwen3.5 emits an empty thinking block even when thinking is disabled.
    # Treat that model-added wrapper as formatting, while still rejecting any
    # non-empty reasoning or other conversational text around the answer.
    value = EMPTY_THINK_RE.sub('', str(value), count=1)
    match = OUTPUT_RE.fullmatch(value)
    if match is None:
        return None
    box = [float(value) for value in match.groups()[1:]]
    if not all(0 <= value <= 1000 for value in box) or box[0] > box[2] or box[1] > box[3]:
        return None
    return match.group(1).strip(), box


def iou(left, right) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0., x2 - x1) * max(0., y2 - y1)
    left_area = max(0., left[2] - left[0]) * max(0., left[3] - left[1])
    right_area = max(0., right[2] - right[0]) * max(0., right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.


def target_from_row(row: Dict) -> Tuple[str, list]:
    answer, box = row.get('answer'), row.get('answer_bbox')
    if answer is not None and box is not None:
        return str(answer), box
    parsed = parse_output(row.get('labels', row.get('target', '')))
    if parsed is None:
        messages = row.get('messages') or []
        assistant = next((m['content'] for m in reversed(messages) if m.get('role') == 'assistant'), '')
        parsed = parse_output(assistant)
    if parsed is None:
        raise ValueError(f'Cannot parse target for sample {row.get("id")}')
    return parsed


def load_predictions(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    result = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            result[str(row['id'])] = str(row.get('response', row.get('prediction', '')))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--ocr-predictions', type=Path,
                        help='Independent OCR-only QA predictions keyed by id; never inferred from GT.')
    parser.add_argument('--correctness', choices=['text', 'grounded'], default='text')
    parser.add_argument('--oracle-gain-threshold', type=float, default=.03,
                        help='Report arbitration potential when oracle exceeds the better branch by this amount.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ocr_predictions = load_predictions(args.ocr_predictions)
    totals = {'count': 0, 'exact_match': 0., 'anls': 0., 'bbox_miou': 0.,
              'grounded_accuracy': 0., 'format_valid_rate': 0.}
    complement = {'both_correct': 0, 'ocr_only_correct': 0, 'internvl_only_correct': 0, 'both_wrong': 0}
    with args.predictions.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            target_text, target_box = target_from_row(row)
            prediction = str(row.get('response', row.get('prediction', '')))
            parsed = parse_output(prediction)
            pred_text, pred_box = parsed if parsed is not None else ('', [0, 0, 0, 0])
            exact = normalize_text(pred_text) == normalize_text(target_text)
            box_iou = iou(pred_box, target_box)
            grounded = exact and box_iou >= .5
            totals['count'] += 1
            totals['exact_match'] += exact
            totals['anls'] += anls(pred_text, target_text)
            totals['bbox_miou'] += box_iou
            totals['grounded_accuracy'] += grounded
            totals['format_valid_rate'] += parsed is not None

            sample_id = str(row.get('id', ''))
            ocr_prediction = ocr_predictions.get(sample_id, row.get('ocr_response'))
            if ocr_prediction is not None:
                ocr_parsed = parse_output(str(ocr_prediction))
                ocr_text, ocr_box = ocr_parsed if ocr_parsed else (str(ocr_prediction), [0, 0, 0, 0])
                ocr_correct = normalize_text(ocr_text) == normalize_text(target_text)
                if args.correctness == 'grounded':
                    ocr_correct = ocr_correct and iou(ocr_box, target_box) >= .5
                internvl_correct = grounded if args.correctness == 'grounded' else exact
                key = ('both_correct' if ocr_correct and internvl_correct else
                       'ocr_only_correct' if ocr_correct else
                       'internvl_only_correct' if internvl_correct else 'both_wrong')
                complement[key] += 1
    count = totals.pop('count')
    metrics = {'count': count, **{key: value / max(count, 1) for key, value in totals.items()}}
    complement_count = sum(complement.values())
    if complement_count:
        ocr_accuracy = (complement['both_correct'] + complement['ocr_only_correct']) / complement_count
        internvl_accuracy = (complement['both_correct'] + complement['internvl_only_correct']) / complement_count
        oracle_accuracy = (complement_count - complement['both_wrong']) / complement_count
        metrics['complementarity'] = {
            'count': complement_count,
            'ocr_only_accuracy': ocr_accuracy,
            'internvl_only_accuracy': internvl_accuracy,
            **complement,
            'oracle_accuracy': oracle_accuracy,
            'oracle_gain_vs_best': oracle_accuracy - max(ocr_accuracy, internvl_accuracy),
            'conflict_arbitration_recommended': (
                oracle_accuracy - max(ocr_accuracy, internvl_accuracy) >= args.oracle_gain_threshold),
            'correctness_definition': args.correctness,
        }
    output = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
