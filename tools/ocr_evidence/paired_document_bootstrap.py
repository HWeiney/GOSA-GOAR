#!/usr/bin/env python3
"""Paired document bootstrap for two SRFUND prediction files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_srfund import anls, iou, normalize_text, parse_output, target_from_row


METRICS = ('exact_match', 'anls', 'bbox_miou', 'grounded_accuracy')


def load_scores(path: Path):
    scores = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            sample_id = str(row.get('id', ''))
            if not sample_id:
                raise ValueError(f'Missing sample id in {path}')
            target_text, target_box = target_from_row(row)
            parsed = parse_output(str(row.get('response', row.get('prediction', ''))))
            pred_text, pred_box = parsed if parsed else ('', [0, 0, 0, 0])
            exact = float(normalize_text(pred_text) == normalize_text(target_text))
            box_iou = iou(pred_box, target_box)
            document_id = sample_id.rsplit(':', 1)[0]
            scores[sample_id] = (document_id, np.asarray([
                exact, anls(pred_text, target_text), box_iou, float(exact > 0 and box_iou >= .5)
            ], dtype=np.float64))
    return scores


def mean_scores(paths):
    runs = [load_scores(path) for path in paths]
    reference_ids = runs[0].keys()
    if any(run.keys() != reference_ids for run in runs[1:]):
        raise ValueError('Prediction ids differ across seeds')
    return {
        sample_id: (runs[0][sample_id][0], np.stack([run[sample_id][1] for run in runs]).mean(axis=0))
        for sample_id in reference_ids
    }


def paired_bootstrap(full_paths, no_injection_paths, iterations: int, seed: int):
    if len(full_paths) != len(no_injection_paths):
        raise ValueError('The number of Full and No-injection runs must match')
    full, no_injection = mean_scores(full_paths), mean_scores(no_injection_paths)
    if full.keys() != no_injection.keys():
        missing_full = sorted(no_injection.keys() - full.keys())[:5]
        missing_no = sorted(full.keys() - no_injection.keys())[:5]
        raise ValueError(f'Prediction ids differ; missing full={missing_full}, missing no-injection={missing_no}')
    documents = defaultdict(list)
    for sample_id, (document_id, full_score) in full.items():
        no_document_id, no_score = no_injection[sample_id]
        if document_id != no_document_id:
            raise ValueError(f'Document mismatch for {sample_id}')
        documents[document_id].append(full_score - no_score)
    document_arrays = [np.stack(rows) for rows in documents.values()]
    point = np.concatenate(document_arrays).mean(axis=0)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((iterations, len(METRICS)), dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(document_arrays), size=len(document_arrays))
        bootstrap[index] = np.concatenate([document_arrays[i] for i in selected]).mean(axis=0)
    low, high = np.quantile(bootstrap, [.025, .975], axis=0)
    return {
        'comparison': 'full_minus_no_injection',
        'query_count': len(full),
        'document_count': len(documents),
        'seed_count': len(full_paths),
        'iterations': iterations,
        'bootstrap_seed': seed,
        'metrics': {
            name: {'delta': float(point[i]), 'ci95': [float(low[i]), float(high[i])]}
            for i, name in enumerate(METRICS)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', type=Path, nargs='+', required=True)
    parser.add_argument('--no-injection', type=Path, nargs='+', required=True)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = paired_bootstrap(args.full, args.no_injection, args.iterations, args.seed)
    output = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    print(output, end='')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')


if __name__ == '__main__':
    main()

