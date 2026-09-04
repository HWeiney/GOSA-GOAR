#!/usr/bin/env python3
"""Build per-seed and aggregate tables for the GOAR injection ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODES = ('gosa_ocr', 'full', 'no_injection', 'spatial_only', 'ocr_only')
METRICS = ('exact_match', 'anls', 'bbox_miou', 'grounded_accuracy')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', default=['42', '123', '3407'])
    parser.add_argument('--output-prefix', type=Path)
    args = parser.parse_args()
    prefix = args.output_prefix or args.result_root / 'injection_ablation_summary'
    rows = []
    for seed in args.seeds:
        values = {}
        for mode in MODES:
            path = args.result_root / f'seed{seed}' / mode / 'metrics.json'
            if not path.is_file():
                raise FileNotFoundError(path)
            values[mode] = json.loads(path.read_text(encoding='utf-8'))
        for mode in MODES:
            row = {'seed': seed, 'mode': mode}
            for metric in METRICS:
                value = float(values[mode][metric])
                row[metric] = value
                row[f'{metric}_vs_full'] = value - float(values['full'][metric])
                row[f'{metric}_vs_gosa_ocr'] = value - float(values['gosa_ocr'][metric])
            rows.append(row)
    fieldnames = list(rows[0])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {}
    for mode in MODES:
        mode_rows = [row for row in rows if row['mode'] == mode]
        aggregate[mode] = {
            key: {'mean': float(np.mean([row[key] for row in mode_rows])),
                  'standard_deviation': float(np.std([row[key] for row in mode_rows], ddof=1))}
            for key in fieldnames if key not in ('seed', 'mode')
        }
    payload = {'per_seed': rows, 'aggregate': aggregate}
    prefix.with_suffix('.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(prefix.with_suffix('.csv'))
    print(prefix.with_suffix('.json'))


if __name__ == '__main__':
    main()

