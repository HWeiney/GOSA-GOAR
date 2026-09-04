#!/usr/bin/env python3
"""Combine four inference benchmark summaries and calculate incremental overhead."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = ('internvl35_2b', 'gosa', 'gosa_ocr', 'goar')


def overhead(current, reference):
    return {
        'latency_mean_percent': 100. * (current['latency_seconds']['mean'] / reference['latency_seconds']['mean'] - 1.),
        'peak_gpu_memory_percent': 100. * (current['peak_gpu_memory_bytes'] / reference['peak_gpu_memory_bytes'] - 1.),
        'parameter_count_delta': current['parameter_count'] - reference['parameter_count'],
        'decoding_tokens_per_second_percent': 100. * (
            current['decoding_tokens_per_second'] / reference['decoding_tokens_per_second'] - 1.),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    models = {name: json.loads((args.result_root / name / 'summary.json').read_text(encoding='utf-8'))
              for name in MODELS}
    result = {
        'models': models,
        'gosa_overhead_vs_internvl35_2b': overhead(models['gosa'], models['internvl35_2b']),
        'goar_overhead_vs_gosa_ocr': overhead(models['goar'], models['gosa_ocr']),
        'paddleocr_note': 'Offline PaddleOCR preprocessing is excluded from every network-forward measurement.',
    }
    output = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    print(output, end='')
    path = args.output or args.result_root / 'efficiency_summary.json'
    path.write_text(output, encoding='utf-8')


if __name__ == '__main__':
    main()

