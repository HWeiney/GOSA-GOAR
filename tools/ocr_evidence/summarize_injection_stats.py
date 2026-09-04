#!/usr/bin/env python3
"""Summarize GOAR gate and injection residual JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def describe(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        'count': int(array.size),
        'mean': float(array.mean()),
        'standard_deviation': float(array.std(ddof=1)) if array.size > 1 else 0.,
        'median': float(np.median(array)),
        'p95': float(np.quantile(array, .95)),
    }


def summarize(path: Path):
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records:
        raise ValueError(f'No injection records found in {path}')
    gates = [row['gate'] for row in records]
    spatial = [row['spatial_residual_ratio'] for row in records]
    evidence = [row['ocr_evidence_residual_ratio'] for row in records
                if row.get('ocr_evidence_residual_ratio') is not None]
    return {
        'sample_count': len(records),
        'gate_type': 'shared_scalar',
        'gate': describe(gates),
        'spatial_residual_ratio': describe(spatial),
        'ocr_evidence_residual_ratio': describe(evidence) if evidence else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.input)
    output = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    print(output, end='')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')


if __name__ == '__main__':
    main()

