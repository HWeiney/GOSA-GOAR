#!/usr/bin/env python3
"""Measure offline PaddleOCR latency separately from model inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from run_ppocrv5 import build_engine, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-jsonl', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--ocr-version', choices=['PP-OCRv5', 'PP-OCRv6'], default='PP-OCRv6')
    parser.add_argument('--text-detection-model-dir', type=Path)
    parser.add_argument('--zh-recognition-model-dir', type=Path)
    parser.add_argument('--en-recognition-model-dir', type=Path)
    parser.add_argument('--device', default='gpu:0')
    args = parser.parse_args()

    images, seen = [], set()
    for dataset_path in args.dataset_jsonl:
        with dataset_path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                image = str(row['images'][0])
                if image not in seen:
                    images.append((image, row.get('language', 'en')))
                    seen.add(image)
    if not images:
        raise ValueError('No images found')
    engines = {language: build_engine(language, args)[0] for language in sorted({item[1] for item in images})}
    for index in range(args.warmup):
        image, language = images[index % len(images)]
        predict(engines[language], image)

    records = []
    for image, language in images[:args.samples]:
        start = time.perf_counter()
        predict(engines[language], image)
        records.append({'image': image, 'language': language, 'latency_seconds': time.perf_counter() - start})
    values = np.asarray([row['latency_seconds'] for row in records], dtype=np.float64)
    result = {
        'component': 'offline_paddleocr',
        'device': args.device,
        'ocr_version': args.ocr_version,
        'warmup_images': args.warmup,
        'measured_images': len(records),
        'latency_seconds': {
            'mean': float(values.mean()), 'median': float(np.median(values)),
            'p95': float(np.quantile(values, .95)),
        },
        'included_in_network_forward': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    args.output.with_suffix('.samples.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

