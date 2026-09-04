#!/usr/bin/env python3
"""Measure batch-one Transformers inference latency, throughput, and GPU memory."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from swift.infer_engine import InferRequest
from swift.pipelines.infer.infer import SwiftInfer


def describe(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(array.mean()),
        'median': float(np.median(array)),
        'p95': float(np.quantile(array, .95)),
    }


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model-path', type=Path,
                        help='Override the base model path stored in the checkpoint training arguments.')
    parser.add_argument('--dataset', type=Path, nargs='+', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--samples', type=int, default=1316)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the H20 inference benchmark')

    swift_args = [
        '--adapters', str(args.checkpoint), '--load_data_args', 'true',
        '--dataset', *[str(path) for path in args.dataset],
        '--remove_unused_columns', 'false', '--infer_backend', 'transformers',
        '--max_batch_size', '1', '--max_new_tokens', str(args.max_new_tokens),
        '--temperature', '0', '--stream', 'false', '--dataset_shuffle', 'false',
    ]
    if args.model_path:
        swift_args[0:0] = ['--model', str(args.model_path)]
    runner = SwiftInfer(swift_args)
    dataset = runner._prepare_val_dataset()
    if len(dataset) == 0:
        raise ValueError('Empty validation dataset')
    sample_count = min(args.samples, len(dataset))
    request_config = runner.args.get_request_config()

    generation_times = []
    original_generate = runner.template.generate

    def timed_generate(model, *generate_args, **generate_kwargs):
        synchronize()
        start = time.perf_counter()
        result = original_generate(model, *generate_args, **generate_kwargs)
        synchronize()
        generation_times.append(time.perf_counter() - start)
        return result

    runner.template.generate = timed_generate

    def run_one(index):
        data = copy.deepcopy(dataset[index % len(dataset)])
        InferRequest.remove_response(data['messages'])
        synchronize()
        start = time.perf_counter()
        response = runner.infer([data], request_config, use_tqdm=False, **runner.infer_kwargs)[0]
        synchronize()
        return time.perf_counter() - start, int(response.usage.completion_tokens)

    for index in range(args.warmup):
        run_one(index)
    generation_times.clear()
    torch.cuda.reset_peak_memory_stats()

    records = []
    for index in range(sample_count):
        latency, tokens = run_one(index)
        records.append({'sample_index': index, 'latency_seconds': latency, 'completion_tokens': tokens,
                        'generation_seconds': generation_times[-1]})

    model = runner.infer_engine.model
    latencies = [row['latency_seconds'] for row in records]
    generated_tokens = sum(row['completion_tokens'] for row in records)
    generation_seconds = sum(row['generation_seconds'] for row in records)
    summary = {
        'model': args.model_name,
        'checkpoint': str(args.checkpoint.resolve()),
        'device': torch.cuda.get_device_name(),
        'batch_size': 1,
        'decoding': 'greedy',
        'warmup_samples': args.warmup,
        'measured_samples': sample_count,
        'max_new_tokens': args.max_new_tokens,
        'latency_seconds': describe(latencies),
        'decoding_tokens_per_second': generated_tokens / max(generation_seconds, 1e-12),
        'generated_tokens': generated_tokens,
        'generation_seconds': generation_seconds,
        'peak_gpu_memory_bytes': int(torch.cuda.max_memory_allocated()),
        'parameter_count': int(sum(parameter.numel() for parameter in model.parameters())),
        'trainable_parameter_count': int(sum(parameter.numel() for parameter in model.parameters()
                                             if parameter.requires_grad)),
        'paddleocr_included': False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath('samples.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
    args.output_dir.joinpath('summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
