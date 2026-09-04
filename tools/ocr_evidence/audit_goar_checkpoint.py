#!/usr/bin/env python3
"""Fail fast when a GOAR adapter was saved without a trained injection path."""

import argparse
from pathlib import Path

from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    weights = args.checkpoint / 'adapter_model.safetensors'
    if not weights.is_file():
        raise FileNotFoundError(weights)

    suffixes = ('goar_adapter.spatial_out.2.weight', 'goar_adapter.evidence_out.weight')
    maxima = {}
    with safe_open(str(weights), framework='pt', device='cpu') as handle:
        keys = list(handle.keys())
        for suffix in suffixes:
            matches = [key for key in keys if key.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(f'Expected exactly one {suffix}, found {matches}')
            maxima[suffix] = handle.get_tensor(matches[0]).float().abs().max().item()

    print('GOAR checkpoint injection audit:')
    for name, value in maxima.items():
        print(f'  {name}: absmax={value:.8g}')
    if not any(value > 0 for value in maxima.values()):
        raise RuntimeError('GOAR spatial/evidence injection weights are all zero; checkpoint is structurally inactive')


if __name__ == '__main__':
    main()
