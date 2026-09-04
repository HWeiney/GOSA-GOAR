# Additional experiments

The evidence-injection ablation reuses trained GOAR checkpoints. It does not retrain four variants: the auxiliary
heads and box predictions remain active, while two runtime switches control whether spatial and OCR evidence residuals
reach the autoregressive generator.

## Evidence injection

Set the base model and six checkpoint paths, then launch all three seeds:

```bash
export MODEL_PATH=/path/to/InternVL3_5-2B
export GOAR_SEED42_CHECKPOINT=/path/to/goar-seed42
export GOAR_SEED123_CHECKPOINT=/path/to/goar-seed123
export GOAR_SEED3407_CHECKPOINT=/path/to/goar-seed3407
export GOSA_OCR_SEED42_CHECKPOINT=/path/to/gosa-ocr-seed42
export GOSA_OCR_SEED123_CHECKPOINT=/path/to/gosa-ocr-seed123
export GOSA_OCR_SEED3407_CHECKPOINT=/path/to/gosa-ocr-seed3407
CUDA_VISIBLE_DEVICES=0 bash scripts/run_injection_ablation.sh
```

For a single run:

```bash
MODEL_PATH=/path/to/InternVL3_5-2B CUDA_VISIBLE_DEVICES=0 \
  bash scripts/evaluate_injection.sh /path/to/goar-seed42 full outputs/seed42/full
```

Supported modes are `full`, `no_injection`, `spatial_only`, `ocr_only`, and the direct `gosa_ocr` baseline. Every
GOAR run records the shared sigmoid gate plus gate-scaled spatial and OCR residual ratios. Summaries contain mean,
sample standard deviation, median, and P95. The launcher reports EM, ANLS, mIoU, grounded accuracy, differences from
Full and GOSA+OCR, and document-level paired bootstrap 95% intervals for Full minus No injection.

## Efficiency

Use seed-matched checkpoints and run every model on the same GPU:

```bash
export MODEL_PATH=/path/to/InternVL3_5-2B
export INTERNVL_CHECKPOINT=/path/to/internvl-checkpoint
export GOSA_CHECKPOINT=/path/to/gosa-checkpoint
export GOSA_OCR_CHECKPOINT=/path/to/gosa-ocr-checkpoint
export GOAR_CHECKPOINT=/path/to/goar-checkpoint
CUDA_VISIBLE_DEVICES=0 BENCHMARK_WARMUP=50 BENCHMARK_SAMPLES=1316 \
  bash scripts/benchmark_efficiency.sh
```

The benchmark uses batch size 1, greedy decoding, identical samples and output limits. It writes raw per-query timing,
mean/median/P95 end-to-end latency, decoding tokens/s, peak allocated GPU memory, parameter counts, GOSA overhead over
InternVL3.5-2B, and GOAR overhead over GOSA+OCR.

PaddleOCR is not included in model-forward timing. Measure it as an offline component:

```bash
python tools/ocr_evidence/benchmark_paddleocr_offline.py \
  --dataset-jsonl annotations/SRFUND/srfund_zh_val.jsonl annotations/SRFUND/srfund_en_val.jsonl \
  --samples 500 --warmup 10 --device gpu:0 --output outputs/efficiency/paddleocr.json
```
