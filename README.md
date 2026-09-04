# GOSA-GOAR Reproduction

Minimal reproduction code for grounded field question answering on document
images. The implementation is based on `ms-swift` and InternVL3.5 and contains
three model changes:

1. **GOSA Local PE** adds learnable row/column embeddings after pixel shuffle.
2. **GOSA Global PE** maps every visual token to its normalized page position
   and distinguishes regular tiles from the full-page thumbnail.
3. **GOAR** selects OCR evidence, refines an OCR anchor, predicts a visual box,
   fuses both sources by uncertainty, and injects the result into the original
   autoregressive generation path.

The generated answer format is:

```text
<ref>answer text</ref><box>[[x1,y1,x2,y2]]</box>
```

Coordinates use the integer `[0, 1000]` page coordinate system.

## Repository layout

```text
annotations/          Public derived annotations and private-data schema only
configs/              Baseline and three-improvement experiment switches
docs/DATASETS.md      Download, preprocessing, and data-use rules
scripts/train.sh      Portable LoRA training entry point
scripts/evaluate.sh   Deterministic inference and metric entry point
swift/                Vendored ms-swift runtime with GOSA/GOAR integration
tools/                Public-dataset conversion, OCR, and evaluation utilities
```

No model weights, document images, OCR caches, checkpoints, logs, machine
paths, user names, or private annotation values are included.

## Installation

Python 3.10 or 3.11 and a CUDA-enabled PyTorch environment are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install PaddleOCR only when OCR prompt or GOAR experiments are required:

```bash
python -m pip install paddleocr
# Install the matching PaddlePaddle GPU or CPU package for your platform.
```

Download an InternVL3.5 checkpoint into `pretrained/InternVL3_5-2B`, or pass
its local path with `MODEL_PATH`. See [docs/DATASETS.md](docs/DATASETS.md) for
dataset preparation.

## Training

Inspect a command without launching training:

```bash
DRY_RUN=true MODEL_PATH=/path/to/InternVL3_5-2B \
  bash scripts/train.sh configs/gosa_ocr_prompt_goar.env
```

Run the full method:

```bash
MODEL_PATH=/path/to/InternVL3_5-2B \
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  bash scripts/train.sh configs/gosa_ocr_prompt_goar.env
```

Useful ablations:

```bash
bash scripts/train.sh configs/baseline.env
bash scripts/train.sh configs/gosa_local_only.env
bash scripts/train.sh configs/gosa_global_only.env
bash scripts/train.sh configs/gosa.env
```

Before an OCR-prompt or GOAR run, create OCR caches:

```bash
python tools/ocr_evidence/run_ppocrv5.py --ocr-version PP-OCRv6 \
  --dataset-jsonl annotations/SRFUND/srfund_zh_train.jsonl \
                    annotations/SRFUND/srfund_en_train.jsonl \
                    annotations/SRFUND/srfund_zh_val.jsonl \
                    annotations/SRFUND/srfund_en_val.jsonl
```

## Evaluation

```bash
MODEL_PATH=/path/to/InternVL3_5-2B \
ADAPTER_PATH=outputs/<run>/checkpoint-<step> \
  bash scripts/evaluate.sh
```

Evidence-injection causal ablations, gate/residual statistics, paired bootstrap confidence intervals, and the
four-model efficiency benchmark are documented in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

The evaluator reports exact match, ANLS, mean IoU, and grounded accuracy. For
SROIE, bounding boxes are OCR-aligned proxy supervision and spatial metrics
must be reported as proxy-grounding results rather than official SROIE scores.

## Code map

- `swift/model/internvl_extensions/gosa.py`: Local PE and Global PE.
- `swift/model/internvl_extensions/goar.py`: GOAR adapter and auxiliary losses.
- `swift/model/models/internlm.py`: model attachment hooks.
- `swift/template/templates/internvl.py`: tile geometry and OCR prompt inputs.
- `swift/trainers/seq2seq_trainer.py`: auxiliary-loss aggregation.
- `tools/ocr_evidence/`: public data conversion, OCR, and evaluation.

The vendored framework retains its upstream Apache-2.0 license. Dataset files
remain subject to their original dataset licenses and terms.
