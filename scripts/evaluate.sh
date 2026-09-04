#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH=${MODEL_PATH:-${ROOT_DIR}/pretrained/InternVL3_5-2B}
ADAPTER_PATH=${ADAPTER_PATH:?Set ADAPTER_PATH to a trained checkpoint}
VAL_DATA=${VAL_DATA:-"${ROOT_DIR}/annotations/SRFUND/srfund_zh_val.jsonl ${ROOT_DIR}/annotations/SRFUND/srfund_en_val.jsonl"}
RESULT_DIR=${RESULT_DIR:-${ROOT_DIR}/outputs/evaluation}
read -r -a VAL_FILES <<< "${VAL_DATA}"
mkdir -p "${RESULT_DIR}"

"${PYTHON_BIN}" -m swift.cli.main infer \
  --model "${MODEL_PATH}" \
  --adapters "${ADAPTER_PATH}" \
  --dataset "${VAL_FILES[@]}" \
  --result_path "${RESULT_DIR}/predictions.jsonl" \
  --infer_backend transformers \
  --max_batch_size 1 \
  --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
  --temperature 0 \
  --stream false

"${PYTHON_BIN}" "${ROOT_DIR}/tools/ocr_evidence/evaluate_srfund.py" \
  --predictions "${RESULT_DIR}/predictions.jsonl" \
  --output "${RESULT_DIR}/metrics.json"
