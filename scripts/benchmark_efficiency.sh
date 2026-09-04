#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to InternVL3.5-2B}
RESULT_ROOT=${RESULT_ROOT:-${ROOT_DIR}/outputs/efficiency}
VAL_DATA=${VAL_DATA:-"${ROOT_DIR}/annotations/SRFUND/srfund_zh_val.jsonl ${ROOT_DIR}/annotations/SRFUND/srfund_en_val.jsonl"}
read -r -a VAL_FILES <<< "${VAL_DATA}"

for model_name in internvl35_2b gosa gosa_ocr goar; do
  case "${model_name}" in
    internvl35_2b) checkpoint=${INTERNVL_CHECKPOINT:?Set INTERNVL_CHECKPOINT}; config=baseline.env ;;
    gosa) checkpoint=${GOSA_CHECKPOINT:?Set GOSA_CHECKPOINT}; config=gosa.env ;;
    gosa_ocr) checkpoint=${GOSA_OCR_CHECKPOINT:?Set GOSA_OCR_CHECKPOINT}; config=gosa_ocr_prompt.env ;;
    goar) checkpoint=${GOAR_CHECKPOINT:?Set GOAR_CHECKPOINT}; config=gosa_ocr_prompt_goar.env ;;
  esac
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/configs/${config}"
  [[ "${model_name}" != goar ]] && ENABLE_GOAR=False
  export OCR_MODE USE_OCR ENABLE_LOCAL_PE ENABLE_GLOBAL_PE ENABLE_GOAR
  if [[ "${model_name}" == goar ]]; then
    export GOAR_BOTTLENECK GOAR_MAX_OCR_LINES GOAR_POINTER_LOSS_WEIGHT
    export GOAR_BRANCH_LOSS_WEIGHT GOAR_UNCERTAINTY_LOSS_WEIGHT GOAR_LOSS_WEIGHT
  fi
  export GOAR_SPATIAL_INJECTION=True GOAR_OCR_EVIDENCE_INJECTION=True GOAR_RECORD_INJECTION_STATS=False
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/ocr_evidence/benchmark_inference.py" \
    --model-path "${MODEL_PATH}" --checkpoint "${checkpoint}" --dataset "${VAL_FILES[@]}" \
    --model-name "${model_name}" --output-dir "${RESULT_ROOT}/${model_name}" \
    --warmup "${BENCHMARK_WARMUP:-50}" --samples "${BENCHMARK_SAMPLES:-1316}" \
    --max-new-tokens "${BENCHMARK_MAX_NEW_TOKENS:-128}"
done
"${PYTHON_BIN}" "${ROOT_DIR}/tools/ocr_evidence/summarize_efficiency.py" --result-root "${RESULT_ROOT}"
