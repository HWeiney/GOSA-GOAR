#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
if (($# != 3)); then
  echo "Usage: $0 CHECKPOINT {gosa_ocr|full|no_injection|spatial_only|ocr_only} OUTPUT_DIR" >&2
  exit 2
fi
ADAPTER_PATH=$(realpath "$1")
MODE=$2
RESULT_DIR=$3
export ADAPTER_PATH RESULT_DIR

case "${MODE}" in
  gosa_ocr)
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/configs/gosa_ocr_prompt.env"
    ENABLE_GOAR=False; GOAR_SPATIAL_INJECTION=False; GOAR_OCR_EVIDENCE_INJECTION=False
    GOAR_RECORD_INJECTION_STATS=False
    ;;
  full|no_injection|spatial_only|ocr_only)
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/configs/gosa_ocr_prompt_goar.env"
    case "${MODE}" in
      full) GOAR_SPATIAL_INJECTION=True; GOAR_OCR_EVIDENCE_INJECTION=True ;;
      no_injection) GOAR_SPATIAL_INJECTION=False; GOAR_OCR_EVIDENCE_INJECTION=False ;;
      spatial_only) GOAR_SPATIAL_INJECTION=True; GOAR_OCR_EVIDENCE_INJECTION=False ;;
      ocr_only) GOAR_SPATIAL_INJECTION=False; GOAR_OCR_EVIDENCE_INJECTION=True ;;
    esac
    GOAR_RECORD_INJECTION_STATS=True
    mkdir -p "${RESULT_DIR}"
    GOAR_INJECTION_STATS_PATH=${RESULT_DIR}/injection_stats.jsonl
    truncate -s 0 "${GOAR_INJECTION_STATS_PATH}"
    ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac
export OCR_MODE USE_OCR ENABLE_LOCAL_PE ENABLE_GLOBAL_PE ENABLE_GOAR
export GOAR_BOTTLENECK GOAR_MAX_OCR_LINES GOAR_POINTER_LOSS_WEIGHT
export GOAR_BRANCH_LOSS_WEIGHT GOAR_UNCERTAINTY_LOSS_WEIGHT GOAR_LOSS_WEIGHT
export GOAR_SPATIAL_INJECTION GOAR_OCR_EVIDENCE_INJECTION GOAR_RECORD_INJECTION_STATS
export GOAR_INJECTION_STATS_PATH

"${SCRIPT_DIR}/evaluate.sh"
if [[ "${MODE}" != gosa_ocr ]]; then
  "${PYTHON_BIN:-python}" "${ROOT_DIR}/tools/ocr_evidence/summarize_injection_stats.py" \
    --input "${GOAR_INJECTION_STATS_PATH}" --output "${RESULT_DIR}/injection_stats_summary.json"
fi
