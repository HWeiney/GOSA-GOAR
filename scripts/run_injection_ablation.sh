#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
RESULT_ROOT=${RESULT_ROOT:-${ROOT_DIR}/outputs/injection_ablation}
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHON_BIN

for seed in 42 123 3407; do
  goar_var=GOAR_SEED${seed}_CHECKPOINT
  strong_var=GOSA_OCR_SEED${seed}_CHECKPOINT
  goar_checkpoint=${!goar_var:?Set ${goar_var}}
  strong_checkpoint=${!strong_var:?Set ${strong_var}}
  for mode in gosa_ocr full no_injection spatial_only ocr_only; do
    checkpoint=${goar_checkpoint}
    [[ "${mode}" == gosa_ocr ]] && checkpoint=${strong_checkpoint}
    "${SCRIPT_DIR}/evaluate_injection.sh" "${checkpoint}" "${mode}" "${RESULT_ROOT}/seed${seed}/${mode}"
  done
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/ocr_evidence/paired_document_bootstrap.py" \
    --full "${RESULT_ROOT}/seed${seed}/full/predictions.jsonl" \
    --no-injection "${RESULT_ROOT}/seed${seed}/no_injection/predictions.jsonl" \
    --output "${RESULT_ROOT}/seed${seed}/full_vs_no_injection_bootstrap.json"
done
"${PYTHON_BIN}" "${ROOT_DIR}/tools/ocr_evidence/summarize_injection_ablation.py" --result-root "${RESULT_ROOT}"
"${PYTHON_BIN}" "${ROOT_DIR}/tools/ocr_evidence/paired_document_bootstrap.py" \
  --full "${RESULT_ROOT}/seed42/full/predictions.jsonl" \
         "${RESULT_ROOT}/seed123/full/predictions.jsonl" \
         "${RESULT_ROOT}/seed3407/full/predictions.jsonl" \
  --no-injection "${RESULT_ROOT}/seed42/no_injection/predictions.jsonl" \
                 "${RESULT_ROOT}/seed123/no_injection/predictions.jsonl" \
                 "${RESULT_ROOT}/seed3407/no_injection/predictions.jsonl" \
  --output "${RESULT_ROOT}/full_vs_no_injection_bootstrap_three_seed.json"
