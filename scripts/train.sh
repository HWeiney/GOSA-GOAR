#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${ROOT_DIR}"
CONFIG_PATH=${1:-${ROOT_DIR}/configs/gosa_ocr_prompt_goar.env}
CONFIG_PATH=$(realpath "${CONFIG_PATH}")
[[ -f "${CONFIG_PATH}" ]] || { echo "Config not found: ${CONFIG_PATH}" >&2; exit 2; }

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OCR_MODE USE_OCR ENABLE_LOCAL_PE ENABLE_GLOBAL_PE
export ENABLE_GOAR=${ENABLE_GOAR:-False}
export GOAR_INFERENCE_BOX_MODE=${GOAR_INFERENCE_BOX_MODE:-fusion}
export GOAR_USE_ROI_FEATURE=${GOAR_USE_ROI_FEATURE:-True}
export GOAR_SOURCE_FUSION_MODE=${GOAR_SOURCE_FUSION_MODE:-uncertainty}
export GOAR_REFINE_OCR_ANCHOR=${GOAR_REFINE_OCR_ANCHOR:-True}
export GOAR_BOTTLENECK=${GOAR_BOTTLENECK:-256}
export GOAR_MAX_OCR_LINES=${GOAR_MAX_OCR_LINES:-128}
export GOAR_POINTER_LOSS_WEIGHT=${GOAR_POINTER_LOSS_WEIGHT:-0.2}
export GOAR_BRANCH_LOSS_WEIGHT=${GOAR_BRANCH_LOSS_WEIGHT:-0.25}
export GOAR_UNCERTAINTY_LOSS_WEIGHT=${GOAR_UNCERTAINTY_LOSS_WEIGHT:-0.1}
export GOAR_LOSS_WEIGHT=${GOAR_LOSS_WEIGHT:-1.0}
export GLOBAL_PE_FUSION=${GLOBAL_PE_FUSION:-add}
export GLOBAL_PE_GATE_INIT=${GLOBAL_PE_GATE_INIT:-0.1}
export GLOBAL_PE_THUMBNAIL=${GLOBAL_PE_THUMBNAIL:-full}
export GLOBAL_PE_COORD_MODE=${GLOBAL_PE_COORD_MODE:-real}

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH=${MODEL_PATH:-${ROOT_DIR}/pretrained/InternVL3_5-2B}
TRAIN_DATA=${TRAIN_DATA:-"${ROOT_DIR}/annotations/SRFUND/srfund_zh_train.jsonl ${ROOT_DIR}/annotations/SRFUND/srfund_en_train.jsonl"}
VAL_DATA=${VAL_DATA:-"${ROOT_DIR}/annotations/SRFUND/srfund_zh_val.jsonl ${ROOT_DIR}/annotations/SRFUND/srfund_en_val.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/outputs/${EXPERIMENT}}
read -r -a TRAIN_FILES <<< "${TRAIN_DATA}"
read -r -a VAL_FILES <<< "${VAL_DATA}"

modules_to_save=()
[[ "${ENABLE_LOCAL_PE,,}" == true ]] && modules_to_save+=(pos_embed_2d)
[[ "${ENABLE_GLOBAL_PE,,}" == true ]] && modules_to_save+=(global_spatial_encoder)
[[ "${ENABLE_GOAR,,}" == true ]] && modules_to_save+=(goar_adapter)

cmd=("${PYTHON_BIN}" -m swift.cli.main sft
  --model "${MODEL_PATH}"
  --dataset "${TRAIN_FILES[@]}"
  --val_dataset "${VAL_FILES[@]}"
  --output_dir "${OUTPUT_DIR}"
  --max_length "${MAX_LENGTH:-8192}"
  --per_device_train_batch_size "${BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${GRAD_ACC:-16}"
  --num_train_epochs "${EPOCHS:-3}"
  --learning_rate "${LEARNING_RATE:-1e-4}"
  --seed "${SEED:-42}"
  --data_seed "${DATA_SEED:-${SEED:-42}}"
  --save_steps "${SAVE_STEPS:-200}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}"
  --logging_steps "${LOGGING_STEPS:-5}"
  --gradient_checkpointing true
  --torch_dtype bfloat16
  --bf16 true
  --freeze_vit true
  --freeze_aligner true
  --tuner_type lora
  --lora_rank "${LORA_RANK:-16}"
  --target_modules all-linear
  --freeze_llm false)

if ((${#modules_to_save[@]})); then
  cmd+=(--modules_to_save "${modules_to_save[@]}")
fi

if [[ "${DRY_RUN:-false}" == true ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
exec "${cmd[@]}"
