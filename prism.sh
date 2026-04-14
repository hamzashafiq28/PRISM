#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

: "${CHECKPOINT:?Set CHECKPOINT}"
: "${FAISS_DIR:?Set FAISS_DIR}"
: "${TEST_JSON:?Set TEST_JSON}"
: "${TRAIN_VAL_CSV:?Set TRAIN_VAL_CSV}"
: "${TEST_CSV:?Set TEST_CSV}"
: "${PATIENT_JSON:?Set PATIENT_JSON}"
: "${H5_ROOT:?Set H5_ROOT}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

ECG_CACHE="${ECG_CACHE:-}"
MEDGEMMA_MODEL="${MEDGEMMA_MODEL:-google/medgemma-4b-it}"

mkdir -p "${OUTPUT_DIR}"

python "${ROOT_DIR}/llava/train/evaluate.py" \
  --checkpoint "${CHECKPOINT}" \
  --faiss_dir "${FAISS_DIR}" \
  --test_json "${TEST_JSON}" \
  --train_val_csv "${TRAIN_VAL_CSV}" \
  --test_csv "${TEST_CSV}" \
  --patient_json "${PATIENT_JSON}" \
  --h5_root "${H5_ROOT}" \
  --ecg_cache "${ECG_CACHE}" \
  --output_dir "${OUTPUT_DIR}" \
  --medgemma "${MEDGEMMA_MODEL}"
