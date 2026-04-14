#!/bin/bash

set -e

module load 2025
module load CUDA/12.8.0

eval "$(conda shell.bash hook)"
conda activate cndp

CONDA_ENV=/projects/prjs1786/Video_Understanding/Video-XL/Video-XL-2/train/miniconda3/envs/cndp
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/gpfs/work3/0/prjs1786/CNDP/New_Code/LLaVA:$PYTHONPATH
export PYTORCH_ALLOC_CONF=expandable_segments:True

PYTHON=${CONDA_ENV}/bin/python

TORCH_CUDNN_SDPA_ENABLED=0 
TORCH_SDPA_ENABLE_CUDNN=0 


CHECKPOINT=/projects/prjs1786/CNDP/New_Code/LLaVA/stage2_ret_out_s1_260k_wof_new/checkpoint_best
FAISS_DIR=/projects/prjs1786/CNDP/New_Code/LLaVA/faiss_test_retrieved

TEST_JSON=/projects/prjs1786/CNDP/New_Code/test_raft2.json
TRAIN_VAL_CSV=/projects/prjs1786/mimic_data/CHRONO/1_dataset/2_ehr_ecg_text/mds_ed_train_val_merged.csv
TEST_CSV=/projects/prjs1786/mimic_data/CHRONO/1_dataset/2_ehr_ecg_text/mds_ed_test.csv
PATIENT_JSON=/projects/prjs1786/mimic_data/CHRONO/data/memmap/patient_data_96h.json
H5_ROOT=/projects/prjs1786/mimic_data/mimic_h5
ECG_CACHE=/projects/prjs1786/CNDP/New_Code/LLaVA/ecg_cache
OUTPUT_DIR=/projects/prjs1786/CNDP/New_Code/LLaVA/stage2_ret_out/eval_prompt_test_raft2

mkdir -p /projects/prjs1786/CNDP/New_Code/LLaVA/logs
mkdir -p "$OUTPUT_DIR"

python /gpfs/work3/0/prjs1786/CNDP/New_Code/LLaVA/llava/train/eval_stage2_retrieval_prompt.py \
    --checkpoint       "$CHECKPOINT"    \
    --faiss_dir        "$FAISS_DIR"     \
    --test_json        "$TEST_JSON"     \
    --train_val_csv    "$TRAIN_VAL_CSV" \
    --test_csv         "$TEST_CSV"      \
    --patient_json     "$PATIENT_JSON"  \
    --h5_root          "$H5_ROOT"       \
    --ecg_cache        "$ECG_CACHE"     \
    --output_dir       "$OUTPUT_DIR"    \
    --medgemma         google/medgemma-4b-it \
    --batch_size       8                \
    --num_workers      16               \
    --ehr_d_model      192              \
    --ehr_n_heads      4                \
    --ehr_n_layers     3                \
    --n_ecg_tokens     8                \
    --n_ehr_tokens     8                \
    --n_hist_tokens    8                \
    --n_ret_ehr_tokens 8                \
    --n_ret_ecg_tokens 8                \
    --proj_hidden      1024             \
    --max_seq_len      2048             \
    --log_interval     50
