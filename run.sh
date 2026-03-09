#!/usr/bin/env bash
set -euo pipefail

# if don't use iterator training, we need to specify the temporary folder with large space for joblib to avoid out of memory errors
# export JOBLIB_TEMP_FOLDER="/srv/data/zhihan/joblib_tmp"

out_path="./results.csv"
log_path="./logs.txt"

DATASETS=${DATASETS:-tictactoe}
if [[ -z "${DATASETS}" ]]; then
  echo "DATASETS is empty. Please set DATASETS, e.g. DATASETS=wine"
  exit 1
fi

run_case() {
  local diffusion_type=$1 # flow (flow-matching), mixed-flow (VFM)
  local ycond=$2 # if True, use ycond for training/generation.
  local n_batch=$3 # 0: no batch, 1: use iterator training
  local device=$4 
  local use_quantile=$5 # if True, apply QuantileTransformer on numerical columns before training/generation and inverse-transform fake samples before evaluation.
  
  CUDA_VISIBLE_DEVICES=${device} python script_generation.py \
    --methods forest_diffusion \
    --diffusion_type "${diffusion_type}" \
    --out_path "${out_path}" \
    --n_t 50 \
    --nexp 1 \
    --ngen 3 \
    --n_tries 1 \
    --duplicate_K 100 \
    --n_batch "${n_batch}" \
    --n_jobs 1 \
    --use_quantile "${use_quantile}" \
    --ycond "${ycond}" >> "${log_path}" 2>&1 
}

run_case mixed-flow False 1 0 False
run_case flow False 1 0 False