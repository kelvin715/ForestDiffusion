#!/usr/bin/env bash
set -euo pipefail

cd /home/zhihan/tabular_gen/ForestDiffusion

# 为 joblib 指定临时目录，避免默认临时分区空间不足
export JOBLIB_TEMP_FOLDER="/srv/data/zhihan/joblib_tmp"
out_path="/home/zhihan/tabular_gen/ForestDiffusion/results_quantile_no_batch.csv"
log_path="/home/zhihan/tabular_gen/ForestDiffusion/logs_quantile_no_batch.txt"

DATASETS=${DATASETS:-tictactoe}
if [[ -z "${DATASETS}" ]]; then
  echo "DATASETS is empty. Please set DATASETS, e.g. DATASETS=wine"
  exit 1
fi

run_case() {
  local diffusion_type=$1
  local ycond=$2
  local n_batch=$3
  local device=$4
  local use_quantile=$5
  
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

# run_case mixed-flow False 1 3 False
# run_case flow False 1 3 False
# run_case mixed-flow False 1 3 True
run_case mixed-flow False 0 2 False
run_case mixed-flow False 0 2 True


# wait
# run_case vp True 1 0
# run_case flow True 1 1
# wait
# run_case vp False 1 1
# wait
