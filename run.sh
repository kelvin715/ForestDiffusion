#!/usr/bin/env bash
set -euo pipefail

cd /proj-vertical-llms-pvc/users/zhihan/tabular_gen/ForestDiffusion
out_path="/proj-vertical-llms-pvc/users/zhihan/tabular_gen/ForestDiffusion/results_new_xgboost.csv"
log_path="/proj-vertical-llms-pvc/users/zhihan/tabular_gen/ForestDiffusion/logs_new_xgboost.txt"

DATASETS=${DATASETS:-adult}
if [[ -z "${DATASETS}" ]]; then
  echo "DATASETS is empty. Please set DATASETS, e.g. DATASETS=wine"
  exit 1
fi

run_case() {
  local diffusion_type=$1
  local ycond=$2
  local n_batch=$3
  local device=$4
  
  CUDA_VISIBLE_DEVICES=${device} python script_generation.py \
    --methods forest_diffusion \
    --diffusion_type "${diffusion_type}" \
    --out_path "${out_path}" \
    --datasets "${DATASETS}" \
    --n_t 10 \
    --nexp 1 \
    --ngen 1 \
    --n_tries 1 \
    --duplicate_K 10 \
    --n_batch "${n_batch}" \
    --n_jobs 4 \
    --ycond "${ycond}" >> "${log_path}" 2>&1 &
}

# run_case flow False 1 0
run_case mixed-flow False 0 1
# wait
# run_case vp True 1 0
# run_case flow True 1 1
# wait
# run_case vp False 1 1
# wait
