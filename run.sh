#!/usr/bin/env bash
set -euo pipefail

cd /home/zhihan/tabular_gen/ForestDiffusion-OT

PYTHON="${PYTHON:-/home/zhihan/tabular_gen/venv/bin/python}"

export JOBLIB_TEMP_FOLDER="/srv/data/zhihan/joblib_tmp"
out_path="/home/zhihan/tabular_gen/ForestDiffusion-OT/results_ot_new_dataset.csv"
log_path="/home/zhihan/tabular_gen/ForestDiffusion-OT/logs_ot_new_dataset.txt"

# DATASETS=${DATASETS:-"iris wine parkinsons climate_model_crashes concrete_compression yacht_hydrodynamics airfoil_self_noise connectionist_bench_sonar ionosphere qsar_biodegradation seeds glass ecoli yeast libras planning_relax blood_transfusion breast_cancer_diagnostic connectionist_bench_vowel concrete_slump wine_quality_red wine_quality_white california bean tictactoe congress car"}
DATASETS=${DATASETS:-"news beijing default adult credit-g shoppers magic"}
# DATASETS=${DATASETS:-"credit-g"}

# Optional: n_t_sampling sweep (e.g. N_T_SAMPLING_LIST="10,20,30,50" N_T_SAMPLING_REPEATS=3)
# N_T_SAMPLING_LIST=${N_T_SAMPLING_LIST:-"10,20,30,40,50,60,70,80,90,100"}
N_T_SAMPLING_LIST=${N_T_SAMPLING_LIST:-"30,40,50,60"}
N_T_SAMPLING_REPEATS=${N_T_SAMPLING_REPEATS:-3}

run_case() {
  local diffusion_type=$1
  local ycond=$2
  local n_batch=$3
  local device=$4
  local use_quantile=$5
  local efvfm_style_impute=$6 # if True, impute missing values like ef-vfm (num mean, cat most_frequent) on train/test before training and TabMetrics.
  local ot_mode=$7
  
  local extra_args=()
  if [[ -n "${N_T_SAMPLING_LIST}" ]]; then
    extra_args+=(--n_t_sampling_list "${N_T_SAMPLING_LIST}" --n_t_sampling_repeats "${N_T_SAMPLING_REPEATS}")
  fi

  CUDA_VISIBLE_DEVICES=${device} "${PYTHON}" script_generation.py \
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
    --use_ot "${ot_mode}" \
    --ycond "${ycond}" \
    --datasets "${dataset}" \
    --efvfm_style_impute "${efvfm_style_impute}" \
    "${extra_args[@]}" \
    >> "${log_path}" 2>&1 
}

for dataset in ${DATASETS}; do
  echo "Running dataset: ${dataset}" | tee -a "${log_path}"

  run_case mixed-flow False 0 1 True True False

  echo "Finished dataset: ${dataset}" | tee -a "${log_path}"
done

