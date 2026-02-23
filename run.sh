#!/usr/bin/env bash
set -euo pipefail

cd /proj-vertical-llms-pvc/users/zhihan/tabular_gen/ForestDiffusion
out_path="/proj-vertical-llms-pvc/users/zhihan/tabular_gen/ForestDiffusion/results_all.csv"

DATASETS=${DATASETS:-car iris wine parkinsons climate_model_crashes concrete_compression yacht_hydrodynamics airfoil_self_noise connectionist_bench_sonar ionosphere qsar_biodegradation seeds glass ecoli yeast libras planning_relax blood_transfusion breast_cancer_diagnostic connectionist_bench_vowel concrete_slump wine_quality_red wine_quality_white california bean tictactoe congress}
if [[ -z "${DATASETS}" ]]; then
  echo "DATASETS is empty. Please set DATASETS, e.g. DATASETS=wine"
  exit 1
fi

run_case() {
  local diffusion_type=$1
  local ycond=$2
  local n_batch=$3
  python script_generation.py \
    --methods forest_diffusion \
    --diffusion_type "${diffusion_type}" \
    --out_path "${out_path}" \
    --n_t 50 \
    --nexp 2 \
    --ngen 2 \
    --n_tries 2 \
    --duplicate_K 100 \
    --n_batch "${n_batch}" \
    --ycond "${ycond}" &
}

run_case flow True 1  
# run_case flow False 1
run_case vp True 1
# run_case vp False 1
# run_case mixed-flow True 0
run_case mixed-flow False 0

wait
