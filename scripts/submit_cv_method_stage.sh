#!/usr/bin/env bash
set -euo pipefail
repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
stage=${1:?usage: submit_cv_method_stage.sh smoke|pilot}
case "${stage}" in
  smoke)
    outcome_config=configs/cv_method_smoke_outcome_only_v1.json
    counterfactual_config=configs/cv_method_smoke_counterfactual_v1.json
    evaluation_config=configs/cv_method_smoke_evaluation_v1.json
    output_root=artifacts/cv-method-v1/phase-a-smoke
    ;;
  pilot)
    outcome_config=configs/cv_method_pilot_outcome_only_v1.json
    counterfactual_config=configs/cv_method_pilot_counterfactual_v1.json
    evaluation_config=configs/cv_method_pilot_evaluation_v1.json
    output_root=artifacts/cv-method-v1/phase-b-pilot
    ;;
  *) echo "unknown stage: ${stage}" >&2; exit 2 ;;
esac
cd "${repo_dir}"
timestamp=$(date +%Y%m%dT%H%M%S)
plan="${output_root}/plans/${timestamp}.json"
plan_sha=$(PYTHONPATH=src "${python_bin}" scripts/freeze_cv_method_stage.py \
  --outcome-config "${outcome_config}" \
  --counterfactual-config "${counterfactual_config}" \
  --evaluation-config "${evaluation_config}" \
  --output-root "${output_root}" --plan "${plan}")
sbatch --export="ALL,CV_METHOD_PLAN=${repo_dir}/${plan},CV_METHOD_PLAN_SHA256=${plan_sha}" \
  scripts/slurm_cv_method_stage.sh
