#!/usr/bin/env bash
set -euo pipefail

repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
matrix=configs/factorized_phase_c_training_matrix_v1.json
output_root=artifacts/factorized-potential-outcomes-v1/phase-c-training
mail_file="${repo}/.slurm-notify-email"
IFS= read -r notify_email < "${mail_file}"
[[ "${notify_email}" == "yihangc@connect.hku.hk" ]] || {
  echo "invalid notification email" >&2; exit 2;
}
[[ -z "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]] || {
  echo "tracked worktree must be clean before Phase-C training submission" >&2; exit 2;
}
cd "${repo}"

seeds=("$@")
if [[ "${#seeds[@]}" -eq 0 ]]; then seeds=(17 29 47); fi
timestamp=$(date +%Y%m%dT%H%M%S)
for seed in "${seeds[@]}"; do
  case "${seed}" in 17|29|47) ;; *) echo "invalid Phase-C seed: ${seed}" >&2; exit 2 ;; esac
  config_dir="${output_root}/plans/${timestamp}-seed-${seed}"
  materialized=$(PYTHONPATH=src "${python_bin}" \
    scripts/materialize_factorized_phase_c_training.py \
    --matrix "${matrix}" --repository-root "${repo}" \
    --seed "${seed}" --output-dir "${config_dir}")
  outcome_config=$(jq -er '.configs.outcome_only' <<< "${materialized}")
  counterfactual_config=$(jq -er '.configs.counterfactual_utility' <<< "${materialized}")
  factorized_config=$(jq -er '.configs.factorized_potential_outcomes' <<< "${materialized}")
  evaluation_config=$(jq -er '.evaluation' <<< "${materialized}")
  plan="${config_dir}/plan.json"
  plan_sha=$(PYTHONPATH=src "${python_bin}" scripts/freeze_cv_method_stage.py \
    --outcome-config "${outcome_config}" \
    --counterfactual-config "${counterfactual_config}" \
    --factorized-config "${factorized_config}" \
    --evaluation-config "${evaluation_config}" \
    --output-root "${output_root}" --plan "${plan}")
  /usr/local/slurm/bin/sbatch \
    --job-name="be-pc-train-s${seed}" \
    --mail-user="${notify_email}" --mail-type=ALL \
    --export="ALL,CV_METHOD_PLAN=${repo}/${plan},CV_METHOD_PLAN_SHA256=${plan_sha}" \
    scripts/slurm_factorized_phase_c_training.sh
done

