#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "usage: $0 {chartqa|docvqa|hrbench} [state-semantic|relational|relational-audit]" >&2
  exit 2
fi
benchmark=$1
case "${benchmark}" in chartqa|docvqa|hrbench) ;; *) exit 2 ;; esac
variant=${2:-state-semantic}
case "${variant}" in state-semantic|relational|relational-audit) ;; *) exit 2 ;; esac

repo=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo}/.slurm-notify-email"
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid notification email" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before submission" >&2
  exit 2
fi

resolve_one() {
  local pattern=$1
  local matches=()
  shopt -s nullglob
  matches=(${pattern})
  shopt -u nullglob
  if [[ "${#matches[@]}" -ne 1 ]]; then
    echo "expected exactly one input for ${pattern}; found ${#matches[@]}" >&2
    exit 2
  fi
  printf '%s\n' "${matches[0]}"
}

train_dir=$(resolve_one "${repo}/artifacts/sequential-acquisition-v1/smoke-${benchmark}-train-256-v1/job-*")
validation_dir=$(resolve_one "${repo}/artifacts/sequential-acquisition-v1/smoke-${benchmark}-validation-128-v1/job-*")
train_features="${train_dir}/features.pt"
validation_features="${validation_dir}/features.pt"
validation_rollouts="${validation_dir}/rollouts.jsonl"
for path in "${train_features}" "${validation_features}" "${validation_rollouts}" \
  "${train_dir}/execution.json" "${validation_dir}/execution.json"; do
  if [[ ! -s "${path}" ]]; then
    echo "missing completed pilot input: ${path}" >&2
    exit 2
  fi
done
for path in "${train_dir}/execution.json" "${validation_dir}/execution.json"; do
  if ! grep -q '"status": "completed"' "${path}"; then
    echo "pilot input is not marked completed: ${path}" >&2
    exit 2
  fi
done

worker="${repo}/scripts/slurm_sequential_critic_pilot.sh"
trainer="${repo}/scripts/train_acquisition_critic.py"
evaluator="${repo}/scripts/eval_sequential_policy.py"
critic_module="${repo}/src/beyond_entropy/acquisition_critic.py"
metrics_module="${repo}/src/beyond_entropy/sequential_metrics.py"
policy_module="${repo}/src/beyond_entropy/stopping_policy.py"
schema_module="${repo}/src/beyond_entropy/sequential_schema.py"
if [[ "${variant}" == "relational" || "${variant}" == "relational-audit" ]]; then
  config="${repo}/configs/sequential_${benchmark}_relational.yaml"
  if [[ "${variant}" == "relational-audit" ]]; then
    run_root="${repo}/artifacts/sequential-acquisition-v1/critic-${benchmark}-relational-audit-256x128-v1"
  else
    run_root="${repo}/artifacts/sequential-acquisition-v1/critic-${benchmark}-relational-pilot-256x128-v1"
  fi
else
  config="${repo}/configs/sequential_${benchmark}.yaml"
  run_root="${repo}/artifacts/sequential-acquisition-v1/critic-${benchmark}-pilot-256x128-v1"
fi
if [[ -e "${run_root}" ]]; then
  echo "refusing to reuse critic pilot root: ${run_root}" >&2
  exit 2
fi
digest() { sha256sum "$1" | cut -d ' ' -f 1; }
revision=$(git -C "${repo}" rev-parse HEAD)
exports="ALL,BE_SEQ_BENCHMARK=${benchmark},BE_SEQ_CONFIG=${config},BE_SEQ_TRAIN_FEATURES=${train_features},BE_SEQ_VALIDATION_FEATURES=${validation_features},BE_SEQ_VALIDATION_ROLLOUTS=${validation_rollouts},BE_SEQ_RUN_ROOT=${run_root},BE_SEQ_CODE_REVISION=${revision},BE_SEQ_WORKER_SHA256=$(digest "${worker}"),BE_SEQ_TRAINER_SHA256=$(digest "${trainer}"),BE_SEQ_EVALUATOR_SHA256=$(digest "${evaluator}"),BE_SEQ_CRITIC_MODULE_SHA256=$(digest "${critic_module}"),BE_SEQ_METRICS_SHA256=$(digest "${metrics_module}"),BE_SEQ_POLICY_SHA256=$(digest "${policy_module}"),BE_SEQ_SCHEMA_SHA256=$(digest "${schema_module}"),BE_SEQ_CONFIG_SHA256=$(digest "${config}"),BE_SEQ_TRAIN_FEATURES_SHA256=$(digest "${train_features}"),BE_SEQ_VALIDATION_FEATURES_SHA256=$(digest "${validation_features}"),BE_SEQ_VALIDATION_ROLLOUTS_SHA256=$(digest "${validation_rollouts}")"

/usr/local/slurm/bin/sbatch \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export="${exports}" \
  "${worker}"
