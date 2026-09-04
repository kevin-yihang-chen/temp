#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <test-transaction-plan.json> <expected-plan-sha256>" >&2
  exit 2
fi
plan=$(realpath "$1")
expected_plan_sha256=$2
repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
mail_file="${repo_dir}/.slurm-notify-email"
worker="${repo_dir}/scripts/slurm_predictability_formal_test_once.sh"
starter="${repo_dir}/scripts/start_predictability_test_transaction.py"
finalizer="${repo_dir}/scripts/finalize_predictability_test_role.py"
spec_builder="${repo_dir}/scripts/build_predictability_test_input_spec.py"
evaluator="${repo_dir}/scripts/evaluate_predictability_matrix_test_once.py"
renderer="${repo_dir}/scripts/render_predictability_audit.py"
transaction_module="${repo_dir}/src/beyond_entropy/predictability_test_transaction.py"
artifacts_module="${repo_dir}/src/beyond_entropy/predictability_matrix_artifacts.py"
matrix_module="${repo_dir}/src/beyond_entropy/predictability_matrix.py"
verdict_module="${repo_dir}/src/beyond_entropy/predictability_verdict.py"

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm notification email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid formal test notification email" >&2
  exit 2
fi
actual_plan_sha256=$(sha256sum "${plan}")
actual_plan_sha256=${actual_plan_sha256%% *}
if [[ "${actual_plan_sha256}" != "${expected_plan_sha256}" ]]; then
  echo "formal test transaction plan SHA-256 mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain)" ]]; then
  echo "worktree must be fully clean before formal test submission" >&2
  exit 2
fi
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "$(jq -er '.code_revision' "${plan}")" != "${code_revision}" ]]; then
  echo "formal test plan code revision differs from clean HEAD" >&2
  exit 2
fi
for output_key in access_ledger test_input_spec report_output final_audit_output; do
  output=$(jq -er --arg key "${output_key}" '.[$key]' "${plan}")
  if [[ -e "${output}" ]]; then
    echo "formal test transaction already started or completed" >&2
    exit 2
  fi
done

requested_time=$(jq -er '.requested_time' "$(jq -er '.execution_config.path' "${plan}")")
quota_output=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(awk '/GPU Quota in Minutes:/ {print $5}' <<< "${quota_output}")
gpu_used=$(awk '/GPU Quota in Minutes:/ {print $7}' <<< "${quota_output}")
gpu_limit=${gpu_limit%,}
if [[ ! "${gpu_limit}" =~ ^[0-9]+$ || ! "${gpu_used}" =~ ^[0-9]+$ ]]; then
  echo "could not parse live GPU quota" >&2
  exit 2
fi
gpu_remaining=$((gpu_limit - gpu_used))
requested_seconds=$(awk -F: '{print ($1*3600)+($2*60)+$3}' <<< "${requested_time}")
requested_minutes=$(((requested_seconds + 59) / 60))
if (( gpu_remaining < requested_minutes )); then
  echo "insufficient GPU quota for one-shot formal test" >&2
  exit 2
fi
available_bytes=$(df -B1 --output=avail "${repo_dir}" | tail -1 | tr -d ' ')
if [[ ! "${available_bytes}" =~ ^[0-9]+$ || "${available_bytes}" -lt 10737418240 ]]; then
  echo "formal test requires at least 10 GiB free" >&2
  exit 2
fi

protocol=$(jq -er '.protocol.path' "${plan}")
model=$(jq -er '.feature_extraction.model' "${protocol}")
model_revision=$(jq -er '.feature_extraction.model_revision' "${protocol}")
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"${python_bin}" - "${model}" "${model_revision}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_files_only=True)
PY

digest() { sha256sum "$1" | cut -d ' ' -f 1; }
export_args="ALL,BE_PRED_TEST_PLAN=${plan},BE_PRED_TEST_PLAN_SHA256=${expected_plan_sha256},BE_PRED_TEST_EXPECTED_CODE_REVISION=${code_revision},BE_PRED_TEST_EXPECTED_GPU_TOKEN=H800,BE_PRED_TEST_WORKER_SHA256=$(digest "${worker}"),BE_PRED_TEST_STARTER_SHA256=$(digest "${starter}"),BE_PRED_TEST_FINALIZER_SHA256=$(digest "${finalizer}"),BE_PRED_TEST_SPEC_BUILDER_SHA256=$(digest "${spec_builder}"),BE_PRED_TEST_EVALUATOR_SHA256=$(digest "${evaluator}"),BE_PRED_TEST_RENDERER_SHA256=$(digest "${renderer}"),BE_PRED_TEST_TRANSACTION_MODULE_SHA256=$(digest "${transaction_module}"),BE_PRED_TEST_ARTIFACTS_MODULE_SHA256=$(digest "${artifacts_module}"),BE_PRED_TEST_MATRIX_MODULE_SHA256=$(digest "${matrix_module}"),BE_PRED_TEST_VERDICT_MODULE_SHA256=$(digest "${verdict_module}")"
submission=$(
  /usr/local/slurm/bin/sbatch \
    --partition=q-hgpu-small \
    --gres=gpu:h800:1 \
    --time="${requested_time}" \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${worker}"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse formal test job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'predictability_formal_test_job_id=%s time=%s code_revision=%s remaining_gpu_minutes_before_submit=%s\n' \
  "${job_id}" "${requested_time}" "${code_revision}" "${gpu_remaining}"
