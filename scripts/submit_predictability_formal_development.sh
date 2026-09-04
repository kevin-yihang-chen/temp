#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "usage: $0 {chartqa|docvqa|hrbench} {train|validation} [--resume]" >&2
  exit 2
fi
benchmark=$1
role=$2
resume=${3:-}
if [[ -n "${resume}" && "${resume}" != "--resume" ]]; then
  echo "third argument must be --resume" >&2
  exit 2
fi
case "${benchmark}" in chartqa|docvqa|hrbench) ;; *) exit 2 ;; esac
case "${role}" in train|validation) ;; *) exit 2 ;; esac

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
mail_file="${repo_dir}/.slurm-notify-email"
protocol="${repo_dir}/configs/predictability_audit_v1.json"
execution_config="${repo_dir}/configs/predictability_formal_execution_v1.json"
worker="${repo_dir}/scripts/slurm_predictability_formal_development.sh"
finalizer="${repo_dir}/scripts/finalize_predictability_development_role.py"
collector_cli="${repo_dir}/src/beyond_entropy/cli.py"
backend_module="${repo_dir}/src/beyond_entropy/qwen_backend.py"
semantic_module="${repo_dir}/src/beyond_entropy/qwen_semantic.py"
features_module="${repo_dir}/src/beyond_entropy/predictability_features.py"
post_action_module="${repo_dir}/src/beyond_entropy/predictability_post_action.py"
image_ops_module="${repo_dir}/src/beyond_entropy/image_ops.py"
audit_module="${repo_dir}/src/beyond_entropy/predictability_audit.py"
artifacts_module="${repo_dir}/src/beyond_entropy/predictability_matrix_artifacts.py"
source_manifest="${repo_dir}/data/predictability-audit-v1/${benchmark}/${role}/manifest.jsonl"
run_root="${repo_dir}/artifacts/predictability-audit-v1/formal-development-v1"
run_dir="${run_root}/${benchmark}/${role}"

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm notification email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid formal development notification email" >&2
  exit 2
fi
if [[ ! -s "${source_manifest}" ]]; then
  echo "frozen development manifest is missing" >&2
  exit 2
fi
if [[ -e "${run_dir}/complete.json" ]]; then
  echo "formal development role is already sealed" >&2
  exit 2
fi
if [[ -e "${run_dir}" && "${resume}" != "--resume" ]]; then
  echo "partial role directory exists; explicit --resume is required" >&2
  exit 2
fi
if [[ ! -e "${run_dir}" && "${resume}" == "--resume" ]]; then
  echo "cannot resume a role directory that does not exist" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before formal development submission" >&2
  exit 2
fi
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
expected_states=$(jq -er --arg b "${benchmark}" --arg r "${role}" '.roles_in_submission_order[] | select(.benchmark == $b and .role == $r) | .states' "${execution_config}")
requested_time=$(jq -er --arg b "${benchmark}" --arg r "${role}" '.roles_in_submission_order[] | select(.benchmark == $b and .role == $r) | .requested_time' "${execution_config}")
checkpoint_interval=$(jq -er '.checkpoint_interval_states' "${execution_config}")
actual_states=$(awk 'NF { count += 1 } END { print count + 0 }' "${source_manifest}")
if [[ "${actual_states}" != "${expected_states}" ]]; then
  echo "development manifest state count differs from execution config" >&2
  exit 2
fi
if [[ "$(jq -r '.test_artifacts_authorized' "${execution_config}")" != "false" ]]; then
  echo "execution config unexpectedly authorizes test" >&2
  exit 2
fi
protocol_digest=$(sha256sum "${protocol}")
protocol_digest=${protocol_digest%% *}
if [[ "$(jq -er '.protocol.sha256' "${execution_config}")" != "${protocol_digest}" ]]; then
  echo "execution config does not bind the current protocol" >&2
  exit 2
fi

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
  echo "insufficient GPU quota for requested formal role walltime" >&2
  exit 2
fi
available_bytes=$(df -B1 --output=avail "${repo_dir}" | tail -1 | tr -d ' ')
if [[ ! "${available_bytes}" =~ ^[0-9]+$ || "${available_bytes}" -lt 10737418240 ]]; then
  echo "formal development requires at least 10 GiB free" >&2
  exit 2
fi

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
export_args="ALL,BE_PRED_EXPECTED_CODE_REVISION=${code_revision},BE_PRED_EXPECTED_GPU_TOKEN=H800,BE_PRED_BENCHMARK=${benchmark},BE_PRED_ROLE=${role},BE_PRED_RUN_ROOT=${run_root},BE_PRED_EXPECTED_STATES=${expected_states},BE_PRED_CHECKPOINT_INTERVAL=${checkpoint_interval},BE_PRED_WORKER_SHA256=$(digest "${worker}"),BE_PRED_FINALIZER_SHA256=$(digest "${finalizer}"),BE_PRED_CLI_SHA256=$(digest "${collector_cli}"),BE_PRED_BACKEND_SHA256=$(digest "${backend_module}"),BE_PRED_SEMANTIC_SHA256=$(digest "${semantic_module}"),BE_PRED_FEATURES_SHA256=$(digest "${features_module}"),BE_PRED_POST_ACTION_SHA256=$(digest "${post_action_module}"),BE_PRED_IMAGE_OPS_SHA256=$(digest "${image_ops_module}"),BE_PRED_AUDIT_SHA256=$(digest "${audit_module}"),BE_PRED_ARTIFACTS_SHA256=$(digest "${artifacts_module}"),BE_PRED_SOURCE_MANIFEST_SHA256=$(digest "${source_manifest}"),BE_PRED_PROTOCOL_SHA256=$(digest "${protocol}"),BE_PRED_EXECUTION_CONFIG_SHA256=$(digest "${execution_config}")"
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
  echo "could not parse formal development job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'predictability_formal_development_job_id=%s benchmark=%s role=%s states=%s time=%s code_revision=%s remaining_gpu_minutes_before_submit=%s\n' \
  "${job_id}" "${benchmark}" "${role}" "${expected_states}" "${requested_time}" "${code_revision}" "${gpu_remaining}"
