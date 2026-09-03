#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 3 ]]; then
  echo "usage: $0 {h100|h800|rtx_4090} [state-count] [chartqa|docvqa|hrbench]" >&2
  exit 2
fi
gpu_type=$1
smoke_count=${2:-1}
benchmark=${3:-chartqa}
if [[ ! "${smoke_count}" =~ ^[1-9][0-9]*$ || "${smoke_count}" -gt 64 ]]; then
  echo "predictability smoke state-count must be in [1, 64]" >&2
  exit 2
fi
case "${benchmark}" in
  chartqa|docvqa|hrbench) ;;
  *)
    echo "unsupported predictability smoke benchmark: ${benchmark}" >&2
    exit 2
    ;;
esac
case "${gpu_type}" in
  h100)
    partition=q-hgpu-small
    gres=gpu:h100:1
    gpu_token=H100
    ;;
  h800)
    partition=q-hgpu-small
    gres=gpu:h800:1
    gpu_token=H800
    ;;
  rtx_4090)
    partition=debug
    gres=gpu:rtx_4090:1
    gpu_token=4090
    ;;
  *)
    echo "unsupported predictability smoke GPU type: ${gpu_type}" >&2
    exit 2
    ;;
esac

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
mail_file="${repo_dir}/.slurm-notify-email"
source_manifest="${repo_dir}/data/predictability-audit-v1/${benchmark}/train/manifest.jsonl"
protocol="${repo_dir}/configs/predictability_audit_v1.json"
worker="${repo_dir}/scripts/slurm_predictability_feature_smoke.sh"
collector_cli="${repo_dir}/src/beyond_entropy/cli.py"
backend_module="${repo_dir}/src/beyond_entropy/qwen_backend.py"
semantic_module="${repo_dir}/src/beyond_entropy/qwen_semantic.py"
features_module="${repo_dir}/src/beyond_entropy/predictability_features.py"
audit_module="${repo_dir}/src/beyond_entropy/predictability_audit.py"
run_root="${repo_dir}/artifacts/predictability-audit-v1/real-feature-smoke-${benchmark}-${smoke_count}-v1"
model=Qwen/Qwen2.5-VL-3B-Instruct
model_revision=66285546d2b821cf421d4f5eb2576359d3770cd3

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm notification email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid predictability smoke notification email" >&2
  exit 2
fi
if [[ ! -s "${source_manifest}" ]]; then
  echo "frozen predictability allocation is not complete" >&2
  exit 2
fi
if [[ -e "${run_root}" ]]; then
  echo "refusing to reuse predictability smoke root: ${run_root}" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before predictability smoke submission" >&2
  exit 2
fi
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

quota_output=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(awk '/GPU Quota in Minutes:/ {print $5}' <<< "${quota_output}")
gpu_used=$(awk '/GPU Quota in Minutes:/ {print $7}' <<< "${quota_output}")
gpu_limit=${gpu_limit%,}
if [[ ! "${gpu_limit}" =~ ^[0-9]+$ || ! "${gpu_used}" =~ ^[0-9]+$ ]]; then
  echo "could not parse live GPU quota" >&2
  exit 2
fi
gpu_remaining=$((gpu_limit - gpu_used))
if [[ "${gpu_remaining}" -lt 60 ]]; then
  echo "predictability smoke needs a 60 GPU-minute reserve; remaining=${gpu_remaining}" >&2
  exit 2
fi

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"${python_bin}" - "${model}" "${model_revision}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_files_only=True)
PY

digest() {
  sha256sum "$1" | cut -d ' ' -f 1
}
export_args="ALL,BE_PRED_EXPECTED_CODE_REVISION=${code_revision},BE_PRED_EXPECTED_GPU_TOKEN=${gpu_token},BE_PRED_BENCHMARK=${benchmark},BE_PRED_RUN_ROOT=${run_root},BE_PRED_SMOKE_COUNT=${smoke_count},BE_PRED_WORKER_SHA256=$(digest "${worker}"),BE_PRED_CLI_SHA256=$(digest "${collector_cli}"),BE_PRED_BACKEND_SHA256=$(digest "${backend_module}"),BE_PRED_SEMANTIC_SHA256=$(digest "${semantic_module}"),BE_PRED_FEATURES_SHA256=$(digest "${features_module}"),BE_PRED_AUDIT_SHA256=$(digest "${audit_module}"),BE_PRED_SOURCE_MANIFEST_SHA256=$(digest "${source_manifest}"),BE_PRED_PROTOCOL_SHA256=$(digest "${protocol}")"

submission=$(
  /usr/local/slurm/bin/sbatch \
    --partition="${partition}" \
    --gres="${gres}" \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${worker}"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse predictability smoke job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'predictability_feature_smoke_job_id=%s gpu_type=%s benchmark=%s states=%s code_revision=%s remaining_gpu_minutes_before_submit=%s\n' \
  "${job_id}" "${gpu_type}" "${benchmark}" "${smoke_count}" "${code_revision}" "${gpu_remaining}"
