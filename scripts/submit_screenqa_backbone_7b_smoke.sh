#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 {h100|h800|rtx_4090}" >&2
  exit 2
fi
gpu_type=$1
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
    echo "unsupported Qwen-7B smoke GPU type: ${gpu_type}" >&2
    exit 2
    ;;
esac

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
mail_file="${repo_dir}/.slurm-notify-email"
worker="${repo_dir}/scripts/slurm_screenqa_backbone_7b_smoke.sh"
collector_cli="${repo_dir}/src/beyond_entropy/cli.py"
backend_module="${repo_dir}/src/beyond_entropy/qwen_backend.py"
rollout_module="${repo_dir}/src/beyond_entropy/rollout.py"
crops_module="${repo_dir}/src/beyond_entropy/crops.py"
benchmarks_module="${repo_dir}/src/beyond_entropy/benchmarks.py"
score_module="${repo_dir}/src/beyond_entropy/answer_likelihood.py"
scorer="${repo_dir}/scripts/score_visual_action_answer_nll.py"
verifier_module="${repo_dir}/src/beyond_entropy/backbone_smoke.py"
verifier="${repo_dir}/scripts/verify_backbone_diagnostic_smoke.py"
run_root="${repo_dir}/artifacts/screenqa-train-factorized-v1/backbone-7b-diagnostic-v1/smoke-${gpu_type}-v1"
model=Qwen/Qwen2.5-VL-7B-Instruct
model_revision=cc594898137f460bfe9f0759e9844b3ce807cfb5

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm notification email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid Qwen-7B smoke notification email" >&2
  exit 2
fi
if [[ -e "${run_root}" ]]; then
  echo "refusing to reuse Qwen-7B smoke root: ${run_root}" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before Qwen-7B smoke submission" >&2
  exit 2
fi
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

active_jobs=$(/usr/local/slurm/bin/squeue -h -u "${USER}" -t PENDING,RUNNING,CONFIGURING,COMPLETING | wc -l)
if [[ "${active_jobs}" -ne 0 ]]; then
  echo "Qwen-7B smoke requires the account's sole concurrent job slot to be free" >&2
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
if [[ "${gpu_remaining}" -lt 120 ]]; then
  echo "Qwen-7B smoke needs a 120 GPU-minute reserve; remaining=${gpu_remaining}" >&2
  exit 2
fi

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"${python_bin}" - "${model}" "${model_revision}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_files_only=True)
PY

worker_sha256=$(sha256sum "${worker}" | cut -d ' ' -f 1)
cli_sha256=$(sha256sum "${collector_cli}" | cut -d ' ' -f 1)
backend_sha256=$(sha256sum "${backend_module}" | cut -d ' ' -f 1)
rollout_sha256=$(sha256sum "${rollout_module}" | cut -d ' ' -f 1)
crops_sha256=$(sha256sum "${crops_module}" | cut -d ' ' -f 1)
benchmarks_sha256=$(sha256sum "${benchmarks_module}" | cut -d ' ' -f 1)
score_module_sha256=$(sha256sum "${score_module}" | cut -d ' ' -f 1)
scorer_sha256=$(sha256sum "${scorer}" | cut -d ' ' -f 1)
verifier_module_sha256=$(sha256sum "${verifier_module}" | cut -d ' ' -f 1)
verifier_cli_sha256=$(sha256sum "${verifier}" | cut -d ' ' -f 1)
export_args="ALL,BE_BB7_EXPECTED_CODE_REVISION=${code_revision},BE_BB7_EXPECTED_GPU_TOKEN=${gpu_token},BE_BB7_RUN_ROOT=${run_root},BE_BB7_WORKER_SHA256=${worker_sha256},BE_BB7_CLI_SHA256=${cli_sha256},BE_BB7_BACKEND_SHA256=${backend_sha256},BE_BB7_ROLLOUT_SHA256=${rollout_sha256},BE_BB7_CROPS_SHA256=${crops_sha256},BE_BB7_BENCHMARKS_SHA256=${benchmarks_sha256},BE_BB7_SCORE_MODULE_SHA256=${score_module_sha256},BE_BB7_SCORER_SHA256=${scorer_sha256},BE_BB7_VERIFIER_MODULE_SHA256=${verifier_module_sha256},BE_BB7_VERIFIER_CLI_SHA256=${verifier_cli_sha256}"

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
  echo "could not parse Qwen-7B smoke job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'screenqa_backbone_7b_smoke_job_id=%s gpu_type=%s code_revision=%s remaining_gpu_minutes_before_submit=%s\n' \
  "${job_id}" "${gpu_type}" "${code_revision}" "${gpu_remaining}"
