#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "--resume" ) ]]; then
  echo "usage: $0 [--resume]" >&2
  exit 2
fi
resume_mode=0
if [[ "$#" -eq 1 ]]; then
  resume_mode=1
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
mail_file="${repo_dir}/.slurm-notify-email"
root="${repo_dir}/artifacts/screenqa-train-factorized-v1/backbone-7b-diagnostic-v1/full-h800-v1"
worker="${repo_dir}/scripts/slurm_screenqa_backbone_7b_full_h800.sh"
collector="${repo_dir}/src/beyond_entropy/cli.py"
backend="${repo_dir}/src/beyond_entropy/qwen_backend.py"
rollout_module="${repo_dir}/src/beyond_entropy/rollout.py"
crops_module="${repo_dir}/src/beyond_entropy/crops.py"
benchmarks_module="${repo_dir}/src/beyond_entropy/benchmarks.py"
rollout_merger_module="${repo_dir}/src/beyond_entropy/rollout_shards.py"
rollout_merger="${repo_dir}/scripts/merge_qwen_rollout_shards.py"
score_module="${repo_dir}/src/beyond_entropy/answer_likelihood.py"
scorer="${repo_dir}/scripts/score_visual_action_answer_nll.py"
score_merger="${repo_dir}/scripts/merge_visual_action_answer_nll.py"
analyzer="${repo_dir}/scripts/analyze_visual_action_proxy_outcomes.py"
audit_module="${repo_dir}/src/beyond_entropy/proxy_outcome_audit.py"
protocol="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-diagnostic-protocol-v1.md"
population_activation="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-population-activation-v1.md"
analysis_implementation="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-analysis-implementation-v1.md"
hardware_activation="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-full-hardware-activation-v1.md"
smoke_completion="${repo_dir}/artifacts/screenqa-train-factorized-v1/backbone-7b-diagnostic-v1/smoke-h800-v2/job-199116/smoke.complete.json"
model=Qwen/Qwen2.5-VL-7B-Instruct
model_revision=cc594898137f460bfe9f0759e9844b3ce807cfb5

protocol_sha256=1cd70d11168e12a2855ec01e8a869d89b82c4e87c3d864c566ed7db02bb61474
population_activation_sha256=a26b8bc6e8a7c81df3cad59f05ac3c9b35b6c340e71f5596175556cd0af6ee6e
analysis_implementation_sha256=3da107bef0fa8614e6cb088f4e54745cba9d8dcb872b5764c705c12bdf773eb1
hardware_activation_sha256=fa456ae16cba431836900282eec952125eff9982849bede5378c86731f0202b1
smoke_completion_sha256=e944437165523b4dab5261822abbeb002872f068d0ecd70b7af023688ae64e11
collector_sha256=6512131e7a9bbe55b65f9229a044df43e0fa9c4564e4c20fca060a2a17059346
backend_sha256=5ee063fb3d8abe3461186e7185960afd002848f1f31aad7b1fdbc1fc53840acb
rollout_sha256=b4e30265e3b0d9bd69119ffd32901679ccd2b59140d7785c299e14465deff455
crops_sha256=ddbd23e1f3e7930f1ae187aa325f0a26406e8cfa78fbf65a525ea43a22b138bf
benchmarks_sha256=d96d95f3814209822f724f131175058dda7044f6fb70b402aae27a806d1a30fc
rollout_merger_module_sha256=b480e939017774dcd5dab483eeb5864425b046468dbe2356d006408063d347b5
rollout_merger_sha256=5ddd3fcbff9d21f036c75efa8591ab70e3cd9a311e7bd6d679dafcb251061744
score_module_sha256=afcf8ec83e513d855532bf64b7ecc61911a21776b005220d4ec2f8a64e18f470
scorer_sha256=230e1cf2d8e264d9092c0b1c390dbd29029049635911455757d52f3ad9062be4
score_merger_sha256=4e5c8f2a97e9bdfed835f592e6cc9e52138134e4b6d1cfcd855c013b05f5974d
analyzer_sha256=0147a7215ac4956eb908322cce880512e6961ee8ba1cf6ce4321c5084c22e266
audit_module_sha256=7ad2fe4a710e60ca3d1d7f69584c9344c2eee6533e176ddb5e28063b16dae5a4

sha256_of() {
  sha256sum "$1" | cut -d ' ' -f 1
}

require_hash() {
  local path=$1
  local expected=$2
  local label=$3
  if [[ ! -s "${path}" || "$(sha256_of "${path}")" != "${expected}" ]]; then
    echo "Qwen-7B full ${label} hash mismatch" >&2
    exit 2
  fi
}

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm notification email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid Qwen-7B full notification email" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before Qwen-7B full submission" >&2
  exit 2
fi
if [[ -d "${root}" && -n "$(find "${root}" -mindepth 1 -print -quit)" ]]; then
  if [[ "${resume_mode}" != 1 ]]; then
    echo "existing Qwen-7B full outputs require --resume" >&2
    exit 2
  fi
elif [[ "${resume_mode}" == 1 ]]; then
  echo "--resume requires an existing partial Qwen-7B full run" >&2
  exit 2
fi

require_hash "${protocol}" "${protocol_sha256}" protocol
require_hash "${population_activation}" "${population_activation_sha256}" "population activation"
require_hash "${analysis_implementation}" "${analysis_implementation_sha256}" "analysis implementation"
require_hash "${hardware_activation}" "${hardware_activation_sha256}" "hardware activation"
require_hash "${smoke_completion}" "${smoke_completion_sha256}" "smoke completion"
require_hash "${collector}" "${collector_sha256}" collector
require_hash "${backend}" "${backend_sha256}" backend
require_hash "${rollout_module}" "${rollout_sha256}" rollout
require_hash "${crops_module}" "${crops_sha256}" crops
require_hash "${benchmarks_module}" "${benchmarks_sha256}" benchmarks
require_hash "${rollout_merger_module}" "${rollout_merger_module_sha256}" "rollout merger module"
require_hash "${rollout_merger}" "${rollout_merger_sha256}" "rollout merger"
require_hash "${score_module}" "${score_module_sha256}" "answer likelihood"
require_hash "${scorer}" "${scorer_sha256}" scorer
require_hash "${score_merger}" "${score_merger_sha256}" "score merger"
require_hash "${analyzer}" "${analyzer_sha256}" analyzer
require_hash "${audit_module}" "${audit_module_sha256}" "audit module"

if [[ "$(jq -r '.passed' "${smoke_completion}")" != true \
  || "$(jq -r '.accelerator_name' "${smoke_completion}")" != *H800* \
  || "$(jq -r '.outcome_use.task_endpoints_computed' "${smoke_completion}")" != false ]]; then
  echo "successful endpoint-blind H800 smoke is required" >&2
  exit 2
fi

active_jobs=$(/usr/local/slurm/bin/squeue -h -u "${USER}" -t PENDING,RUNNING,CONFIGURING,COMPLETING | wc -l)
if [[ "${active_jobs}" -ne 0 ]]; then
  echo "Qwen-7B full run requires the account's sole job slot to be free" >&2
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
if [[ "${gpu_remaining}" -lt 240 ]]; then
  echo "Qwen-7B full run needs a 240 GPU-minute reserve; remaining=${gpu_remaining}" >&2
  exit 2
fi

hardware_snapshot=$(/usr/local/slurm/bin/sinfo -N -h -o '%N|%P|%t|%G|%C')
if ! grep -qi 'h800' <<< "${hardware_snapshot}"; then
  echo "live Slurm inventory has no H800 node" >&2
  exit 2
fi

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"${python_bin}" - "${model}" "${model_revision}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_files_only=True)
PY

code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
worker_sha256=$(sha256_of "${worker}")
submit_epoch=$(date +%s)
export_args="PATH=/usr/local/bin:/usr/bin:/bin,HOME=/userhome/cs3/yihangc,LANG=C.UTF-8,BE_BB7_FULL_EXPECTED_CODE_REVISION=${code_revision},BE_BB7_FULL_RUN_ROOT=${root},BE_BB7_FULL_WORKER_SHA256=${worker_sha256},BE_BB7_FULL_COLLECTOR_SHA256=${collector_sha256},BE_BB7_FULL_BACKEND_SHA256=${backend_sha256},BE_BB7_FULL_ROLLOUT_SHA256=${rollout_sha256},BE_BB7_FULL_CROPS_SHA256=${crops_sha256},BE_BB7_FULL_BENCHMARKS_SHA256=${benchmarks_sha256},BE_BB7_FULL_ROLLOUT_MERGER_MODULE_SHA256=${rollout_merger_module_sha256},BE_BB7_FULL_ROLLOUT_MERGER_CLI_SHA256=${rollout_merger_sha256},BE_BB7_FULL_SCORE_MODULE_SHA256=${score_module_sha256},BE_BB7_FULL_SCORER_SHA256=${scorer_sha256},BE_BB7_FULL_SCORE_MERGER_SHA256=${score_merger_sha256},BE_BB7_FULL_ANALYZER_SHA256=${analyzer_sha256},BE_BB7_FULL_AUDIT_MODULE_SHA256=${audit_module_sha256},BE_BB7_FULL_PROTOCOL_SHA256=${protocol_sha256},BE_BB7_FULL_POPULATION_ACTIVATION_SHA256=${population_activation_sha256},BE_BB7_FULL_ANALYSIS_IMPLEMENTATION_SHA256=${analysis_implementation_sha256},BE_BB7_FULL_HARDWARE_ACTIVATION_SHA256=${hardware_activation_sha256},BE_BB7_FULL_SMOKE_COMPLETION_SHA256=${smoke_completion_sha256},BE_BB7_FULL_RESUME=${resume_mode},BE_BB7_SUBMIT_EPOCH=${submit_epoch}"

submission=$(
  /usr/local/slurm/bin/sbatch \
    --partition=q-hgpu-small \
    --gres=gpu:h800:4 \
    --cpus-per-task=32 \
    --mem=384G \
    --time=01:00:00 \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${worker}"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse Qwen-7B full job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'screenqa_qwen7b_full_job_id=%s code_revision=%s gpu_type=h800 gpu_count=4 resume=%s remaining_gpu_minutes_before_submit=%s\n' \
  "${job_id}" "${code_revision}" "${resume_mode}" "${gpu_remaining}"
printf '%s\n' "${hardware_snapshot}"
