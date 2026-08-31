#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
mail_file="${repo_dir}/.slurm-notify-email"
root="${repo_dir}/artifacts/screenqa-train-factorized-v1/backbone-7b-diagnostic-v1/full-h800-v1"
worker="${repo_dir}/scripts/slurm_screenqa_backbone_7b_runtime_recovery.sh"
recovery_module="${repo_dir}/src/beyond_entropy/rollout_runtime_recovery.py"
recovery_cli="${repo_dir}/scripts/recover_rollout_runtime_provenance.py"
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
analysis_implementation="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-analysis-implementation-v1.md"
model=Qwen/Qwen2.5-VL-7B-Instruct
model_revision=cc594898137f460bfe9f0759e9844b3ce807cfb5
frozen_collector_sha256=6512131e7a9bbe55b65f9229a044df43e0fa9c4564e4c20fca060a2a17059346
frozen_backend_sha256=5ee063fb3d8abe3461186e7185960afd002848f1f31aad7b1fdbc1fc53840acb
frozen_rollout_sha256=b4e30265e3b0d9bd69119ffd32901679ccd2b59140d7785c299e14465deff455
frozen_crops_sha256=ddbd23e1f3e7930f1ae187aa325f0a26406e8cfa78fbf65a525ea43a22b138bf
frozen_benchmarks_sha256=d96d95f3814209822f724f131175058dda7044f6fb70b402aae27a806d1a30fc
frozen_rollout_merger_module_sha256=b480e939017774dcd5dab483eeb5864425b046468dbe2356d006408063d347b5
frozen_rollout_merger_sha256=5ddd3fcbff9d21f036c75efa8591ab70e3cd9a311e7bd6d679dafcb251061744
frozen_score_module_sha256=afcf8ec83e513d855532bf64b7ecc61911a21776b005220d4ec2f8a64e18f470
frozen_scorer_sha256=230e1cf2d8e264d9092c0b1c390dbd29029049635911455757d52f3ad9062be4
frozen_score_merger_sha256=4e5c8f2a97e9bdfed835f592e6cc9e52138134e4b6d1cfcd855c013b05f5974d
frozen_analyzer_sha256=0147a7215ac4956eb908322cce880512e6961ee8ba1cf6ce4321c5084c22e266
frozen_audit_module_sha256=7ad2fe4a710e60ca3d1d7f69584c9344c2eee6533e176ddb5e28063b16dae5a4
frozen_protocol_sha256=1cd70d11168e12a2855ec01e8a869d89b82c4e87c3d864c566ed7db02bb61474
frozen_analysis_implementation_sha256=3da107bef0fa8614e6cb088f4e54745cba9d8dcb872b5764c705c12bdf773eb1

sha256_of() {
  sha256sum "$1" | cut -d ' ' -f 1
}

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm notification email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid runtime recovery notification email" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before runtime recovery submission" >&2
  exit 2
fi
if [[ -e "${root}/analysis/report.json" || -e "${root}/merged-scores/answer-nll.jsonl" ]]; then
  echo "Qwen-7B runtime recovery is no longer applicable" >&2
  exit 2
fi
for shard_index in 0 1 2 3; do
  shard_name=$(printf 'shard-%05d-of-00004' "${shard_index}")
  rollout_provenance="${root}/rollout-shards/${shard_name}/rollouts.provenance.json"
  score_provenance="${root}/score-shards/answer-nll-shard-${shard_index}-of-4.provenance.json"
  if [[ "$(jq -r '.runtime_measurement' "${rollout_provenance}")" != null \
    || "$(jq -r '.records' "${score_provenance}")" -ne 640 \
    || "$(jq -r '.runtime_measurement.accelerator_name' "${score_provenance}")" != "NVIDIA H800" ]]; then
    echo "Qwen-7B incomplete-run shape does not match the runtime recovery case" >&2
    exit 2
  fi
done

active_jobs=$(/usr/local/slurm/bin/squeue -h -u "${USER}" -t PENDING,RUNNING,CONFIGURING,COMPLETING | wc -l)
if [[ "${active_jobs}" -ne 0 ]]; then
  echo "runtime recovery requires the account's sole job slot to be free" >&2
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
if [[ "${gpu_remaining}" -lt 80 ]]; then
  echo "runtime recovery needs an 80 GPU-minute reserve; remaining=${gpu_remaining}" >&2
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
recovery_module_sha256=$(sha256_of "${recovery_module}")
recovery_cli_sha256=$(sha256_of "${recovery_cli}")
collector_sha256=$(sha256_of "${collector}")
backend_sha256=$(sha256_of "${backend}")
rollout_sha256=$(sha256_of "${rollout_module}")
crops_sha256=$(sha256_of "${crops_module}")
benchmarks_sha256=$(sha256_of "${benchmarks_module}")
rollout_merger_module_sha256=$(sha256_of "${rollout_merger_module}")
rollout_merger_sha256=$(sha256_of "${rollout_merger}")
score_module_sha256=$(sha256_of "${score_module}")
scorer_sha256=$(sha256_of "${scorer}")
score_merger_sha256=$(sha256_of "${score_merger}")
analyzer_sha256=$(sha256_of "${analyzer}")
audit_module_sha256=$(sha256_of "${audit_module}")
protocol_sha256=$(sha256_of "${protocol}")
analysis_implementation_sha256=$(sha256_of "${analysis_implementation}")
for pair in \
  "${collector_sha256}:${frozen_collector_sha256}" \
  "${backend_sha256}:${frozen_backend_sha256}" \
  "${rollout_sha256}:${frozen_rollout_sha256}" \
  "${crops_sha256}:${frozen_crops_sha256}" \
  "${benchmarks_sha256}:${frozen_benchmarks_sha256}" \
  "${rollout_merger_module_sha256}:${frozen_rollout_merger_module_sha256}" \
  "${rollout_merger_sha256}:${frozen_rollout_merger_sha256}" \
  "${score_module_sha256}:${frozen_score_module_sha256}" \
  "${scorer_sha256}:${frozen_scorer_sha256}" \
  "${score_merger_sha256}:${frozen_score_merger_sha256}" \
  "${analyzer_sha256}:${frozen_analyzer_sha256}" \
  "${audit_module_sha256}:${frozen_audit_module_sha256}" \
  "${protocol_sha256}:${frozen_protocol_sha256}" \
  "${analysis_implementation_sha256}:${frozen_analysis_implementation_sha256}"; do
  if [[ "${pair%%:*}" != "${pair#*:}" ]]; then
    echo "runtime recovery scientific component differs from the frozen implementation" >&2
    exit 2
  fi
done
submit_epoch=$(date +%s)
export_args="PATH=/usr/local/bin:/usr/bin:/bin,HOME=/userhome/cs3/yihangc,LANG=C.UTF-8,BE_BB7_RECOVERY_EXPECTED_CODE_REVISION=${code_revision},BE_BB7_RECOVERY_RUN_ROOT=${root},BE_BB7_RECOVERY_WORKER_SHA256=${worker_sha256},BE_BB7_RECOVERY_MODULE_SHA256=${recovery_module_sha256},BE_BB7_RECOVERY_CLI_SHA256=${recovery_cli_sha256},BE_BB7_RECOVERY_COLLECTOR_SHA256=${collector_sha256},BE_BB7_RECOVERY_BACKEND_SHA256=${backend_sha256},BE_BB7_RECOVERY_ROLLOUT_SHA256=${rollout_sha256},BE_BB7_RECOVERY_CROPS_SHA256=${crops_sha256},BE_BB7_RECOVERY_BENCHMARKS_SHA256=${benchmarks_sha256},BE_BB7_RECOVERY_ROLLOUT_MERGER_MODULE_SHA256=${rollout_merger_module_sha256},BE_BB7_RECOVERY_ROLLOUT_MERGER_CLI_SHA256=${rollout_merger_sha256},BE_BB7_RECOVERY_SCORE_MODULE_SHA256=${score_module_sha256},BE_BB7_RECOVERY_SCORER_SHA256=${scorer_sha256},BE_BB7_RECOVERY_SCORE_MERGER_SHA256=${score_merger_sha256},BE_BB7_RECOVERY_ANALYZER_SHA256=${analyzer_sha256},BE_BB7_RECOVERY_AUDIT_MODULE_SHA256=${audit_module_sha256},BE_BB7_RECOVERY_PROTOCOL_SHA256=${protocol_sha256},BE_BB7_RECOVERY_ANALYSIS_IMPLEMENTATION_SHA256=${analysis_implementation_sha256},BE_BB7_RECOVERY_SUBMIT_EPOCH=${submit_epoch}"
submission=$(
  /usr/local/slurm/bin/sbatch \
    --partition=q-h800 \
    --gres=gpu:h800:4 \
    --cpus-per-task=32 \
    --mem=384G \
    --time=00:20:00 \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${worker}"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse runtime recovery job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'screenqa_qwen7b_runtime_recovery_job_id=%s code_revision=%s gpu_count=4 remaining_gpu_minutes_before_submit=%s\n' \
  "${job_id}" "${code_revision}" "${gpu_remaining}"
