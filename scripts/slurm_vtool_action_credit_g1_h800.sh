#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --job-name=be-vtool-g1-signed
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-vtool-g1-signed-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "usage: worker EXPECTED_REVISION WORKER_SHA256 LAUNCHER_SHA256 CONFIG_SHA256 AUDIT_SHA256 AUDIT_PATH SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_launcher_sha256=$3
expected_config_sha256=$4
expected_audit_sha256=$5
runtime_audit=$6
submit_epoch=$7
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "G1 submit epoch must be an integer" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
runtime=/userhome/cs3/yihangc/Documents/runtime/vtool-action-credit-g1
python_bin=/userhome/cs3/yihangc/anaconda3/envs/beyond-entropy-vtool-g1/bin/python
worker="${repo}/scripts/slurm_vtool_action_credit_g1_h800.sh"
launcher="${repo}/scripts/run_vtool_action_credit_g1.py"
config="${repo}/configs/vtool_action_credit_g1_v1.json"
audit_script="${repo}/scripts/audit_refocus_g1_runtime_dataset.py"
analyzer="${repo}/scripts/analyze_vtool_action_credit_g1.py"
jq_bin=/userhome/cs3/yihangc/anaconda3/bin/jq
output_dir="${repo}/artifacts/docvqa-train-factorized-v2/g1-runs/paired-signed-v1/job-${SLURM_JOB_ID}"
analysis_report="${output_dir}/rollout-analysis.json"
status_dir="${repo}/artifacts/docvqa-train-factorized-v2/g1-execution-status"
status_file="${status_dir}/paired-signed-job-${SLURM_JOB_ID}.json"
worker_start_epoch=$(date +%s)

write_worker_status() {
  local exit_code=$?
  trap - EXIT
  mkdir -p "${status_dir}"
  local decision=failed
  local scientific_decision=not_available
  if [[ "${exit_code}" -eq 0 ]]; then
    decision=completed
  fi
  if [[ -s "${analysis_report}" ]]; then
    scientific_decision=$("${jq_bin}" -er '.decision' "${analysis_report}" 2>/dev/null || printf invalid)
  fi
  "${jq_bin}" -n \
    --arg schema vtool_action_credit_g1_worker_status_v1 \
    --arg decision "${decision}" \
    --arg job_id "${SLURM_JOB_ID}" \
    --arg code_revision "${expected_revision}" \
    --arg output_dir "${output_dir}" \
    --arg scientific_decision "${scientific_decision}" \
    --argjson exit_code "${exit_code}" \
    --argjson started_epoch "${worker_start_epoch}" \
    --argjson ended_epoch "$(date +%s)" \
    '{schema:$schema,decision:$decision,scientific_decision:$scientific_decision,job_id:$job_id,code_revision:$code_revision,output_dir:$output_dir,exit_code:$exit_code,started_epoch:$started_epoch,ended_epoch:$ended_epoch}' \
    > "${status_file}.tmp"
  mv "${status_file}.tmp" "${status_file}"
  exit "${exit_code}"
}
trap write_worker_status EXIT

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "G1 ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "G1 tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "G1 repository worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${launcher}" "${expected_launcher_sha256}" launcher
require_hash "${config}" "${expected_config_sha256}" config
require_hash "${runtime_audit}" "${expected_audit_sha256}" runtime-audit
if [[ ! -x "${python_bin}" ]]; then
  echo "G1 frozen Python environment is absent" >&2
  exit 2
fi
if [[ ! -x "${analyzer}" ]]; then
  echo "G1 rollout analyzer is absent or not executable" >&2
  exit 2
fi
if [[ ! -x "${jq_bin}" ]]; then
  echo "G1 frozen jq executable is absent" >&2
  exit 2
fi
if [[ -e "${output_dir}" || -e "${status_file}" ]]; then
  echo "G1 refuses to overwrite an existing job artifact" >&2
  exit 2
fi

train_sha256=$("${jq_bin}" -er '.data.train.paired_sha256' "${config}")
train_row_manifest_sha256=$("${jq_bin}" -er '.data.train.row_id_manifest_sha256' "${config}")
model_revision=$("${jq_bin}" -er '.model.revision' "${config}")
audit_script_sha256=$(sha "${audit_script}")
if ! "${jq_bin}" -e \
  --arg train_sha256 "${train_sha256}" '
    .decision == "refocus_g1_runtime_dataset_audit_passed" and
    .dataset_sha256 == $train_sha256 and
    .dataset_rows == 72 and
    .structural_groups == 64 and
    .prompt_tokens.max <= .prompt_tokens.frozen_limit and
    .protected_split_contents_accessed == false and
    .model_weights_loaded == false and
    (.checks | all(.[] == true))
  ' "${runtime_audit}" >/dev/null; then
  echo "G1 runtime dataset audit contract failed" >&2
  exit 2
fi
if ! "${jq_bin}" -e \
  --arg row_manifest "${train_row_manifest_sha256}" \
  --arg model_revision "${model_revision}" \
  --arg audit_script_sha256 "${audit_script_sha256}" '
    .row_id_manifest_sha256 == $row_manifest and
    .model_revision == $model_revision and
    .audit_script_sha256 == $audit_script_sha256
  ' "${runtime_audit}" >/dev/null; then
  echo "G1 runtime audit provenance contract failed" >&2
  exit 2
fi
if ! "${jq_bin}" -e '
  .frozen_before_g1_results == true and
  .resources.gpu_count == 4 and
  .resources.gpu_type == "H800" and
  .resources.max_walltime_minutes == 120 and
  .resources.slurm_mail_type == "ALL" and
  .training.total_optimizer_steps == 2 and
  .training.save_frequency_steps == 2 and
  .training.max_actor_checkpoints_to_keep == 1 and
  .training.actor_ppo_mini_batch_size == 8 and
  .training.rollout_n == 4 and
  .arms.paired_signed_credit.action_credit.enabled == true and
  .arms.paired_signed_credit.action_credit.mode == "signed" and
  .data.protected_split_contents_accessed == false
' "${config}" >/dev/null; then
  echo "G1 frozen signed-arm config contract failed" >&2
  exit 2
fi

available_kb=$(df -Pk "${repo}" | awk 'NR==2 {print $4}')
if [[ -z "${available_kb}" || "${available_kb}" -lt 33554432 ]]; then
  echo "G1 requires at least 32 GiB free persistent disk for one resumable checkpoint" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Slurm exposed no GPUs to G1" >&2
  exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_gpus[@]}" -ne 4 ]]; then
  echo "G1 requires exactly four visible H800 GPUs" >&2
  exit 2
fi
gpu_report=$("${python_bin}" -c 'import json, torch; print(json.dumps({"count": torch.cuda.device_count(), "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))')
if ! "${jq_bin}" -e '.count == 4 and (.names | all(contains("H800")))' <<< "${gpu_report}" >/dev/null; then
  echo "G1 PyTorch runtime did not expose exactly four H800 GPUs: ${gpu_report}" >&2
  exit 2
fi

ray_tmp=/dev/shm/beyond-entropy-vtool-g1-${SLURM_JOB_ID}
mkdir -p "${ray_tmp}"
export RAY_TMPDIR="${ray_tmp}"
export TMPDIR="${ray_tmp}"
export PYTHONPATH="${runtime}:${repo}:${repo}/src"
export WANDB_MODE=disabled
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN OPENAI_API_KEY OPENAI_API_BASE \
  OPENAI_BASE_URL VTOOL_JUDGE_API_BASE HTTP_PROXY HTTPS_PROXY ALL_PROXY \
  http_proxy https_proxy all_proxy

queue_wait_seconds=$((worker_start_epoch - submit_epoch))
if [[ "${queue_wait_seconds}" -lt 0 ]]; then
  echo "G1 submit epoch is in the future" >&2
  exit 2
fi
echo "paired-signed G1 start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "tracked revision: ${expected_revision}"
echo "queue wait seconds: ${queue_wait_seconds}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"${python_bin}" "${launcher}" \
  --arm paired_signed_credit \
  --output-dir "${output_dir}" \
  --config "${config}" \
  --mode execute

if ! "${jq_bin}" -e \
  --arg job_id "${SLURM_JOB_ID}" '
    .arm == "paired_signed_credit" and
    .exit_code == 0 and
    .slurm_job_id == $job_id
  ' "${output_dir}/execution.json" >/dev/null; then
  echo "G1 execution report contract failed" >&2
  exit 2
fi
for step in 1 2; do
  if [[ ! -s "${output_dir}/rollouts/${step}.jsonl" ]]; then
    echo "G1 rollout dump for step ${step} is absent" >&2
    exit 2
  fi
done
if [[ "$(cat "${output_dir}/checkpoints/latest_checkpointed_iteration.txt")" != 2 ]]; then
  echo "G1 final resumable checkpoint tracker is absent or incorrect" >&2
  exit 2
fi
if [[ ! -d "${output_dir}/checkpoints/global_step_2/actor" ]]; then
  echo "G1 final actor checkpoint is absent" >&2
  exit 2
fi

"${python_bin}" "${analyzer}" \
  --rollout-dir "${output_dir}/rollouts" \
  --config "${config}" \
  --expected-arm paired_signed_credit \
  --output "${analysis_report}"
if ! "${jq_bin}" -e '
  (.decision == "paired_signed_g1_smoke_gate_passed" or
   .decision == "paired_signed_g1_stop_rule_triggered") and
  .rows == 64 and
  .pair_mismatch_count == 0 and
  .judge_failure_count == 0 and
  (.checks | all(.[] == true))
' "${analysis_report}" >/dev/null; then
  echo "G1 rollout analysis contract failed" >&2
  exit 2
fi

echo "paired-signed G1 end: $(date --iso-8601=seconds)"
du -sh "${output_dir}"
echo "scientific_decision=$("${jq_bin}" -r '.decision' "${analysis_report}")"
echo "output=${output_dir} queue_wait_seconds=${queue_wait_seconds}"
