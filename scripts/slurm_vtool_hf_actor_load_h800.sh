#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=be-vtool-actor-load
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-vtool-actor-load-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "usage: worker EXPECTED_REVISION WORKER_SHA256 SMOKE_SHA256 CONFIG_SHA256 DATASET_SHA256 DATASET_PATH SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_smoke_sha256=$3
expected_config_sha256=$4
expected_dataset_sha256=$5
dataset=$6
submit_epoch=$7
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "actor-load submit epoch must be an integer" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
runtime=/userhome/cs3/yihangc/Documents/runtime/vtool-action-credit-g1
python_bin=/userhome/cs3/yihangc/anaconda3/envs/beyond-entropy-vtool-g1/bin/python
jq_bin=/userhome/cs3/yihangc/anaconda3/bin/jq
worker="${repo}/scripts/slurm_vtool_hf_actor_load_h800.sh"
smoke="${repo}/scripts/smoke_vtool_hf_actor_load.py"
config="${repo}/configs/vtool_action_credit_g1_v1.json"
runtime_patch="${repo}/integrations/vtool_action_credit/vtool-training-v2-d2aa283.patch"
output_dir="${repo}/artifacts/docvqa-train-factorized-v2/actor-load-smoke/vtool-hf-actor-sdpa-v1/job-${SLURM_JOB_ID}"
report="${output_dir}/report.json"
status_dir="${repo}/artifacts/docvqa-train-factorized-v2/g1-execution-status"
status_file="${status_dir}/actor-load-job-${SLURM_JOB_ID}.json"
worker_start_epoch=$(date +%s)

write_worker_status() {
  local exit_code=$?
  trap - EXIT
  mkdir -p "${status_dir}"
  local decision=failed
  local smoke_decision=not_available
  if [[ "${exit_code}" -eq 0 ]]; then
    decision=completed
  fi
  if [[ -s "${report}" ]]; then
    smoke_decision=$("${jq_bin}" -er '.decision' "${report}" 2>/dev/null || printf invalid)
  fi
  "${jq_bin}" -n \
    --arg schema vtool_hf_actor_load_worker_status_v1 \
    --arg decision "${decision}" \
    --arg smoke_decision "${smoke_decision}" \
    --arg job_id "${SLURM_JOB_ID}" \
    --arg code_revision "${expected_revision}" \
    --arg output_dir "${output_dir}" \
    --argjson exit_code "${exit_code}" \
    --argjson started_epoch "${worker_start_epoch}" \
    --argjson ended_epoch "$(date +%s)" \
    '{schema:$schema,decision:$decision,smoke_decision:$smoke_decision,job_id:$job_id,code_revision:$code_revision,output_dir:$output_dir,exit_code:$exit_code,started_epoch:$started_epoch,ended_epoch:$ended_epoch}' \
    > "${status_file}.tmp"
  mv "${status_file}.tmp" "${status_file}"
  exit "${exit_code}"
}
trap write_worker_status EXIT

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "actor-load ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "actor-load tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "actor-load repository worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${smoke}" "${expected_smoke_sha256}" smoke-script
require_hash "${config}" "${expected_config_sha256}" config
require_hash "${dataset}" "${expected_dataset_sha256}" dataset
if [[ ! -x "${python_bin}" || ! -x "${jq_bin}" || ! -d "${runtime}/verl" ]]; then
  echo "actor-load frozen runtime is absent" >&2
  exit 2
fi

model=$("${jq_bin}" -er '.model.local_snapshot' "${config}")
expected_model_revision=$("${jq_bin}" -er '.model.revision' "${config}")
expected_runtime_commit=$("${jq_bin}" -er '.runtime.upstream_commit' "${config}")
expected_runtime_patch_sha256=$("${jq_bin}" -er '.runtime.patch_sha256' "${config}")
mapfile -t expected_weight_sha256 < <(
  "${jq_bin}" -er '.model.weight_blob_sha256[]' "${config}"
)
if [[ ! -d "${model}" || "$(basename "${model}")" != "${expected_model_revision}" ]]; then
  echo "actor-load model snapshot does not match frozen config" >&2
  exit 2
fi
if [[ "$(git -C "${runtime}" rev-parse HEAD)" != "${expected_runtime_commit}" ]]; then
  echo "actor-load runtime commit does not match frozen config" >&2
  exit 2
fi
runtime_status=$(git -C "${runtime}" status --short)
expected_runtime_status=$' M verl/experimental/agent_loop/agent_loop.py\n M verl/trainer/ppo/ray_trainer.py'
if [[ "${runtime_status}" != "${expected_runtime_status}" ]]; then
  echo "actor-load runtime has unexpected modifications" >&2
  exit 2
fi
require_hash "${runtime_patch}" "${expected_runtime_patch_sha256}" runtime-patch
runtime_patch_sha256=$(
  git -C "${runtime}" diff --unified=0 -- \
    verl/experimental/agent_loop/agent_loop.py \
    verl/trainer/ppo/ray_trainer.py | sha256sum | awk '{print $1}'
)
if [[ "${runtime_patch_sha256}" != "${expected_runtime_patch_sha256}" ]]; then
  echo "actor-load runtime modifications do not match frozen patch" >&2
  exit 2
fi
if [[ "${#expected_weight_sha256[@]}" -ne 2 ]]; then
  echo "actor-load expects two frozen model weight shards" >&2
  exit 2
fi
require_hash "${model}/model-00001-of-00002.safetensors" "${expected_weight_sha256[0]}" model-weight-1
require_hash "${model}/model-00002-of-00002.safetensors" "${expected_weight_sha256[1]}" model-weight-2
if ! "${jq_bin}" -e '
  .training.actor_attention_implementation == "sdpa" and
  .training.actor_use_remove_padding == false and
  .training.actor_dtype == "bfloat16" and
  .data.protected_split_contents_accessed == false
' "${config}" >/dev/null; then
  echo "actor-load frozen attention contract failed" >&2
  exit 2
fi
if [[ -e "${output_dir}" || -e "${status_file}" ]]; then
  echo "actor-load refuses to overwrite an existing artifact" >&2
  exit 2
fi
available_kb=$(df -Pk "${repo}" | awk 'NR==2 {print $4}')
if [[ -z "${available_kb}" || "${available_kb}" -lt 8388608 ]]; then
  echo "actor-load smoke requires at least 8 GiB free disk" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Slurm exposed no GPU to actor-load smoke" >&2
  exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_gpus[@]}" -ne 1 ]]; then
  echo "actor-load smoke requires exactly one visible H800" >&2
  exit 2
fi

export PYTHONPATH="${runtime}:${repo}:${repo}/src"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN OPENAI_API_KEY OPENAI_API_BASE \
  OPENAI_BASE_URL VTOOL_JUDGE_API_BASE HTTP_PROXY HTTPS_PROXY ALL_PROXY \
  http_proxy https_proxy all_proxy

queue_wait_seconds=$((worker_start_epoch - submit_epoch))
if [[ "${queue_wait_seconds}" -lt 0 ]]; then
  echo "actor-load submit epoch is in the future" >&2
  exit 2
fi
echo "vtool HF actor-load smoke start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "tracked revision: ${expected_revision}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"${python_bin}" "${smoke}" \
  --config "${config}" \
  --dataset "${dataset}" \
  --output "${report}" \
  --mode gpu-forward

if ! "${jq_bin}" -e \
  --arg model_revision "${expected_model_revision}" \
  --arg dataset_sha256 "${expected_dataset_sha256}" '
  .decision == "vtool_hf_actor_gpu_forward_smoke_passed" and
  .model_revision == $model_revision and
  .dataset_sha256 == $dataset_sha256 and
  .attention_implementation == "sdpa" and
  .use_remove_padding == false and
  .model_weights_loaded == true and
  .optimizer_step_performed == false and
  .protected_split_contents_accessed == false and
  .prompt_tokens > 0 and
  (.checks | all(.[]; . == true))
' "${report}" >/dev/null; then
  echo "actor-load report contract failed" >&2
  exit 2
fi
echo "vtool HF actor-load smoke end: $(date --iso-8601=seconds)"
echo "report=${report} queue_wait_seconds=${queue_wait_seconds}"
