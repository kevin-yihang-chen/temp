#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=be-typed-b0
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-typed-b0-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "usage: worker REVISION WORKER_SHA RUNNER_SHA CONFIG_SHA DATASET_SHA CONVERTER_REPORT_SHA PROCESSOR_REPORT_SHA SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_runner_sha256=$3
expected_config_sha256=$4
expected_dataset_sha256=$5
expected_converter_report_sha256=$6
expected_processor_report_sha256=$7
submit_epoch=$8
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "B0 generation submit epoch must be an integer" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
runtime=/userhome/cs3/yihangc/Documents/runtime/vtool-action-credit-g1
python_bin=/userhome/cs3/yihangc/anaconda3/envs/beyond-entropy-vtool-g1/bin/python
jq_bin=/userhome/cs3/yihangc/anaconda3/bin/jq
worker="${repo}/scripts/slurm_refocus_typed_action_b0_generation_h800.sh"
runner="${repo}/scripts/run_refocus_typed_action_b0_generation.py"
config="${repo}/configs/refocus_typed_action_b0_generation_v1.json"
runtime_patch="${repo}/integrations/vtool_action_credit/vtool-training-v2-d2aa283.patch"
dataset_rel=$("${jq_bin}" -er '.data.dataset' "${config}")
converter_report_rel=$("${jq_bin}" -er '.data.converter_report' "${config}")
processor_report_rel=$("${jq_bin}" -er '.data.processor_executor_report' "${config}")
dataset="${repo}/${dataset_rel}"
converter_report="${repo}/${converter_report_rel}"
processor_report="${repo}/${processor_report_rel}"
output_dir="${repo}/artifacts/docvqa-train-factorized-v2/model-generation-smoke/refocus-typed-action-b0-v1/job-${SLURM_JOB_ID}"
report="${output_dir}/report.json"
status_dir="${repo}/artifacts/docvqa-train-factorized-v2/b0-execution-status"
status_file="${status_dir}/typed-action-b0-job-${SLURM_JOB_ID}.json"
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
  if [[ -s "${report}" ]]; then
    scientific_decision=$("${jq_bin}" -er '.scientific_decision' "${report}" 2>/dev/null || printf invalid)
  fi
  "${jq_bin}" -n \
    --arg schema refocus_typed_action_b0_worker_status_v1 \
    --arg decision "${decision}" \
    --arg scientific_decision "${scientific_decision}" \
    --arg job_id "${SLURM_JOB_ID}" \
    --arg code_revision "${expected_revision}" \
    --arg output_dir "${output_dir}" \
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
    echo "B0 generation ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "B0 generation tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "B0 generation repository worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${runner}" "${expected_runner_sha256}" runner
require_hash "${config}" "${expected_config_sha256}" config
require_hash "${dataset}" "${expected_dataset_sha256}" dataset
require_hash "${converter_report}" "${expected_converter_report_sha256}" converter-report
require_hash "${processor_report}" "${expected_processor_report_sha256}" processor-report
if [[ ! -x "${python_bin}" || ! -x "${jq_bin}" || ! -d "${runtime}/verl" ]]; then
  echo "B0 frozen runtime is absent" >&2
  exit 2
fi

model=$("${jq_bin}" -er '.model.local_snapshot' "${config}")
expected_model_revision=$("${jq_bin}" -er '.model.revision' "${config}")
expected_runtime_commit=$("${jq_bin}" -er '.runtime.upstream_commit' "${config}")
expected_runtime_patch_sha256=$("${jq_bin}" -er '.runtime.patch_sha256' "${config}")
expected_refocus_tools_sha256=$("${jq_bin}" -er '.runtime.refocus_tools_sha256' "${config}")
mapfile -t expected_weight_sha256 < <("${jq_bin}" -er '.model.weight_blob_sha256[]' "${config}")
if [[ ! -d "${model}" || "$(basename "${model}")" != "${expected_model_revision}" ]]; then
  echo "B0 model snapshot does not match the protocol" >&2
  exit 2
fi
if [[ "$(git -C "${runtime}" rev-parse HEAD)" != "${expected_runtime_commit}" ]]; then
  echo "B0 runtime commit does not match the protocol" >&2
  exit 2
fi
runtime_status=$(git -C "${runtime}" status --short)
expected_runtime_status=$' M verl/experimental/agent_loop/agent_loop.py\n M verl/trainer/ppo/ray_trainer.py'
if [[ "${runtime_status}" != "${expected_runtime_status}" ]]; then
  echo "B0 runtime has unexpected modifications" >&2
  exit 2
fi
require_hash "${runtime_patch}" "${expected_runtime_patch_sha256}" runtime-patch
runtime_patch_sha256=$(
  git -C "${runtime}" diff --unified=0 -- \
    verl/experimental/agent_loop/agent_loop.py \
    verl/trainer/ppo/ray_trainer.py | sha256sum | awk '{print $1}'
)
if [[ "${runtime_patch_sha256}" != "${expected_runtime_patch_sha256}" ]]; then
  echo "B0 runtime modifications do not match the frozen patch" >&2
  exit 2
fi
require_hash "${runtime}/recipe/vtool/refocus_tools.py" "${expected_refocus_tools_sha256}" refocus-tools
if [[ "${#expected_weight_sha256[@]}" -ne 2 ]]; then
  echo "B0 protocol must freeze exactly two model weight shards" >&2
  exit 2
fi
require_hash "${model}/model-00001-of-00002.safetensors" "${expected_weight_sha256[0]}" model-weight-1
require_hash "${model}/model-00002-of-00002.safetensors" "${expected_weight_sha256[1]}" model-weight-2

if ! "${jq_bin}" -e \
  --arg dataset_sha256 "${expected_dataset_sha256}" \
  --arg converter_sha256 "${expected_converter_report_sha256}" \
  --arg processor_sha256 "${expected_processor_report_sha256}" '
  .schema == "refocus_typed_action_b0_generation_protocol_v1" and
  .study_role == "baseline_correctness_only" and
  .uses_reward_target == false and
  .data.dataset_sha256 == $dataset_sha256 and
  .data.converter_report_sha256 == $converter_sha256 and
  .data.processor_executor_report_sha256 == $processor_sha256 and
  .data.protected_split_contents_accessed == false and
  .sampling.generation_count == 16 and
  (.sampling.seeds | length) == 16 and
  .sampling.n == 1 and
  .sampling.max_tokens == 128 and
  .resources.gpu_count == 1 and
  .resources.gpu_type == "H800" and
  .resources.optimizer_steps == 0 and
  .resources.checkpoints_written == 0 and
  .resources.notification_email == "yihangc@connect.hku.hk" and
  .resources.slurm_mail_type == "ALL" and
  .analysis.raw_model_text_execution_allowed == false
' "${config}" >/dev/null; then
  echo "B0 protocol values changed" >&2
  exit 2
fi
if ! "${jq_bin}" -e '
  .decision == "refocus_typed_action_b0_real_runtime_smoke_passed" and
  (.checks | all(.[]; . == true)) and
  .checkpoints_written == 0 and
  .protected_split_contents_accessed == false
' "${processor_report}" >/dev/null; then
  echo "B0 prerequisite report contract failed" >&2
  exit 2
fi
if [[ -e "${output_dir}" || -e "${status_file}" ]]; then
  echo "B0 generation refuses to overwrite an existing artifact" >&2
  exit 2
fi
minimum_disk_gib=$("${jq_bin}" -er '.resources.minimum_free_persistent_disk_gib' "${config}")
available_kb=$(df -Pk "${repo}" | awk 'NR==2 {print $4}')
if [[ -z "${available_kb}" || "${available_kb}" -lt $((minimum_disk_gib * 1024 * 1024)) ]]; then
  echo "B0 generation lacks the frozen persistent-disk reserve" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Slurm exposed no GPU to B0 generation" >&2
  exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_gpus[@]}" -ne 1 ]]; then
  echo "B0 generation requires exactly one visible H800" >&2
  exit 2
fi
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "${gpu_name}" != *H800* ]]; then
  echo "B0 generation received a non-H800 GPU: ${gpu_name}" >&2
  exit 2
fi

export PYTHONPATH="${runtime}:${repo}:${repo}/src"
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
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN OPENAI_API_KEY \
  OPENAI_API_BASE OPENAI_BASE_URL VTOOL_JUDGE_API_BASE HTTP_PROXY HTTPS_PROXY \
  ALL_PROXY http_proxy https_proxy all_proxy

mkdir -p "${output_dir}"
queue_wait_seconds=$((worker_start_epoch - submit_epoch))
if [[ "${queue_wait_seconds}" -lt 0 ]]; then
  echo "B0 submit epoch is in the future" >&2
  exit 2
fi
echo "typed-action B0 generation start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "tracked revision: ${expected_revision}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"${python_bin}" "${runner}" \
  --config "${config}" \
  --code-revision "${expected_revision}" \
  --output "${report}"

if ! "${jq_bin}" -e \
  --arg revision "${expected_revision}" \
  --arg model_revision "${expected_model_revision}" \
  --arg dataset_sha256 "${expected_dataset_sha256}" '
  .decision == "refocus_typed_action_b0_generation_smoke_completed" and
  (.scientific_decision == "typed_action_b0_format_gate_passed" or
   .scientific_decision == "typed_action_b0_insufficient_tool_intent_support" or
   .scientific_decision == "typed_action_b0_malformed_tool_intent") and
  (.format_gate_qualified == (.scientific_decision == "typed_action_b0_format_gate_passed")) and
  .code_revision == $revision and
  .model_revision == $model_revision and
  .dataset_sha256 == $dataset_sha256 and
  .generation_count == 16 and
  (.completions | length) == 16 and
  (.checks | all(.[]; . == true)) and
  ([.completions[].raw_model_text_executed] | all(.[]; . == false)) and
  .reward_target_used == false and
  .raw_model_text_executed == false and
  .model_weights_loaded == true and
  .optimizer_steps == 0 and
  .checkpoints_written == 0 and
  .protected_split_contents_accessed == false
' "${report}" >/dev/null; then
  echo "B0 generation report contract failed" >&2
  exit 2
fi
mapfile -t output_files < <(find "${output_dir}" -mindepth 1 -maxdepth 1 -type f -printf '%f\n')
mapfile -t output_dirs < <(find "${output_dir}" -mindepth 1 -type d -printf '%p\n')
if [[ "${#output_files[@]}" -ne 1 || "${output_files[0]}" != "report.json" || "${#output_dirs[@]}" -ne 0 ]]; then
  echo "B0 generation wrote unexpected files or checkpoint directories" >&2
  exit 2
fi
echo "typed-action B0 generation end: $(date --iso-8601=seconds)"
echo "report=${report} report_sha256=$(sha "${report}") queue_wait_seconds=${queue_wait_seconds} checkpoints=0"
