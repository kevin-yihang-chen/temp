#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=be-vtool-load
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-vtool-model-load-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "usage: worker EXPECTED_REVISION WORKER_SHA256 SMOKE_SHA256 DATASET_SHA256 CONFIG_SHA256 DATASET_PATH SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_smoke_sha256=$3
expected_dataset_sha256=$4
expected_config_sha256=$5
dataset=$6
submit_epoch=$7
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "model-load submit epoch must be an integer" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
runtime=/userhome/cs3/yihangc/Documents/runtime/vtool-action-credit-g1
python_bin=/userhome/cs3/yihangc/anaconda3/envs/beyond-entropy-vtool-g1/bin/python
model=/userhome/cs3/yihangc/Data/hf_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
worker="${repo}/scripts/slurm_vtool_vllm_model_load_h800.sh"
smoke="${repo}/scripts/smoke_vtool_vllm_model_load.py"
config="${repo}/configs/vtool_action_credit_g1_v1.json"
runtime_patch="${repo}/integrations/vtool_action_credit/vtool-training-v2-d2aa283.patch"
output_dir="${repo}/artifacts/docvqa-train-factorized-v2/model-load-smoke/vtool-vllm-h800-v1/job-${SLURM_JOB_ID}"
report="${output_dir}/report.json"

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "model-load ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "model-load tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "model-load tracked worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${smoke}" "${expected_smoke_sha256}" smoke-script
require_hash "${dataset}" "${expected_dataset_sha256}" dataset
require_hash "${config}" "${expected_config_sha256}" config
if [[ ! -x "${python_bin}" || ! -d "${runtime}/verl" || ! -d "${model}" ]]; then
  echo "model-load runtime, Python, or model snapshot is absent" >&2
  exit 2
fi

expected_model_revision=$(jq -er '.model.revision' "${config}")
expected_runtime_commit=$(jq -er '.runtime.upstream_commit' "${config}")
expected_runtime_patch_sha256=$(jq -er '.runtime.patch_sha256' "${config}")
mapfile -t expected_weight_sha256 < <(
  jq -er '.model.weight_blob_sha256[]' "${config}"
)
if [[ "$(basename "${model}")" != "${expected_model_revision}" ]]; then
  echo "model-load snapshot revision does not match frozen config" >&2
  exit 2
fi
if [[ "$(git -C "${runtime}" rev-parse HEAD)" != "${expected_runtime_commit}" ]]; then
  echo "model-load runtime commit does not match frozen config" >&2
  exit 2
fi
if [[ "${#expected_weight_sha256[@]}" -ne 2 ]]; then
  echo "model-load config must freeze exactly two model weight shards" >&2
  exit 2
fi
require_hash "${runtime_patch}" "${expected_runtime_patch_sha256}" runtime-patch
runtime_patch_sha256=$(
  git -C "${runtime}" diff --unified=0 -- \
    verl/experimental/agent_loop/agent_loop.py \
    verl/trainer/ppo/ray_trainer.py | sha256sum | awk '{print $1}'
)
if [[ "${runtime_patch_sha256}" != "${expected_runtime_patch_sha256}" ]]; then
  echo "model-load runtime modifications do not match the frozen patch" >&2
  exit 2
fi
require_hash \
  "${model}/model-00001-of-00002.safetensors" \
  "${expected_weight_sha256[0]}" \
  model-weight-1
require_hash \
  "${model}/model-00002-of-00002.safetensors" \
  "${expected_weight_sha256[1]}" \
  model-weight-2
if [[ -e "${output_dir}" ]]; then
  echo "model-load output directory already exists" >&2
  exit 2
fi
available_kb=$(df -Pk "${repo}" | awk 'NR==2 {print $4}')
if [[ -z "${available_kb}" || "${available_kb}" -lt 8388608 ]]; then
  echo "model-load smoke requires at least 8 GiB free disk" >&2
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
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HTTP_PROXY HTTPS_PROXY ALL_PROXY \
  http_proxy https_proxy all_proxy

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Slurm exposed no GPU to model-load smoke" >&2
  exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_gpus[@]}" -ne 1 ]]; then
  echo "model-load smoke requires exactly one visible H800" >&2
  exit 2
fi

mkdir -p "${output_dir}"
job_start_epoch=$(date +%s)
queue_wait_seconds=$((job_start_epoch - submit_epoch))
if [[ "${queue_wait_seconds}" -lt 0 ]]; then
  echo "model-load submit epoch is in the future" >&2
  exit 2
fi
echo "vtool model-load smoke start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "tracked revision: ${expected_revision}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"${python_bin}" "${smoke}" \
  --dataset "${dataset}" \
  --model "${model}" \
  --output "${report}"

if ! jq -e '
  .decision == "vtool_vllm_model_load_smoke_passed" and
  .protected_split_contents_accessed == false and
  .optimizer_step_performed == false and
  .prompt_tokens > 0 and
  .completion_tokens > 0
' "${report}" >/dev/null; then
  echo "model-load report contract failed" >&2
  exit 2
fi
echo "vtool model-load smoke end: $(date --iso-8601=seconds)"
echo "report=${report} queue_wait_seconds=${queue_wait_seconds}"
