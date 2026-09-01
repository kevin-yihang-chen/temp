#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --job-name=be-infovqa-decar-fitbench
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-infovqa-decar-fitbench-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: worker EXPECTED_REVISION RUNNER_SHA256 SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
runner_sha256=$2
submit_epoch=$3
if [[ ! "${runner_sha256}" =~ ^[0-9a-f]{64}$ || ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "DECAR fit benchmark received malformed provenance" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
runner="${repo}/scripts/benchmark_infographicvqa_decar_fit_runtime.py"
output="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1/ops/oof-fit-runtime-benchmark-v1/report.json"
cd "${repo}"

if [[ "$(git rev-parse HEAD)" != "${expected_revision}" \
  || -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "DECAR fit benchmark revision or tracked worktree changed" >&2
  exit 2
fi
if [[ ! -f "${runner}" \
  || "$(sha256sum "${runner}" | awk '{print $1}')" != "${runner_sha256}" ]]; then
  echo "DECAR fit benchmark runner SHA-256 mismatch" >&2
  exit 2
fi
if [[ -e "${output}" ]]; then
  echo "DECAR fit benchmark refuses to overwrite output" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Slurm exposed no GPU to DECAR fit benchmark" >&2
  exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_gpus[@]}" -ne 1 ]]; then
  echo "DECAR fit benchmark requires exactly one H800" >&2
  exit 2
fi
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ "${gpu_name}" != "NVIDIA H800" ]]; then
  echo "DECAR fit benchmark accelerator changed: ${gpu_name}" >&2
  exit 2
fi

export PYTHONPATH="${repo}/src"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

queue_wait_seconds=$(( $(date +%s) - submit_epoch ))
echo "DECAR fit benchmark start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID} queue_wait_seconds=${queue_wait_seconds}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
"${python_bin}" "${runner}" \
  --decisions 23946 --sources 2204 --embedding-dim 3584 --epochs 5 \
  --device cuda:0 --seed 20260917 --output "${output}"

if ! jq -e '
  .schema == "infographicvqa_decar_full_shape_fit_benchmark_v1" and
  .configuration.decisions == 23946 and .configuration.sources == 2204 and
  .configuration.embedding_dim == 3584 and .configuration.candidates == 4 and
  .configuration.scalar_dim == 16 and .configuration.benchmark_epochs == 5 and
  .runtime.accelerator == "NVIDIA H800" and
  (.runtime.projected_registered_fit_seconds > 0) and
  .contracts.synthetic_inputs_only == true and
  .contracts.task_outcomes_read == false and
  .contracts.scientific_endpoints_computed == false and
  .contracts.validation_or_test_inputs_used == false and
  .contracts.credentials_present == false
' "${output}" >/dev/null; then
  echo "DECAR fit benchmark output contract failed" >&2
  exit 2
fi
echo "DECAR fit benchmark end: $(date --iso-8601=seconds)"
printf 'infographicvqa_decar_fit_benchmark=%s output_sha256=%s\n' \
  "${output}" "$(sha256sum "${output}" | awk '{print $1}')"
