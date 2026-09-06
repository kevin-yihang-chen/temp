#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --job-name=be-phase-c-rollout
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-phase-c-rollout-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL
#SBATCH --no-requeue

set -euo pipefail

required=(
  BE_PHASE_C_BENCHMARK BE_PHASE_C_CODE_REVISION BE_PHASE_C_MANIFEST_SHA256
  BE_PHASE_C_ALLOCATION_REPORT_SHA256 BE_PHASE_C_WORKER_SHA256
  BE_PHASE_C_GENERATOR_SHA256 BE_PHASE_C_MERGER_CLI_SHA256
  BE_PHASE_C_MERGER_MODULE_SHA256 BE_PHASE_C_BACKEND_SHA256
  BE_PHASE_C_ROLLOUT_MODULE_SHA256 BE_PHASE_C_SCHEMA_SHA256
  BE_PHASE_C_SHARDING_SHA256 BE_PHASE_C_BENCHMARKS_SHA256
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing ${name}" >&2; exit 2; }
done

repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
worker="${repo}/scripts/slurm_factorized_phase_c_train_rollouts.sh"
generator="${repo}/scripts/generate_counterfactual_prefixes.py"
merger_cli="${repo}/scripts/merge_sequential_rollout_shards.py"
merger_module="${repo}/src/beyond_entropy/sequential_rollout_shards.py"
backend="${repo}/src/beyond_entropy/qwen_backend.py"
rollout_module="${repo}/src/beyond_entropy/sequential_rollout.py"
schema="${repo}/src/beyond_entropy/sequential_schema.py"
sharding="${repo}/src/beyond_entropy/sharding.py"
benchmarks="${repo}/src/beyond_entropy/benchmarks.py"
allocation_report="${repo}/data/factorized-phase-c-v1/allocation.report.json"
manifest="${repo}/data/factorized-phase-c-v1/${BE_PHASE_C_BENCHMARK}/train/manifest.jsonl"
shard_count=4

case "${BE_PHASE_C_BENCHMARK}" in
  chartqa|docvqa|hrbench) ;;
  *) echo "invalid Phase-C benchmark: ${BE_PHASE_C_BENCHMARK}" >&2; exit 2 ;;
esac

check_hash() {
  local actual
  actual=$(sha256sum "$1")
  actual=${actual%% *}
  [[ "${actual}" == "$2" ]] || {
    echo "Phase-C rollout $3 hash mismatch" >&2
    exit 2
  }
}

[[ "$(git -C "${repo}" rev-parse HEAD)" == "${BE_PHASE_C_CODE_REVISION}" ]] || {
  echo "Phase-C rollout code revision mismatch" >&2; exit 2;
}
[[ -z "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]] || {
  echo "tracked worktree must be clean" >&2; exit 2;
}
check_hash "${manifest}" "${BE_PHASE_C_MANIFEST_SHA256}" manifest
check_hash "${allocation_report}" "${BE_PHASE_C_ALLOCATION_REPORT_SHA256}" allocation_report
check_hash "${worker}" "${BE_PHASE_C_WORKER_SHA256}" worker
check_hash "${generator}" "${BE_PHASE_C_GENERATOR_SHA256}" generator
check_hash "${merger_cli}" "${BE_PHASE_C_MERGER_CLI_SHA256}" merger_cli
check_hash "${merger_module}" "${BE_PHASE_C_MERGER_MODULE_SHA256}" merger_module
check_hash "${backend}" "${BE_PHASE_C_BACKEND_SHA256}" backend
check_hash "${rollout_module}" "${BE_PHASE_C_ROLLOUT_MODULE_SHA256}" rollout_module
check_hash "${schema}" "${BE_PHASE_C_SCHEMA_SHA256}" schema
check_hash "${sharding}" "${BE_PHASE_C_SHARDING_SHA256}" sharding
check_hash "${benchmarks}" "${BE_PHASE_C_BENCHMARKS_SHA256}" benchmarks

IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ "${#allocated_gpus[@]}" -eq "${shard_count}" ]] || {
  echo "expected ${shard_count} visible GPUs, got ${#allocated_gpus[@]}" >&2
  exit 2
}

run_root="${repo}/artifacts/factorized-potential-outcomes-v1/phase-c-data/${BE_PHASE_C_BENCHMARK}/train-v1/job-${SLURM_JOB_ID}"
[[ ! -e "${run_root}" ]] || { echo "run root already exists: ${run_root}" >&2; exit 2; }
mkdir -p "${run_root}"

export PYTHONPATH="${repo}/src"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

run_shard() {
  local index=$1
  local shard_name
  shard_name=$(printf 'shard-%05d-of-%05d' "${index}" "${shard_count}")
  mkdir -p "${run_root}/${shard_name}"
  CUDA_VISIBLE_DEVICES="${allocated_gpus[${index}]}" \
    "${python_bin}" "${generator}" \
      --manifest "${manifest}" \
      --output "${run_root}/${shard_name}/rollouts.jsonl" \
      --benchmark "${BE_PHASE_C_BENCHMARK}" \
      --dataset-role train \
      --model Qwen/Qwen2.5-VL-3B-Instruct \
      --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
      --device-map cuda:0 \
      --generation-seed 0 \
      --shard-count "${shard_count}" \
      --shard-index "${index}" \
      --checkpoint-interval 16
}

pids=()
failed=0
for index in 0 1 2 3; do
  run_shard "${index}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
[[ "${failed}" -eq 0 ]] || { echo "one or more rollout shards failed" >&2; exit 1; }

"${python_bin}" "${merger_cli}" \
  --manifest "${manifest}" \
  --expected-manifest-sha256 "${BE_PHASE_C_MANIFEST_SHA256}" \
  --run-root "${run_root}" \
  --shard-count "${shard_count}" \
  --output-dir "${run_root}/merged" \
  --expected-code-revision "${BE_PHASE_C_CODE_REVISION}" \
  --benchmark "${BE_PHASE_C_BENCHMARK}" \
  --dataset-role train \
  --generation-seed 0
