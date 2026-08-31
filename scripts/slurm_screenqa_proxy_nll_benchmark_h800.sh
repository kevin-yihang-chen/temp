#!/usr/bin/env bash
#SBATCH --partition=q-hgpu-small
#SBATCH --gres=gpu:h800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --job-name=be-proxy-nll-h800-bench
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-proxy-nll-h800-benchmark-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_PROXY_EXPECTED_CODE_REVISION BE_PROXY_SCORER_SHA256
  BE_PROXY_SCORE_MODULE_SHA256 BE_PROXY_BENCHMARK_WORKER_SHA256
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
manifest="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-manifest-v1/manifest.jsonl"
rollouts="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-rollouts-v1/merged/rollouts.jsonl"
output_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/proxy-to-outcome-audit-v1/benchmark-h800-${SLURM_JOB_ID}"
output="${output_dir}/answer-nll.jsonl"
scorer="${repo_dir}/scripts/score_visual_action_answer_nll.py"
score_module="${repo_dir}/src/beyond_entropy/answer_likelihood.py"
worker="${repo_dir}/scripts/slurm_screenqa_proxy_nll_benchmark_h800.sh"
manifest_sha256=a2b6941e2a073b24571d2ccb50960f7c1cd70cb0ce53dc8339c7ec44a47f67ec
rollouts_sha256=0437d2a499adccb1b4e19eb0160583789cee00edf244718ecae9e290108bb8c9
model_revision=66285546d2b821cf421d4f5eb2576359d3770cd3

check_hash() {
  local path=$1
  local expected=$2
  local name=$3
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ScreenQA proxy benchmark ${name} hash mismatch" >&2
    exit 2
  fi
}

actual_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_revision}" != "${BE_PROXY_EXPECTED_CODE_REVISION}" ]]; then
  echo "ScreenQA proxy benchmark code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA proxy benchmark" >&2
  exit 2
fi
check_hash "${scorer}" "${BE_PROXY_SCORER_SHA256}" "scorer"
check_hash "${score_module}" "${BE_PROXY_SCORE_MODULE_SHA256}" "score module"
check_hash "${worker}" "${BE_PROXY_BENCHMARK_WORKER_SHA256}" "worker"
check_hash "${manifest}" "${manifest_sha256}" "manifest"
check_hash "${rollouts}" "${rollouts_sha256}" "rollouts"
for protected in \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/calibration-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/formal-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/reserve-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/untouched-manifest-v1"; do
  if [[ -e "${protected}" && ( ! -d "${protected}" || -n "$(find "${protected}" -mindepth 1 -print -quit)" ) ]]; then
    echo "ScreenQA protected role opened before proxy benchmark: ${protected}" >&2
    exit 2
  fi
done
if [[ -e "${output_dir}" ]]; then
  echo "ScreenQA proxy benchmark output already exists" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "${output_dir}"
start_epoch=$(date +%s)

cd "${repo_dir}"
"${python_bin}" "${scorer}" \
  --manifest "${manifest}" \
  --rollouts "${rollouts}" \
  --output "${output}" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --expected-rollouts-sha256 "${rollouts_sha256}" \
  --shard-count 227 \
  --shard-index 0 \
  --checkpoint-interval 8 \
  --resume \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision "${model_revision}" \
  --device-map cuda:0 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --system-prompt "You are a helpful assistant." \
  --code-revision "${actual_revision}" \
  --scientific-status "64-decision H800 engineering throughput benchmark on opened ScreenQA development data; not a scientific result"

end_epoch=$(date +%s)
elapsed_seconds=$((end_epoch - start_epoch))
decisions=$(jq -r '.decisions' "${output%.jsonl}.provenance.json")
records=$(jq -r '.records' "${output%.jsonl}.provenance.json")
if [[ "${decisions}" -ne 64 || "${records}" -ne 320 || "${elapsed_seconds}" -le 0 ]]; then
  echo "ScreenQA proxy H800 benchmark output contract failed" >&2
  exit 2
fi
jq -n \
  --arg schema screenqa_proxy_nll_h800_benchmark_v1 \
  --arg output_sha256 "$(sha256sum "${output}" | cut -d ' ' -f 1)" \
  --arg code_revision "${actual_revision}" \
  --argjson decisions "${decisions}" \
  --argjson records "${records}" \
  --argjson elapsed_seconds "${elapsed_seconds}" \
  '{
    schema: $schema,
    gpu_type: "h800",
    gpu_count: 1,
    decisions: $decisions,
    records: $records,
    elapsed_seconds: $elapsed_seconds,
    decisions_per_second: ($decisions / $elapsed_seconds),
    records_per_second: ($records / $elapsed_seconds),
    projected_four_gpu_full_wall_seconds: ((14511 / ($decisions / $elapsed_seconds)) / 4),
    projected_four_gpu_gpu_minutes: ((14511 / ($decisions / $elapsed_seconds)) / 60),
    output_sha256: $output_sha256,
    code_revision: $code_revision,
    scientific_status: "engineering throughput only; not a scientific result"
  }' > "${output_dir}/benchmark.json"
printf 'screenqa_proxy_nll_h800_benchmark_sha256=%s decisions=%s records=%s elapsed_seconds=%s\n' \
  "$(sha256sum "${output_dir}/benchmark.json" | cut -d ' ' -f 1)" \
  "${decisions}" "${records}" "${elapsed_seconds}"
