#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=be-docvqa-proxy-smoke
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-docvqa-proxy-nll-smoke-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_DOCVQA_PROXY_EXPECTED_CODE_REVISION
  BE_DOCVQA_PROXY_SCORER_SHA256
  BE_DOCVQA_PROXY_MODULE_SHA256
  BE_DOCVQA_PROXY_WORKER_SHA256
  BE_DOCVQA_PROXY_PROTOCOL_SHA256
  BE_DOCVQA_PROXY_IMPLEMENTATION_SHA256
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
manifest="${repo_dir}/data/docvqa-train-factorized-v2/ranker-training/manifest.jsonl"
rollouts="${repo_dir}/artifacts/docvqa-train-factorized-v2/ranker-training/qwen3b-c4-seed0/rollouts.jsonl"
protocol="${repo_dir}/artifacts/docvqa-train-factorized-v2/ops/proxy-to-outcome-cross-domain-protocol-v1.md"
implementation="${repo_dir}/artifacts/docvqa-train-factorized-v2/ops/proxy-to-outcome-cross-domain-implementation-v1.md"
output_dir="${repo_dir}/artifacts/docvqa-train-factorized-v2/proxy-to-outcome-cross-domain-v1/smoke-v1"
output="${output_dir}/answer-nll.jsonl"
scorer="${repo_dir}/scripts/score_visual_action_answer_nll.py"
module="${repo_dir}/src/beyond_entropy/answer_likelihood.py"
worker="${repo_dir}/scripts/slurm_docvqa_proxy_nll_smoke.sh"
manifest_sha256=871ea5b924badba8e0f23477fbd40d2e77085a5f69726d0c48e674e72d64a25d
rollouts_sha256=9109d5c896e59959937c2171e897dd81341237ff51fc5745fb4c56325ddd8fc3
protocol_sha256=106879da7d15db351a4145e5a06c43fc3f33803182d1ca4e6f08362b076f8cbe
implementation_sha256=43d884c2749539d9d25b05038e7d4897783c1ae38041a00ff70440b42d4e5a73
scorer_sha256=d278b8cd50a58133d6f512467dce8b53a38a690ade3e874b9721c61adabe523d
module_sha256=10c2b647b6ebbc036d6ce06b046521476b4f3d26e73e66b63b7d3f32382b51e4
model_revision=66285546d2b821cf421d4f5eb2576359d3770cd3

check_hash() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "DocVQA proxy smoke ${label} hash mismatch" >&2
    exit 2
  fi
}

actual_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_revision}" != "${BE_DOCVQA_PROXY_EXPECTED_CODE_REVISION}" ]]; then
  echo "DocVQA proxy smoke code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before DocVQA proxy smoke" >&2
  exit 2
fi
check_hash "${manifest}" "${manifest_sha256}" manifest
check_hash "${rollouts}" "${rollouts_sha256}" rollouts
check_hash "${protocol}" "${protocol_sha256}" protocol
check_hash "${implementation}" "${implementation_sha256}" implementation
check_hash "${scorer}" "${scorer_sha256}" scorer
check_hash "${module}" "${module_sha256}" module
check_hash "${worker}" "${BE_DOCVQA_PROXY_WORKER_SHA256}" worker
if [[ "${BE_DOCVQA_PROXY_SCORER_SHA256}" != "${scorer_sha256}" \
  || "${BE_DOCVQA_PROXY_MODULE_SHA256}" != "${module_sha256}" \
  || "${BE_DOCVQA_PROXY_PROTOCOL_SHA256}" != "${protocol_sha256}" \
  || "${BE_DOCVQA_PROXY_IMPLEMENTATION_SHA256}" != "${implementation_sha256}" ]]; then
  echo "DocVQA proxy smoke submitted contract mismatch" >&2
  exit 2
fi
if [[ -e "${output}" || -e "${output%.jsonl}.provenance.json" ]]; then
  echo "DocVQA proxy smoke output already exists" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${output_dir}"

args=(
  --manifest "${manifest}"
  --rollouts "${rollouts}"
  --output "${output}"
  --expected-manifest-sha256 "${manifest_sha256}"
  --expected-rollouts-sha256 "${rollouts_sha256}"
  --shard-count 13580
  --shard-index 0
  --checkpoint-interval 1
  --resume
  --model Qwen/Qwen2.5-VL-3B-Instruct
  --model-revision "${model_revision}"
  --device-map cuda:0
  --dtype bfloat16
  --attention-implementation sdpa
  --min-pixels 200704
  --max-pixels 602112
  --system-prompt "You are a helpful assistant."
  --code-revision "${actual_revision}"
  --scientific-status "one-decision real-Qwen engineering smoke on opened DocVQA ranker-development data; no protected-role input and not a scientific result"
)

cd "${repo_dir}"
"${python_bin}" "${scorer}" "${args[@]}"
first_sha256=$(sha256sum "${output}" | cut -d ' ' -f 1)
"${python_bin}" "${scorer}" "${args[@]}"
second_sha256=$(sha256sum "${output}" | cut -d ' ' -f 1)
provenance="${output%.jsonl}.provenance.json"
if [[ "${first_sha256}" != "${second_sha256}" ]]; then
  echo "DocVQA proxy smoke resume changed complete bytes" >&2
  exit 2
fi
if [[ "$(wc -l < "${output}")" -ne 5 \
  || "$(jq -r '.records' "${provenance}")" -ne 5 \
  || "$(jq -r '.decisions' "${provenance}")" -ne 1 \
  || "$(jq -r '.raw_targets_written' "${provenance}")" != false \
  || "$(jq -r '.measurement_config.accelerator_name' "${provenance}")" != *4090* ]]; then
  echo "DocVQA proxy smoke output contract failed" >&2
  exit 2
fi
if rg -q '"target_answer"[[:space:]]*:' "${output}"; then
  echo "DocVQA proxy smoke leaked raw target text" >&2
  exit 2
fi
printf 'docvqa_proxy_nll_smoke_sha256=%s records=5 resume_stable=true\n' "${second_sha256}"
