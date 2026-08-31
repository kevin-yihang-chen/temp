#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=be-screenqa-proxy-nll-smoke
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-proxy-nll-smoke-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_PROXY_EXPECTED_CODE_REVISION:?missing BE_PROXY_EXPECTED_CODE_REVISION}"
: "${BE_PROXY_SCORER_SHA256:?missing BE_PROXY_SCORER_SHA256}"
: "${BE_PROXY_MODULE_SHA256:?missing BE_PROXY_MODULE_SHA256}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
manifest="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-manifest-v1/manifest.jsonl"
rollouts="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-rollouts-v1/merged/rollouts.jsonl"
output_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/proxy-to-outcome-audit-v1/smoke-v1"
output="${output_dir}/answer-nll.jsonl"
scorer="${repo_dir}/scripts/score_visual_action_answer_nll.py"
module="${repo_dir}/src/beyond_entropy/answer_likelihood.py"
manifest_sha256=a2b6941e2a073b24571d2ccb50960f7c1cd70cb0ce53dc8339c7ec44a47f67ec
rollouts_sha256=0437d2a499adccb1b4e19eb0160583789cee00edf244718ecae9e290108bb8c9
model_revision=66285546d2b821cf421d4f5eb2576359d3770cd3

actual_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_revision}" != "${BE_PROXY_EXPECTED_CODE_REVISION}" ]]; then
  echo "ScreenQA proxy-NLL smoke code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA proxy-NLL smoke" >&2
  exit 2
fi
actual_scorer_sha256=$(sha256sum "${scorer}")
actual_scorer_sha256=${actual_scorer_sha256%% *}
actual_module_sha256=$(sha256sum "${module}")
actual_module_sha256=${actual_module_sha256%% *}
if [[ "${actual_scorer_sha256}" != "${BE_PROXY_SCORER_SHA256}" \
  || "${actual_module_sha256}" != "${BE_PROXY_MODULE_SHA256}" ]]; then
  echo "ScreenQA proxy-NLL smoke implementation hash mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "${manifest}" | cut -d ' ' -f 1)" != "${manifest_sha256}" \
  || "$(sha256sum "${rollouts}" | cut -d ' ' -f 1)" != "${rollouts_sha256}" ]]; then
  echo "ScreenQA proxy-NLL smoke input hash mismatch" >&2
  exit 2
fi
for protected in \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/calibration-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/formal-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/reserve-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/untouched-manifest-v1"; do
  if [[ -e "${protected}" && ( ! -d "${protected}" || -n "$(find "${protected}" -mindepth 1 -print -quit)" ) ]]; then
    echo "ScreenQA protected role opened before proxy-NLL smoke: ${protected}" >&2
    exit 2
  fi
done

export PYTHONPATH="${repo_dir}/src"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BE_CODE_REVISION="${actual_revision}"
mkdir -p "${output_dir}"

args=(
  --manifest "${manifest}"
  --rollouts "${rollouts}"
  --output "${output}"
  --expected-manifest-sha256 "${manifest_sha256}"
  --expected-rollouts-sha256 "${rollouts_sha256}"
  --shard-count 14511
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
  --scientific-status "one-decision real-Qwen engineering smoke on opened ScreenQA development data; not a benchmark or candidate result"
)

cd "${repo_dir}"
"${python_bin}" "${scorer}" "${args[@]}"
first_sha256=$(sha256sum "${output}" | cut -d ' ' -f 1)
"${python_bin}" "${scorer}" "${args[@]}"
second_sha256=$(sha256sum "${output}" | cut -d ' ' -f 1)
if [[ "${first_sha256}" != "${second_sha256}" ]]; then
  echo "ScreenQA proxy-NLL resume changed complete smoke bytes" >&2
  exit 2
fi
if [[ "$(wc -l < "${output}")" -ne 5 \
  || "$(jq -r '.records' "${output%.jsonl}.provenance.json")" -ne 5 \
  || "$(jq -r '.raw_targets_written' "${output%.jsonl}.provenance.json")" != false ]]; then
  echo "ScreenQA proxy-NLL smoke output contract failed" >&2
  exit 2
fi
printf 'screenqa_proxy_nll_smoke_sha256=%s records=5 resume_stable=true\n' "${second_sha256}"
