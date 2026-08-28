#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-textvqa-train-scale-%x-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SCALE_ROLE:?missing BE_SCALE_ROLE}"
: "${BE_SCALE_MANIFEST:?missing BE_SCALE_MANIFEST}"
: "${BE_SCALE_MANIFEST_SHA256:?missing BE_SCALE_MANIFEST_SHA256}"
: "${BE_SCALE_EXPECTED_STATES:?missing BE_SCALE_EXPECTED_STATES}"
: "${BE_SCALE_RUN_DIR:?missing BE_SCALE_RUN_DIR}"
: "${BE_SCALE_SCIENTIFIC_STATUS:?missing BE_SCALE_SCIENTIFIC_STATUS}"
: "${BE_CODE_REVISION:?missing BE_CODE_REVISION}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
rollouts="${BE_SCALE_RUN_DIR}/rollouts.jsonl"

actual_manifest_sha256=$(sha256sum "${BE_SCALE_MANIFEST}")
actual_manifest_sha256=${actual_manifest_sha256%% *}
if [[ "${actual_manifest_sha256}" != "${BE_SCALE_MANIFEST_SHA256}" ]]; then
  echo "Manifest SHA-256 mismatch inside Slurm job" >&2
  exit 2
fi
if [[ "$(wc -l < "${BE_SCALE_MANIFEST}")" -ne "${BE_SCALE_EXPECTED_STATES}" ]]; then
  echo "Manifest state count mismatch inside Slurm job" >&2
  exit 2
fi

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src"

cd "${repo_dir}"
mkdir -p "${BE_SCALE_RUN_DIR}"
"${python_bin}" -m beyond_entropy collect-qwen \
  --manifest "${BE_SCALE_MANIFEST}" \
  --expected-manifest-sha256 "${BE_SCALE_MANIFEST_SHA256}" \
  --output "${rollouts}" \
  --resume \
  --checkpoint-interval 32 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --scorer textvqa \
  --candidate-count 4 \
  --proposer ug-grid \
  --visual-crop-ratio 2.0 \
  --visual-cost 1.0 \
  --generation-seeds 0 \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 20260828 \
  --scientific-status "${BE_SCALE_SCIENTIFIC_STATUS}" \
  --max-new-tokens 32 \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --attention-implementation sdpa \
  --system-prompt "You are a helpful assistant."
