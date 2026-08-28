#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-textvqa-attention-fresh-features-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_FORMAL_ROLLOUTS:?missing BE_FORMAL_ROLLOUTS}"
: "${BE_EXPECTED_ROLLOUTS_SHA256:?missing BE_EXPECTED_ROLLOUTS_SHA256}"
: "${BE_FORMAL_FEATURE_DIR:?missing BE_FORMAL_FEATURE_DIR}"
: "${BE_FROZEN_MODEL:?missing BE_FROZEN_MODEL}"
: "${BE_EXPECTED_MODEL_SHA256:?missing BE_EXPECTED_MODEL_SHA256}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
model_revision=66285546d2b821cf421d4f5eb2576359d3770cd3
base_features="${BE_FORMAL_FEATURE_DIR}/features-label-free.pt"
multimodal_features="${BE_FORMAL_FEATURE_DIR}/features-multimodal-label-free.pt"
attention_features="${BE_FORMAL_FEATURE_DIR}/features-question-region-attention-label-free.pt"

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src"

cd "${repo_dir}"
mkdir -p "${BE_FORMAL_FEATURE_DIR}"
if [[ "$(wc -l < "${BE_FORMAL_ROLLOUTS}")" -ne 15830 ]]; then
  echo "fresh-split formal rollout row-count mismatch" >&2
  exit 2
fi
actual_rollouts_sha256=$(sha256sum "${BE_FORMAL_ROLLOUTS}")
actual_rollouts_sha256=${actual_rollouts_sha256%% *}
if [[ "${actual_rollouts_sha256}" != "${BE_EXPECTED_ROLLOUTS_SHA256}" ]]; then
  echo "fresh-split formal rollout SHA-256 mismatch" >&2
  exit 2
fi
actual_model_sha256=$(sha256sum "${BE_FROZEN_MODEL}")
actual_model_sha256=${actual_model_sha256%% *}
if [[ "${actual_model_sha256}" != "${BE_EXPECTED_MODEL_SHA256}" ]]; then
  echo "frozen model SHA-256 mismatch" >&2
  exit 2
fi

export BE_CODE_REVISION
BE_CODE_REVISION=$(git rev-parse HEAD)
base_resume=()
if [[ -e "${base_features}" ]]; then
  base_resume=(--resume)
fi
"${python_bin}" -m beyond_entropy extract-qwen-features \
  --rollouts "${BE_FORMAL_ROLLOUTS}" \
  --expected-rollouts-sha256 "${BE_EXPECTED_ROLLOUTS_SHA256}" \
  --output "${base_features}" \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision "${model_revision}" \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --attention-implementation sdpa \
  --question-feature-mode input_mean \
  --exclude-outcomes \
  "${base_resume[@]}"

multimodal_resume=()
if [[ -e "${multimodal_features}" ]]; then
  multimodal_resume=(--resume)
fi
"${python_bin}" scripts/reembed_contextual_questions.py \
  --source-features "${base_features}" \
  --rollouts "${BE_FORMAL_ROLLOUTS}" \
  --output "${multimodal_features}" \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --revision "${model_revision}" \
  --device-map cuda:0 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --mode multimodal-original \
  --batch-size 4 \
  --checkpoint-interval 64 \
  "${multimodal_resume[@]}"

attention_resume=()
if [[ -e "${attention_features}" ]]; then
  attention_resume=(--resume)
fi
"${python_bin}" scripts/extract_question_region_attention.py \
  --source-features "${multimodal_features}" \
  --rollouts "${BE_FORMAL_ROLLOUTS}" \
  --output "${attention_features}" \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --revision "${model_revision}" \
  --device-map cuda:0 \
  --dtype bfloat16 \
  --top-layers 4 \
  --checkpoint-interval 32 \
  "${attention_resume[@]}"

"${python_bin}" scripts/audit_label_free_semantic_features.py \
  --features "${attention_features}" \
  --rollouts "${BE_FORMAL_ROLLOUTS}" \
  > "${BE_FORMAL_FEATURE_DIR}/label-free-audit.json"
cat "${BE_FORMAL_FEATURE_DIR}/label-free-audit.json"

