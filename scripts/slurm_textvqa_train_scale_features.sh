#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-textvqa-train-scale-%x-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SCALE_ROLE:?missing BE_SCALE_ROLE}"
: "${BE_SCALE_MANIFEST:?missing BE_SCALE_MANIFEST}"
: "${BE_SCALE_MANIFEST_SHA256:?missing BE_SCALE_MANIFEST_SHA256}"
: "${BE_SCALE_EXPECTED_STATES:?missing BE_SCALE_EXPECTED_STATES}"
: "${BE_SCALE_ROLLOUTS:?missing BE_SCALE_ROLLOUTS}"
: "${BE_SCALE_SCIENTIFIC_STATUS:?missing BE_SCALE_SCIENTIFIC_STATUS}"
: "${BE_SCALE_FEATURE_DIR:?missing BE_SCALE_FEATURE_DIR}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
model_revision=66285546d2b821cf421d4f5eb2576359d3770cd3
base_features="${BE_SCALE_FEATURE_DIR}/features-label-free.pt"
multimodal_features="${BE_SCALE_FEATURE_DIR}/features-multimodal-label-free.pt"
attention_features="${BE_SCALE_FEATURE_DIR}/features-question-region-attention-label-free.pt"
rollout_audit="${BE_SCALE_FEATURE_DIR}/rollouts.audit.json"

if [[ -n "${BE_SCALE_POLICY_FREEZE:-}" ]]; then
  : "${BE_SCALE_POLICY_FREEZE_SHA256:?missing BE_SCALE_POLICY_FREEZE_SHA256}"
  : "${BE_SCALE_FROZEN_MODEL:?missing BE_SCALE_FROZEN_MODEL}"
  : "${BE_SCALE_FROZEN_MODEL_SHA256:?missing BE_SCALE_FROZEN_MODEL_SHA256}"
  : "${BE_SCALE_FORMAL_AUDIT:?missing BE_SCALE_FORMAL_AUDIT}"
  : "${BE_SCALE_FORMAL_AUDIT_SHA256:?missing BE_SCALE_FORMAL_AUDIT_SHA256}"
  PYTHONPATH="${repo_dir}/src" /userhome/cs3/yihangc/anaconda3/bin/python \
    "${repo_dir}/scripts/verify_scaled_textvqa_formal_gate.py" \
    --policy-freeze "${BE_SCALE_POLICY_FREEZE}" \
    --expected-policy-freeze-sha256 "${BE_SCALE_POLICY_FREEZE_SHA256}" \
    --model "${BE_SCALE_FROZEN_MODEL}" \
    --expected-model-sha256 "${BE_SCALE_FROZEN_MODEL_SHA256}" \
    --manifest "${BE_SCALE_MANIFEST}" \
    --expected-manifest-sha256 "${BE_SCALE_MANIFEST_SHA256}" \
    --audit "${BE_SCALE_FORMAL_AUDIT}" \
    --expected-audit-sha256 "${BE_SCALE_FORMAL_AUDIT_SHA256}"
fi

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src"
tracked_status=$(git -C "${repo_dir}" status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before scaled feature extraction" >&2
  exit 2
fi
export BE_CODE_REVISION
BE_CODE_REVISION=$(git -C "${repo_dir}" rev-parse HEAD)

cd "${repo_dir}"
mkdir -p "${BE_SCALE_FEATURE_DIR}"
"${python_bin}" scripts/audit_scaled_textvqa_rollouts.py \
  --manifest "${BE_SCALE_MANIFEST}" \
  --manifest-sha256 "${BE_SCALE_MANIFEST_SHA256}" \
  --rollouts "${BE_SCALE_ROLLOUTS}" \
  --expected-states "${BE_SCALE_EXPECTED_STATES}" \
  --expected-model-revision "${model_revision}" \
  --expected-scientific-status "${BE_SCALE_SCIENTIFIC_STATUS}" \
  --output "${rollout_audit}"

rollouts_sha256=$(sha256sum "${BE_SCALE_ROLLOUTS}")
rollouts_sha256=${rollouts_sha256%% *}
base_resume=()
if [[ -e "${base_features}" ]]; then
  base_resume=(--resume)
fi
"${python_bin}" -m beyond_entropy extract-qwen-features \
  --rollouts "${BE_SCALE_ROLLOUTS}" \
  --expected-rollouts-sha256 "${rollouts_sha256}" \
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
  --rollouts "${BE_SCALE_ROLLOUTS}" \
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
  --rollouts "${BE_SCALE_ROLLOUTS}" \
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
  --rollouts "${BE_SCALE_ROLLOUTS}" \
  > "${BE_SCALE_FEATURE_DIR}/label-free-audit.json"
test -s "${BE_SCALE_FEATURE_DIR}/label-free-audit.json"
