#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-textvqa-fv2-formal-features-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_FV2_POLICY_FREEZE BE_FV2_POLICY_FREEZE_SHA256 BE_FV2_MODEL
  BE_FV2_MODEL_SHA256 BE_FV2_MANIFEST BE_FV2_MANIFEST_SHA256
  BE_FV2_MANIFEST_PROVENANCE BE_FV2_MANIFEST_PROVENANCE_SHA256
  BE_FV2_FORMAL_AUDIT BE_FV2_FORMAL_AUDIT_SHA256 BE_FV2_EXPECTED_STATES
  BE_FV2_ROLLOUTS BE_FV2_FEATURE_DIR BE_FV2_SCIENTIFIC_STATUS BE_CODE_REVISION
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
model_revision=66285546d2b821cf421d4f5eb2576359d3770cd3
base_features="${BE_FV2_FEATURE_DIR}/features-label-free.pt"
multimodal_features="${BE_FV2_FEATURE_DIR}/features-multimodal-label-free.pt"
attention_features="${BE_FV2_FEATURE_DIR}/features-question-region-attention-label-free.pt"
rollout_audit="${BE_FV2_FEATURE_DIR}/rollouts.audit.json"

if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_CODE_REVISION}" ]]; then
  echo "formal feature revision differs from the frozen submission" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before formal feature extraction" >&2
  exit 2
fi
export PYTHONPATH="${repo_dir}/src"
/userhome/cs3/yihangc/anaconda3/bin/python \
  "${repo_dir}/scripts/verify_factorized_v2_formal_gate.py" \
  --policy-freeze "${BE_FV2_POLICY_FREEZE}" \
  --expected-policy-freeze-sha256 "${BE_FV2_POLICY_FREEZE_SHA256}" \
  --model "${BE_FV2_MODEL}" \
  --expected-model-sha256 "${BE_FV2_MODEL_SHA256}" \
  --manifest "${BE_FV2_MANIFEST}" \
  --expected-manifest-sha256 "${BE_FV2_MANIFEST_SHA256}" \
  --manifest-provenance "${BE_FV2_MANIFEST_PROVENANCE}" \
  --expected-manifest-provenance-sha256 "${BE_FV2_MANIFEST_PROVENANCE_SHA256}" \
  --audit "${BE_FV2_FORMAL_AUDIT}" \
  --expected-audit-sha256 "${BE_FV2_FORMAL_AUDIT_SHA256}"

export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BE_CODE_REVISION
cd "${repo_dir}"
mkdir -p "${BE_FV2_FEATURE_DIR}"
"${python_bin}" scripts/audit_scaled_textvqa_rollouts.py \
  --manifest "${BE_FV2_MANIFEST}" \
  --manifest-sha256 "${BE_FV2_MANIFEST_SHA256}" \
  --rollouts "${BE_FV2_ROLLOUTS}" \
  --expected-states "${BE_FV2_EXPECTED_STATES}" \
  --expected-model-revision "${model_revision}" \
  --expected-scientific-status "${BE_FV2_SCIENTIFIC_STATUS}" \
  --output "${rollout_audit}"

rollouts_sha256=$(sha256sum "${BE_FV2_ROLLOUTS}")
rollouts_sha256=${rollouts_sha256%% *}
"${python_bin}" -m beyond_entropy extract-qwen-features \
  --rollouts "${BE_FV2_ROLLOUTS}" \
  --expected-rollouts-sha256 "${rollouts_sha256}" \
  --output "${base_features}" \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision "${model_revision}" \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --attention-implementation sdpa \
  --question-feature-mode input_mean \
  --exclude-outcomes \
  --resume

"${python_bin}" scripts/reembed_contextual_questions.py \
  --source-features "${base_features}" \
  --rollouts "${BE_FV2_ROLLOUTS}" \
  --output "${multimodal_features}" \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --revision "${model_revision}" \
  --device-map cuda:0 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --mode multimodal-original \
  --batch-size 4 \
  --checkpoint-interval 64 \
  --resume

"${python_bin}" scripts/extract_question_region_attention.py \
  --source-features "${multimodal_features}" \
  --rollouts "${BE_FV2_ROLLOUTS}" \
  --output "${attention_features}" \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --revision "${model_revision}" \
  --device-map cuda:0 \
  --dtype bfloat16 \
  --top-layers 4 \
  --checkpoint-interval 32 \
  --resume

"${python_bin}" scripts/audit_label_free_semantic_features.py \
  --features "${attention_features}" \
  --rollouts "${BE_FV2_ROLLOUTS}" \
  > "${BE_FV2_FEATURE_DIR}/label-free-audit.json"
test -s "${BE_FV2_FEATURE_DIR}/label-free-audit.json"
