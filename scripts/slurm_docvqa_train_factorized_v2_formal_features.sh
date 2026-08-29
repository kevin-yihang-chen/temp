#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-docvqa-fv2-formal-features-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_DOCVQA_POLICY_FREEZE BE_DOCVQA_POLICY_FREEZE_SHA256 BE_DOCVQA_MODEL
  BE_DOCVQA_MODEL_SHA256 BE_DOCVQA_MANIFEST BE_DOCVQA_MANIFEST_SHA256
  BE_DOCVQA_MANIFEST_PROVENANCE BE_DOCVQA_MANIFEST_PROVENANCE_SHA256
  BE_DOCVQA_FORMAL_AUDIT BE_DOCVQA_FORMAL_AUDIT_SHA256
  BE_DOCVQA_EXPECTED_STATES BE_DOCVQA_ROLLOUTS BE_DOCVQA_FEATURE_DIR
  BE_DOCVQA_EXPECTED_CODE_REVISION
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
base_features="${BE_DOCVQA_FEATURE_DIR}/features-label-free.pt"
multimodal_features="${BE_DOCVQA_FEATURE_DIR}/features-multimodal-label-free.pt"
attention_features="${BE_DOCVQA_FEATURE_DIR}/features-question-region-attention-label-free.pt"
rollout_audit="${BE_DOCVQA_FEATURE_DIR}/rollouts.audit.json"

if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_DOCVQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "DocVQA formal feature revision differs from policy freeze" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before DocVQA formal features" >&2
  exit 2
fi
export PYTHONPATH="${repo_dir}/src"
"${python_bin}" "${repo_dir}/scripts/verify_docvqa_train_factorized_v2_formal_gate.py" \
  --policy-freeze "${BE_DOCVQA_POLICY_FREEZE}" \
  --expected-policy-freeze-sha256 "${BE_DOCVQA_POLICY_FREEZE_SHA256}" \
  --model "${BE_DOCVQA_MODEL}" \
  --expected-model-sha256 "${BE_DOCVQA_MODEL_SHA256}" \
  --manifest "${BE_DOCVQA_MANIFEST}" \
  --expected-manifest-sha256 "${BE_DOCVQA_MANIFEST_SHA256}" \
  --manifest-provenance "${BE_DOCVQA_MANIFEST_PROVENANCE}" \
  --expected-manifest-provenance-sha256 "${BE_DOCVQA_MANIFEST_PROVENANCE_SHA256}" \
  --audit "${BE_DOCVQA_FORMAL_AUDIT}" \
  --expected-audit-sha256 "${BE_DOCVQA_FORMAL_AUDIT_SHA256}"

export BE_CODE_REVISION="${BE_DOCVQA_EXPECTED_CODE_REVISION}"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "${repo_dir}"
mkdir -p "${BE_DOCVQA_FEATURE_DIR}"
"${python_bin}" scripts/audit_docvqa_train_factorized_v2_formal_rollouts.py \
  --policy-freeze "${BE_DOCVQA_POLICY_FREEZE}" \
  --expected-policy-freeze-sha256 "${BE_DOCVQA_POLICY_FREEZE_SHA256}" \
  --model "${BE_DOCVQA_MODEL}" \
  --expected-model-sha256 "${BE_DOCVQA_MODEL_SHA256}" \
  --manifest "${BE_DOCVQA_MANIFEST}" \
  --expected-manifest-sha256 "${BE_DOCVQA_MANIFEST_SHA256}" \
  --manifest-provenance "${BE_DOCVQA_MANIFEST_PROVENANCE}" \
  --expected-manifest-provenance-sha256 "${BE_DOCVQA_MANIFEST_PROVENANCE_SHA256}" \
  --formal-audit "${BE_DOCVQA_FORMAL_AUDIT}" \
  --expected-formal-audit-sha256 "${BE_DOCVQA_FORMAL_AUDIT_SHA256}" \
  --rollouts "${BE_DOCVQA_ROLLOUTS}" \
  --expected-states "${BE_DOCVQA_EXPECTED_STATES}" \
  --expected-code-revision "${BE_DOCVQA_EXPECTED_CODE_REVISION}" \
  --output "${rollout_audit}" \
  --resume

rollouts_sha256=$(sha256sum "${BE_DOCVQA_ROLLOUTS}")
rollouts_sha256=${rollouts_sha256%% *}
"${python_bin}" -m beyond_entropy extract-qwen-features \
  --rollouts "${BE_DOCVQA_ROLLOUTS}" \
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
  --rollouts "${BE_DOCVQA_ROLLOUTS}" \
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
  --rollouts "${BE_DOCVQA_ROLLOUTS}" \
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
  --rollouts "${BE_DOCVQA_ROLLOUTS}" \
  > "${BE_DOCVQA_FEATURE_DIR}/label-free-audit.json"
test -s "${BE_DOCVQA_FEATURE_DIR}/label-free-audit.json"
