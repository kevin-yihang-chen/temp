#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-docvqa-train-factorized-v2-%x-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_DOCVQA_ROLE:?missing BE_DOCVQA_ROLE}"
: "${BE_DOCVQA_MANIFEST:?missing BE_DOCVQA_MANIFEST}"
: "${BE_DOCVQA_MANIFEST_SHA256:?missing BE_DOCVQA_MANIFEST_SHA256}"
: "${BE_DOCVQA_MANIFEST_PROVENANCE_SHA256:?missing BE_DOCVQA_MANIFEST_PROVENANCE_SHA256}"
: "${BE_DOCVQA_MANIFEST_AUDIT:?missing BE_DOCVQA_MANIFEST_AUDIT}"
: "${BE_DOCVQA_MANIFEST_AUDIT_SHA256:?missing BE_DOCVQA_MANIFEST_AUDIT_SHA256}"
: "${BE_DOCVQA_EXPECTED_STATES:?missing BE_DOCVQA_EXPECTED_STATES}"
: "${BE_DOCVQA_ROLLOUTS:?missing BE_DOCVQA_ROLLOUTS}"
: "${BE_DOCVQA_ROLLOUTS_SHA256:?missing BE_DOCVQA_ROLLOUTS_SHA256}"
: "${BE_DOCVQA_FEATURE_DIR:?missing BE_DOCVQA_FEATURE_DIR}"
: "${BE_DOCVQA_EXPECTED_CODE_REVISION:?missing BE_DOCVQA_EXPECTED_CODE_REVISION}"
: "${BE_DOCVQA_ALLOCATION:?missing BE_DOCVQA_ALLOCATION}"
: "${BE_DOCVQA_ALLOCATION_SHA256:?missing BE_DOCVQA_ALLOCATION_SHA256}"
: "${BE_DOCVQA_ALLOCATION_AUDIT:?missing BE_DOCVQA_ALLOCATION_AUDIT}"
: "${BE_DOCVQA_ALLOCATION_AUDIT_SHA256:?missing BE_DOCVQA_ALLOCATION_AUDIT_SHA256}"
: "${BE_DOCVQA_PROTOCOL:?missing BE_DOCVQA_PROTOCOL}"
: "${BE_DOCVQA_FORMAL_OUTPUT_DIR:?missing BE_DOCVQA_FORMAL_OUTPUT_DIR}"

case "${BE_DOCVQA_ROLE}" in
  ranker_training)
    scientific_status="fresh DocVQA-train factorized-v2 ranker sibling bank; outcomes may train the sole candidate only"
    ;;
  risk_calibration)
    scientific_status="fresh DocVQA-train factorized-v2 calibration sibling bank; outcomes may calibrate the sole frozen candidate only"
    : "${BE_DOCVQA_CANDIDATE:?missing BE_DOCVQA_CANDIDATE}"
    : "${BE_DOCVQA_CANDIDATE_SHA256:?missing BE_DOCVQA_CANDIDATE_SHA256}"
    : "${BE_DOCVQA_CANDIDATE_AUDIT:?missing BE_DOCVQA_CANDIDATE_AUDIT}"
    : "${BE_DOCVQA_CANDIDATE_AUDIT_SHA256:?missing BE_DOCVQA_CANDIDATE_AUDIT_SHA256}"
    ;;
  *)
    echo "Unsupported DocVQA development role: ${BE_DOCVQA_ROLE}" >&2
    exit 2
    ;;
esac

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
model_revision=66285546d2b821cf421d4f5eb2576359d3770cd3
base_features="${BE_DOCVQA_FEATURE_DIR}/features-label-free.pt"
multimodal_features="${BE_DOCVQA_FEATURE_DIR}/features-multimodal-label-free.pt"
attention_features="${BE_DOCVQA_FEATURE_DIR}/features-question-region-attention-label-free.pt"
rollout_audit="${BE_DOCVQA_FEATURE_DIR}/rollouts.audit.json"

tracked_status=$(git -C "${repo_dir}" status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before DocVQA feature extraction" >&2
  exit 2
fi
actual_code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_code_revision}" != "${BE_DOCVQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "DocVQA feature code revision mismatch" >&2
  exit 2
fi
actual_rollouts_sha256=$(sha256sum "${BE_DOCVQA_ROLLOUTS}")
actual_rollouts_sha256=${actual_rollouts_sha256%% *}
if [[ "${actual_rollouts_sha256}" != "${BE_DOCVQA_ROLLOUTS_SHA256}" ]]; then
  echo "DocVQA feature rollout SHA-256 mismatch" >&2
  exit 2
fi

export BE_CODE_REVISION="${actual_code_revision}"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_dir}/src"

cd "${repo_dir}"
verify_args=(
  --role "${BE_DOCVQA_ROLE}"
  --manifest-dir "$(dirname "${BE_DOCVQA_MANIFEST}")"
  --expected-manifest-sha256 "${BE_DOCVQA_MANIFEST_SHA256}"
  --expected-manifest-provenance-sha256 "${BE_DOCVQA_MANIFEST_PROVENANCE_SHA256}"
  --manifest-audit "${BE_DOCVQA_MANIFEST_AUDIT}"
  --expected-manifest-audit-sha256 "${BE_DOCVQA_MANIFEST_AUDIT_SHA256}"
  --allocation "${BE_DOCVQA_ALLOCATION}"
  --allocation-audit "${BE_DOCVQA_ALLOCATION_AUDIT}"
  --expected-allocation-sha256 "${BE_DOCVQA_ALLOCATION_SHA256}"
  --expected-allocation-audit-sha256 "${BE_DOCVQA_ALLOCATION_AUDIT_SHA256}"
  --protocol "${BE_DOCVQA_PROTOCOL}"
  --expected-code-revision "${BE_DOCVQA_EXPECTED_CODE_REVISION}"
  --formal-output-dir "${BE_DOCVQA_FORMAL_OUTPUT_DIR}"
)
if [[ "${BE_DOCVQA_ROLE}" == "risk_calibration" ]]; then
  verify_args+=(
    --candidate "${BE_DOCVQA_CANDIDATE}"
    --candidate-audit "${BE_DOCVQA_CANDIDATE_AUDIT}"
    --expected-candidate-sha256 "${BE_DOCVQA_CANDIDATE_SHA256}"
    --expected-candidate-audit-sha256 "${BE_DOCVQA_CANDIDATE_AUDIT_SHA256}"
  )
fi
"${python_bin}" scripts/verify_docvqa_train_factorized_v2_manifest.py \
  "${verify_args[@]}"
mkdir -p "${BE_DOCVQA_FEATURE_DIR}"
if [[ "${BE_DOCVQA_ROLE}" == "risk_calibration" ]]; then
  "${python_bin}" scripts/audit_docvqa_train_factorized_v2_rollouts.py \
    --candidate "${BE_DOCVQA_CANDIDATE}" \
    --candidate-audit "${BE_DOCVQA_CANDIDATE_AUDIT}" \
    --expected-candidate-sha256 "${BE_DOCVQA_CANDIDATE_SHA256}" \
    --expected-candidate-audit-sha256 "${BE_DOCVQA_CANDIDATE_AUDIT_SHA256}" \
    --allocation "${BE_DOCVQA_ALLOCATION}" \
    --allocation-audit "${BE_DOCVQA_ALLOCATION_AUDIT}" \
    --expected-allocation-sha256 "${BE_DOCVQA_ALLOCATION_SHA256}" \
    --expected-allocation-audit-sha256 "${BE_DOCVQA_ALLOCATION_AUDIT_SHA256}" \
    --manifest-dir "$(dirname "${BE_DOCVQA_MANIFEST}")" \
    --expected-manifest-sha256 "${BE_DOCVQA_MANIFEST_SHA256}" \
    --expected-manifest-provenance-sha256 "${BE_DOCVQA_MANIFEST_PROVENANCE_SHA256}" \
    --rollouts "${BE_DOCVQA_ROLLOUTS}" \
    --protocol "${BE_DOCVQA_PROTOCOL}" \
    --expected-code-revision "${BE_DOCVQA_EXPECTED_CODE_REVISION}" \
    --output "${rollout_audit}" \
    --resume
else
  "${python_bin}" scripts/audit_scaled_textvqa_rollouts.py \
    --manifest "${BE_DOCVQA_MANIFEST}" \
    --manifest-sha256 "${BE_DOCVQA_MANIFEST_SHA256}" \
    --rollouts "${BE_DOCVQA_ROLLOUTS}" \
    --expected-states "${BE_DOCVQA_EXPECTED_STATES}" \
    --expected-model-revision "${model_revision}" \
    --expected-scientific-status "${scientific_status}" \
    --output "${rollout_audit}"
fi

base_resume=()
if [[ -e "${base_features}" ]]; then
  base_resume=(--resume)
fi
"${python_bin}" -m beyond_entropy extract-qwen-features \
  --rollouts "${BE_DOCVQA_ROLLOUTS}" \
  --expected-rollouts-sha256 "${BE_DOCVQA_ROLLOUTS_SHA256}" \
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
  "${multimodal_resume[@]}"

attention_resume=()
if [[ -e "${attention_features}" ]]; then
  attention_resume=(--resume)
fi
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
  "${attention_resume[@]}"

"${python_bin}" scripts/audit_label_free_semantic_features.py \
  --features "${attention_features}" \
  --rollouts "${BE_DOCVQA_ROLLOUTS}" \
  > "${BE_DOCVQA_FEATURE_DIR}/label-free-audit.json"
test -s "${BE_DOCVQA_FEATURE_DIR}/label-free-audit.json"
