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
: "${BE_DOCVQA_RUN_DIR:?missing BE_DOCVQA_RUN_DIR}"
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
rollouts="${BE_DOCVQA_RUN_DIR}/rollouts.jsonl"

tracked_status=$(git -C "${repo_dir}" status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before DocVQA rollout" >&2
  exit 2
fi
actual_code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_code_revision}" != "${BE_DOCVQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "DocVQA rollout code revision mismatch" >&2
  exit 2
fi
actual_manifest_sha256=$(sha256sum "${BE_DOCVQA_MANIFEST}")
actual_manifest_sha256=${actual_manifest_sha256%% *}
if [[ "${actual_manifest_sha256}" != "${BE_DOCVQA_MANIFEST_SHA256}" ]]; then
  echo "DocVQA manifest SHA-256 mismatch" >&2
  exit 2
fi
if [[ "$(wc -l < "${BE_DOCVQA_MANIFEST}")" -ne "${BE_DOCVQA_EXPECTED_STATES}" ]]; then
  echo "DocVQA manifest state count mismatch" >&2
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
mkdir -p "${BE_DOCVQA_RUN_DIR}"
"${python_bin}" -m beyond_entropy collect-qwen \
  --manifest "${BE_DOCVQA_MANIFEST}" \
  --expected-manifest-sha256 "${BE_DOCVQA_MANIFEST_SHA256}" \
  --output "${rollouts}" \
  --resume \
  --checkpoint-interval 32 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --scorer docvqa \
  --candidate-count 4 \
  --proposer ug-grid \
  --visual-crop-ratio 2.0 \
  --visual-cost 1.0 \
  --generation-seeds 0 \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 20260829 \
  --scientific-status "${scientific_status}" \
  --max-new-tokens 32 \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --attention-implementation sdpa \
  --system-prompt "You are a helpful assistant."
