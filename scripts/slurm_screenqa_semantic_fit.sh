#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=be-screenqa-semantic-fit
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-semantic-fit-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_SCREENQA_RANKER_ROLLOUTS BE_SCREENQA_RANKER_ROLLOUTS_SHA256
  BE_SCREENQA_SEMANTIC_FEATURES BE_SCREENQA_SEMANTIC_FEATURES_SHA256
  BE_SCREENQA_LABEL_FREE_AUDIT BE_SCREENQA_LABEL_FREE_AUDIT_SHA256
  BE_SCREENQA_FEATURE_BUNDLE BE_SCREENQA_FEATURE_BUNDLE_SHA256
  BE_SCREENQA_SEMANTIC_ACTIVATION BE_SCREENQA_SEMANTIC_ACTIVATION_SHA256
  BE_SCREENQA_V2_PROTOCOL BE_SCREENQA_V2_PROTOCOL_SHA256
  BE_SCREENQA_SEMANTIC_FIT_DIR BE_SCREENQA_SEMANTIC_CANDIDATE_DIR
  BE_SCREENQA_FEATURE_CODE_REVISION BE_SCREENQA_FIT_CODE_REVISION
  BE_SCREENQA_SEMANTIC_FIT_WORKER_SHA256 BE_SCREENQA_FEATURE_JOB_ID
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
worker="${repo_dir}/scripts/slurm_screenqa_semantic_fit.sh"
fit_audit="$(dirname "${BE_SCREENQA_SEMANTIC_FIT_DIR}")/semantic-fit.audit.json"
calibration_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/calibration-manifest-v1"
formal_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/formal-manifest-v1"
reserve_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/reserve-manifest-v1"
untouched_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/untouched-manifest-v1"

check_hash() {
  local path=$1 expected=$2 name=$3 actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ScreenQA semantic fit ${name} SHA-256 mismatch" >&2
    exit 2
  fi
}
check_hash "${worker}" "${BE_SCREENQA_SEMANTIC_FIT_WORKER_SHA256}" "worker"
check_hash "${BE_SCREENQA_RANKER_ROLLOUTS}" "${BE_SCREENQA_RANKER_ROLLOUTS_SHA256}" "rollouts"
check_hash "${BE_SCREENQA_SEMANTIC_FEATURES}" "${BE_SCREENQA_SEMANTIC_FEATURES_SHA256}" "features"
check_hash "${BE_SCREENQA_LABEL_FREE_AUDIT}" "${BE_SCREENQA_LABEL_FREE_AUDIT_SHA256}" "label-free audit"
check_hash "${BE_SCREENQA_FEATURE_BUNDLE}" "${BE_SCREENQA_FEATURE_BUNDLE_SHA256}" "feature bundle"
check_hash "${BE_SCREENQA_SEMANTIC_ACTIVATION}" "${BE_SCREENQA_SEMANTIC_ACTIVATION_SHA256}" "activation"
check_hash "${BE_SCREENQA_V2_PROTOCOL}" "${BE_SCREENQA_V2_PROTOCOL_SHA256}" "v2 protocol"
if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_SCREENQA_FIT_CODE_REVISION}" ]]; then
  echo "ScreenQA semantic fit code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA semantic fit" >&2
  exit 2
fi
for protected in "${calibration_dir}" "${formal_dir}" "${reserve_dir}" "${untouched_dir}"; do
  if [[ -e "${protected}" && ( ! -d "${protected}" || -n "$(find "${protected}" -mindepth 1 -print -quit)" ) ]]; then
    echo "ScreenQA protected role was opened before semantic fit: ${protected}" >&2
    exit 2
  fi
done
if [[ -e "${BE_SCREENQA_SEMANTIC_FIT_DIR}" || -e "${BE_SCREENQA_SEMANTIC_CANDIDATE_DIR}" || -e "${fit_audit}" ]]; then
  echo "ScreenQA semantic fit output already exists" >&2
  exit 2
fi
(
  cd "$(dirname "${BE_SCREENQA_FEATURE_BUNDLE}")"
  sha256sum --check "$(basename "${BE_SCREENQA_FEATURE_BUNDLE}")"
)
if [[ "$(jq -r '.decisions // 0' "${BE_SCREENQA_LABEL_FREE_AUDIT}")" -ne 14511 \
  || "$(jq -r '.outcomes_included_metadata' "${BE_SCREENQA_LABEL_FREE_AUDIT}")" != false \
  || "$(jq -r '.outcome_fields_present | length' "${BE_SCREENQA_LABEL_FREE_AUDIT}")" -ne 0 \
  || "$(jq -r '.features_sha256 // ""' "${BE_SCREENQA_LABEL_FREE_AUDIT}")" != "${BE_SCREENQA_SEMANTIC_FEATURES_SHA256}" \
  || "$(jq -r '.rollouts_sha256 // ""' "${BE_SCREENQA_LABEL_FREE_AUDIT}")" != "${BE_SCREENQA_RANKER_ROLLOUTS_SHA256}" ]]; then
  echo "ScreenQA semantic fit label-free audit contract failed" >&2
  exit 2
fi
if [[ "$(jq -r '.semantic_code_revision // ""' "${BE_SCREENQA_SEMANTIC_ACTIVATION}")" != "${BE_SCREENQA_FEATURE_CODE_REVISION}" \
  || "$(jq -r '.semantic_escalation_activated' "${BE_SCREENQA_SEMANTIC_ACTIVATION}")" != true \
  || "$(jq -r '.calibration_outcomes_opened' "${BE_SCREENQA_SEMANTIC_ACTIVATION}")" != false \
  || "$(jq -r '.formal_outcomes_opened' "${BE_SCREENQA_SEMANTIC_ACTIVATION}")" != false ]]; then
  echo "ScreenQA semantic fit activation contract failed" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
export BE_CODE_REVISION="${BE_SCREENQA_FIT_CODE_REVISION}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${repo_dir}"
"${python_bin}" scripts/fit_multidomain_action_value.py \
  --domain "screenqa=${BE_SCREENQA_RANKER_ROLLOUTS}" \
  --features "screenqa=${BE_SCREENQA_SEMANTIC_FEATURES}" \
  --model-family factorized-oof \
  --feature-mode hybrid-context-semantic \
  --oof-folds 5 \
  --bootstrap-resamples 2000 \
  --lambda-cost 0.05 \
  --alpha 0.1 \
  --alpha 1.0 \
  --alpha 10.0 \
  --alpha 100.0 \
  --alpha 1000.0 \
  --seed 20260831 \
  --output-dir "${BE_SCREENQA_SEMANTIC_FIT_DIR}"

"${python_bin}" scripts/freeze_screenqa_semantic_candidate.py \
  --report "${BE_SCREENQA_SEMANTIC_FIT_DIR}/report.json" \
  --model "${BE_SCREENQA_SEMANTIC_FIT_DIR}/model.json" \
  --features "${BE_SCREENQA_SEMANTIC_FEATURES}" \
  --expected-features-sha256 "${BE_SCREENQA_SEMANTIC_FEATURES_SHA256}" \
  --label-free-audit "${BE_SCREENQA_LABEL_FREE_AUDIT}" \
  --expected-label-free-audit-sha256 "${BE_SCREENQA_LABEL_FREE_AUDIT_SHA256}" \
  --activation "${BE_SCREENQA_SEMANTIC_ACTIVATION}" \
  --expected-activation-sha256 "${BE_SCREENQA_SEMANTIC_ACTIVATION_SHA256}" \
  --protocol "${BE_SCREENQA_V2_PROTOCOL}" \
  --expected-protocol-sha256 "${BE_SCREENQA_V2_PROTOCOL_SHA256}" \
  --expected-rollouts-sha256 "${BE_SCREENQA_RANKER_ROLLOUTS_SHA256}" \
  --output-dir "${BE_SCREENQA_SEMANTIC_CANDIDATE_DIR}"
(
  cd "${BE_SCREENQA_SEMANTIC_CANDIDATE_DIR}"
  sha256sum --check SHA256SUMS
)

"${python_bin}" - \
  "${fit_audit}" \
  "${BE_SCREENQA_SEMANTIC_FIT_DIR}/model.json" \
  "${BE_SCREENQA_SEMANTIC_FIT_DIR}/report.json" \
  "${BE_SCREENQA_SEMANTIC_CANDIDATE_DIR}/candidate.audit.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

output, model, report, candidate_path = map(Path, sys.argv[1:5])
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

payload = {
    "passed": True,
    "scientific_status": "sole frozen v2 semantic candidate fit after both v1 low-capacity candidates failed; no protected outcomes opened",
    "semantic_feature_job_id": os.environ["BE_SCREENQA_FEATURE_JOB_ID"],
    "semantic_fit_job_id": os.environ["SLURM_JOB_ID"],
    "feature_code_revision": os.environ["BE_SCREENQA_FEATURE_CODE_REVISION"],
    "fit_code_revision": os.environ["BE_SCREENQA_FIT_CODE_REVISION"],
    "activation_sha256": os.environ["BE_SCREENQA_SEMANTIC_ACTIVATION_SHA256"],
    "features_sha256": os.environ["BE_SCREENQA_SEMANTIC_FEATURES_SHA256"],
    "label_free_audit_sha256": os.environ["BE_SCREENQA_LABEL_FREE_AUDIT_SHA256"],
    "model_sha256": sha256_file(model),
    "report_sha256": sha256_file(report),
    "candidate_audit_sha256": sha256_file(candidate_path),
    "candidate_frozen": candidate.get("candidate_frozen"),
    "ranker_development_stopped": candidate.get("ranker_development_stopped"),
    "calibration_outcomes_opened": False,
    "formal_outcomes_opened": False,
    "reserve_outcomes_opened": False,
}
temporary = output.with_name(output.name + ".tmp")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
    handle.write("\n")
temporary.replace(output)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
sha256sum "${fit_audit}"
