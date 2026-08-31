#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=be-screenqa-fit-recover
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-ranker-fit-recovery-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SCREENQA_RANKER_ROLLOUTS:?missing BE_SCREENQA_RANKER_ROLLOUTS}"
: "${BE_SCREENQA_RANKER_ROLLOUTS_SHA256:?missing BE_SCREENQA_RANKER_ROLLOUTS_SHA256}"
: "${BE_SCREENQA_PROTOCOL:?missing BE_SCREENQA_PROTOCOL}"
: "${BE_SCREENQA_PROTOCOL_SHA256:?missing BE_SCREENQA_PROTOCOL_SHA256}"
: "${BE_SCREENQA_FIT_ROOT:?missing BE_SCREENQA_FIT_ROOT}"
: "${BE_SCREENQA_EXPECTED_CODE_REVISION:?missing BE_SCREENQA_EXPECTED_CODE_REVISION}"
: "${BE_SCREENQA_CONTEXT_MODEL_SHA256:?missing BE_SCREENQA_CONTEXT_MODEL_SHA256}"
: "${BE_SCREENQA_CONTEXT_REPORT_SHA256:?missing BE_SCREENQA_CONTEXT_REPORT_SHA256}"
: "${BE_SCREENQA_INPUT_AUDIT_SHA256:?missing BE_SCREENQA_INPUT_AUDIT_SHA256}"
: "${BE_SCREENQA_RECOVERY_WORKER_SHA256:?missing BE_SCREENQA_RECOVERY_WORKER_SHA256}"
: "${BE_SCREENQA_PREVIOUS_JOB_ID:?missing BE_SCREENQA_PREVIOUS_JOB_ID}"
: "${BE_SCREENQA_PREVIOUS_JOB_STATE:?missing BE_SCREENQA_PREVIOUS_JOB_STATE}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
worker_path="${repo_dir}/scripts/slurm_screenqa_ranker_fit_recovery.sh"
context_dir="${BE_SCREENQA_FIT_ROOT}/context-geometry-oof-v1"
spatial_dir="${BE_SCREENQA_FIT_ROOT}/spatial-context-geometry-oof-v1"
candidate_dir="${BE_SCREENQA_FIT_ROOT}/candidate-v1"
input_audit="${BE_SCREENQA_FIT_ROOT}/ranker-rollouts.audit.json"
recovery_audit="${BE_SCREENQA_FIT_ROOT}/ranker-fit-recovery.audit.json"
context_model="${context_dir}/model.json"
context_report="${context_dir}/report.json"

if [[ "${BE_SCREENQA_PREVIOUS_JOB_ID}" != "196911" \
  || "${BE_SCREENQA_PREVIOUS_JOB_STATE}" != "TIMEOUT" ]]; then
  echo "ScreenQA recovery predecessor contract mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked worktree must be clean before ScreenQA ranker recovery" >&2
  exit 2
fi
actual_code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_code_revision}" != "${BE_SCREENQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "ScreenQA ranker recovery code revision mismatch" >&2
  exit 2
fi
actual_worker_sha256=$(sha256sum "${worker_path}")
actual_worker_sha256=${actual_worker_sha256%% *}
if [[ "${actual_worker_sha256}" != "${BE_SCREENQA_RECOVERY_WORKER_SHA256}" ]]; then
  echo "ScreenQA ranker recovery worker SHA-256 mismatch" >&2
  exit 2
fi
for required_file in \
  "${BE_SCREENQA_RANKER_ROLLOUTS}" \
  "${BE_SCREENQA_PROTOCOL}" \
  "${input_audit}" \
  "${context_model}" \
  "${context_report}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "missing ScreenQA recovery input: ${required_file}" >&2
    exit 2
  fi
done
if [[ -e "${spatial_dir}" || -e "${candidate_dir}" || -e "${recovery_audit}" ]]; then
  echo "ScreenQA recovery output must not already exist" >&2
  exit 2
fi

check_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ScreenQA recovery SHA-256 mismatch: ${path}" >&2
    exit 2
  fi
}

check_sha256 "${BE_SCREENQA_RANKER_ROLLOUTS}" "${BE_SCREENQA_RANKER_ROLLOUTS_SHA256}"
check_sha256 "${BE_SCREENQA_PROTOCOL}" "${BE_SCREENQA_PROTOCOL_SHA256}"
check_sha256 "${input_audit}" "${BE_SCREENQA_INPUT_AUDIT_SHA256}"
check_sha256 "${context_model}" "${BE_SCREENQA_CONTEXT_MODEL_SHA256}"
check_sha256 "${context_report}" "${BE_SCREENQA_CONTEXT_REPORT_SHA256}"

export PYTHONPATH="${repo_dir}/src"
export BE_CODE_REVISION="${BE_SCREENQA_EXPECTED_CODE_REVISION}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${repo_dir}"
"${python_bin}" - \
  "${input_audit}" \
  "${context_model}" \
  "${context_report}" \
  "${BE_SCREENQA_EXPECTED_CODE_REVISION}" \
  "${BE_SCREENQA_RANKER_ROLLOUTS_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

input_audit_path, model_path, report_path = map(Path, sys.argv[1:4])
expected_revision, expected_rollouts_sha256 = sys.argv[4:6]
input_audit = json.loads(input_audit_path.read_text(encoding="utf-8"))
model = json.loads(model_path.read_text(encoding="utf-8"))
report = json.loads(report_path.read_text(encoding="utf-8"))
if (
    input_audit.get("passed") is not True
    or input_audit.get("rollouts_sha256") != expected_rollouts_sha256
    or input_audit.get("calibration_outcomes_opened") is not False
    or input_audit.get("formal_outcomes_opened") is not False
    or input_audit.get("reserve_outcomes_opened") is not False
):
    raise SystemExit("ScreenQA recovery input audit contract mismatch")
if (
    report.get("feature_mode") != "context-geometry"
    or model.get("feature_mode") != "context-geometry"
    or report.get("development_decisions") != 14511
    or report.get("run", {}).get("code_revision") != expected_revision
    or report.get("run", {}).get("formal_outcomes_used") is not False
    or report.get("run", {})
    .get("development_inputs", {})
    .get("screenqa", {})
    .get("sha256")
    != expected_rollouts_sha256
):
    raise SystemExit("ScreenQA preserved context model contract mismatch")
print("screenqa_preserved_context_contract=passed")
PY

"${python_bin}" scripts/fit_multidomain_action_value.py \
  --domain "screenqa=${BE_SCREENQA_RANKER_ROLLOUTS}" \
  --model-family factorized-oof \
  --oof-folds 5 \
  --bootstrap-resamples 2000 \
  --lambda-cost 0.05 \
  --alpha 0.1 \
  --alpha 1.0 \
  --alpha 10.0 \
  --alpha 100.0 \
  --alpha 1000.0 \
  --seed 20260831 \
  --feature-mode spatial-context-geometry \
  --output-dir "${spatial_dir}"

"${python_bin}" scripts/select_screenqa_ranker_candidate.py \
  --context-report "${context_report}" \
  --context-model "${context_model}" \
  --spatial-report "${spatial_dir}/report.json" \
  --spatial-model "${spatial_dir}/model.json" \
  --protocol "${BE_SCREENQA_PROTOCOL}" \
  --expected-protocol-sha256 "${BE_SCREENQA_PROTOCOL_SHA256}" \
  --output-dir "${candidate_dir}"

cd "${candidate_dir}"
sha256sum --check SHA256SUMS
cd "${repo_dir}"

"${python_bin}" - \
  "${recovery_audit}" \
  "${worker_path}" \
  "${context_model}" \
  "${context_report}" \
  "${spatial_dir}/model.json" \
  "${spatial_dir}/report.json" \
  "${candidate_dir}/candidate.audit.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


output, worker, context_model, context_report, spatial_model, spatial_report, candidate_audit = map(
    Path, sys.argv[1:8]
)
candidate = json.loads(candidate_audit.read_text(encoding="utf-8"))
payload = {
    "passed": True,
    "scientific_status": "registered spatial ranker recovery after scheduler timeout; preserved context output reused only after exact hash and provenance verification",
    "previous_job_id": os.environ["BE_SCREENQA_PREVIOUS_JOB_ID"],
    "previous_job_state": os.environ["BE_SCREENQA_PREVIOUS_JOB_STATE"],
    "recovery_job_id": os.environ["SLURM_JOB_ID"],
    "fit_code_revision": os.environ["BE_SCREENQA_EXPECTED_CODE_REVISION"],
    "ranker_rollouts_sha256": os.environ["BE_SCREENQA_RANKER_ROLLOUTS_SHA256"],
    "recovery_worker_sha256": sha256_file(worker),
    "context_model_sha256": sha256_file(context_model),
    "context_report_sha256": sha256_file(context_report),
    "spatial_model_sha256": sha256_file(spatial_model),
    "spatial_report_sha256": sha256_file(spatial_report),
    "candidate_audit_sha256": sha256_file(candidate_audit),
    "candidate_frozen": candidate.get("candidate_frozen"),
    "semantic_escalation_required": candidate.get("semantic_escalation_required"),
    "calibration_outcomes_opened": False,
    "formal_outcomes_opened": False,
    "reserve_outcomes_opened": False,
}
with output.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

sha256sum "${recovery_audit}"
