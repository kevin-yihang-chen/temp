#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=be-screenqa-formal-eval
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-formal-evaluate-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SCREENQA_FORMAL_MANIFEST_DIR:?missing BE_SCREENQA_FORMAL_MANIFEST_DIR}"
: "${BE_SCREENQA_FORMAL_MANIFEST_SHA256:?missing BE_SCREENQA_FORMAL_MANIFEST_SHA256}"
: "${BE_SCREENQA_FORMAL_MANIFEST_AUDIT_SHA256:?missing BE_SCREENQA_FORMAL_MANIFEST_AUDIT_SHA256}"
: "${BE_SCREENQA_CANDIDATE_DIR:?missing BE_SCREENQA_CANDIDATE_DIR}"
: "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256:?missing BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}"
: "${BE_SCREENQA_CALIBRATION_DIR:?missing BE_SCREENQA_CALIBRATION_DIR}"
: "${BE_SCREENQA_CALIBRATION_BUNDLE_SHA256:?missing BE_SCREENQA_CALIBRATION_BUNDLE_SHA256}"
: "${BE_SCREENQA_FORMAL_RUN_ROOT:?missing BE_SCREENQA_FORMAL_RUN_ROOT}"
: "${BE_SCREENQA_FORMAL_EVALUATION_DIR:?missing BE_SCREENQA_FORMAL_EVALUATION_DIR}"
: "${BE_SCREENQA_EXPECTED_CODE_REVISION:?missing BE_SCREENQA_EXPECTED_CODE_REVISION}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/bin/python
merged="${BE_SCREENQA_FORMAL_RUN_ROOT}/merged"
rollouts="${merged}/rollouts.jsonl"
merge_audit="${merged}/rollouts.merge.json"
rollout_audit="${merged}/rollouts.audit.json"
bank_completion="${merged}/formal-bank.complete.json"
report="${BE_SCREENQA_FORMAL_EVALUATION_DIR}/report.json"
rendered="${BE_SCREENQA_FORMAL_EVALUATION_DIR}/report.md"
result_completion="${BE_SCREENQA_FORMAL_EVALUATION_DIR}/formal-result.complete.json"

if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA formal evaluation" >&2
  exit 2
fi
if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_SCREENQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "ScreenQA formal evaluator code revision mismatch" >&2
  exit 2
fi
if [[ -e "${rendered}" && ! -e "${report}" ]]; then
  echo "ScreenQA rendered formal result exists without canonical JSON" >&2
  exit 2
fi
for path in "${rollouts}" "${merge_audit}" "${rollout_audit}" "${bank_completion}" "${merged}/SHA256SUMS"; do
  if [[ ! -s "${path}" ]]; then
    echo "missing completed ScreenQA formal-bank input: ${path}" >&2
    exit 2
  fi
done
cd "${merged}"
sha256sum --check SHA256SUMS
rollouts_sha256=$(sha256sum "${rollouts}")
rollouts_sha256=${rollouts_sha256%% *}
merge_audit_sha256=$(sha256sum "${merge_audit}")
merge_audit_sha256=${merge_audit_sha256%% *}
rollout_audit_sha256=$(sha256sum "${rollout_audit}")
rollout_audit_sha256=${rollout_audit_sha256%% *}
bank_completion_sha256=$(sha256sum "${bank_completion}")
bank_completion_sha256=${bank_completion_sha256%% *}

if [[ -e "${result_completion}" ]]; then
  if [[ ! -s "${BE_SCREENQA_FORMAL_EVALUATION_DIR}/SHA256SUMS" ]]; then
    cd "${BE_SCREENQA_FORMAL_EVALUATION_DIR}"
    sha256sum report.json report.md rollouts.reverified.json \
      formal-result.complete.json > SHA256SUMS
  fi
  cd "${BE_SCREENQA_FORMAL_EVALUATION_DIR}"
  sha256sum --check SHA256SUMS
  echo "ScreenQA one-shot formal evaluation was already finalized; no reevaluation performed"
  exit 0
fi

export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${repo_dir}"
if [[ ! -e "${report}" ]]; then
  "${python_bin}" -m scripts.evaluate_screenqa_formal \
    --candidate-dir "${BE_SCREENQA_CANDIDATE_DIR}" \
    --expected-candidate-bundle-sha256 "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}" \
    --calibration-dir "${BE_SCREENQA_CALIBRATION_DIR}" \
    --expected-calibration-bundle-sha256 "${BE_SCREENQA_CALIBRATION_BUNDLE_SHA256}" \
    --formal-manifest-dir "${BE_SCREENQA_FORMAL_MANIFEST_DIR}" \
    --expected-manifest-sha256 "${BE_SCREENQA_FORMAL_MANIFEST_SHA256}" \
    --expected-manifest-audit-sha256 "${BE_SCREENQA_FORMAL_MANIFEST_AUDIT_SHA256}" \
    --formal-run-root "${BE_SCREENQA_FORMAL_RUN_ROOT}" \
    --rollouts "${rollouts}" \
    --expected-rollouts-sha256 "${rollouts_sha256}" \
    --merge-audit "${merge_audit}" \
    --expected-merge-audit-sha256 "${merge_audit_sha256}" \
    --rollout-audit "${rollout_audit}" \
    --expected-rollout-audit-sha256 "${rollout_audit_sha256}" \
    --bank-completion "${bank_completion}" \
    --expected-bank-completion-sha256 "${bank_completion_sha256}" \
    --expected-code-revision "${BE_SCREENQA_EXPECTED_CODE_REVISION}" \
    --output "${report}" \
    --bootstrap-resamples 20000 \
    --bootstrap-confidence 0.975 \
    --bootstrap-seed 20260831
fi

report_sha256=$(sha256sum "${report}")
report_sha256=${report_sha256%% *}
"${python_bin}" - "${report}" "${BE_SCREENQA_EXPECTED_CODE_REVISION}" \
  "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}" \
  "${BE_SCREENQA_CALIBRATION_BUNDLE_SHA256}" \
  "${BE_SCREENQA_FORMAL_MANIFEST_SHA256}" "${rollouts_sha256}" \
  "${bank_completion_sha256}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
run = report.get("run")
expected = {
    "code_revision": sys.argv[2],
    "candidate_bundle_sha256": sys.argv[3],
    "calibration_bundle_sha256": sys.argv[4],
    "manifest_sha256": sys.argv[5],
    "rollouts_sha256": sys.argv[6],
    "bank_completion_sha256": sys.argv[7],
    "no_target_derived_tuning": True,
    "formal_outcomes_used": True,
    "bootstrap_resamples": 20000,
    "bootstrap_confidence": 0.975,
    "bootstrap_seed": 20260831,
}
if not isinstance(run, dict) or any(run.get(k) != v for k, v in expected.items()):
    raise SystemExit("existing ScreenQA formal report is not bound to frozen inputs")
if report.get("n_sources") != 1471 or report.get("n_decisions") != 14672:
    raise SystemExit("existing ScreenQA formal report population changed")
PY
if [[ ! -e "${rendered}" ]]; then
  "${python_bin}" -m scripts.render_screenqa_formal \
    --report "${report}" \
    --expected-report-sha256 "${report_sha256}" \
    --output "${rendered}"
fi

"${python_bin}" - "${result_completion}" "${report}" "${rendered}" \
  "${BE_SCREENQA_FORMAL_EVALUATION_DIR}/rollouts.reverified.json" \
  "${BE_SCREENQA_EXPECTED_CODE_REVISION}" \
  "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}" \
  "${BE_SCREENQA_CALIBRATION_BUNDLE_SHA256}" \
  "${BE_SCREENQA_FORMAL_MANIFEST_SHA256}" "${rollouts_sha256}" \
  "${bank_completion_sha256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


completion, report, rendered, reaudit = map(Path, sys.argv[1:5])
payload = {
    "schema_version": 1,
    "passed": True,
    "one_shot_formal_evaluation_complete": True,
    "formal_outcomes_used_for_tuning": False,
    "report_sha256": sha(report),
    "rendered_report_sha256": sha(rendered),
    "fresh_rollout_audit_sha256": sha(reaudit),
    "code_revision": sys.argv[5],
    "candidate_bundle_sha256": sys.argv[6],
    "calibration_bundle_sha256": sys.argv[7],
    "manifest_sha256": sys.argv[8],
    "rollouts_sha256": sys.argv[9],
    "bank_completion_sha256": sys.argv[10],
}
serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
if completion.exists():
    if completion.read_text(encoding="utf-8") != serialized:
        raise SystemExit("existing ScreenQA formal result completion marker changed")
else:
    with completion.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
print(serialized, end="")
PY
cd "${BE_SCREENQA_FORMAL_EVALUATION_DIR}"
sha256sum report.json report.md rollouts.reverified.json \
  formal-result.complete.json > SHA256SUMS
sha256sum --check SHA256SUMS
