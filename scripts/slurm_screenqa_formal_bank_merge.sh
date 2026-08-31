#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=be-screenqa-formal-merge
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-formal-bank-merge-%j.out
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
: "${BE_SCREENQA_FORMAL_EXPECTED_STATES:?missing BE_SCREENQA_FORMAL_EXPECTED_STATES}"
: "${BE_SCREENQA_FORMAL_RUN_ROOT:?missing BE_SCREENQA_FORMAL_RUN_ROOT}"
: "${BE_SCREENQA_EXPECTED_CODE_REVISION:?missing BE_SCREENQA_EXPECTED_CODE_REVISION}"
: "${BE_SCREENQA_SHARD_COUNT:?missing BE_SCREENQA_SHARD_COUNT}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/bin/python
manifest="${BE_SCREENQA_FORMAL_MANIFEST_DIR}/manifest.jsonl"
output_dir="${BE_SCREENQA_FORMAL_RUN_ROOT}/merged"
output="${output_dir}/rollouts.jsonl"
diagnostic="${output_dir}/rollouts.diagnostic.json"
merge_audit="${output_dir}/rollouts.merge.json"
rollout_audit="${output_dir}/rollouts.audit.json"
completion="${output_dir}/formal-bank.complete.json"

tracked_status=$(git -C "${repo_dir}" status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before ScreenQA formal merge" >&2
  exit 2
fi
actual_code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_code_revision}" != "${BE_SCREENQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "ScreenQA formal merge code revision mismatch" >&2
  exit 2
fi
if [[ "${BE_SCREENQA_FORMAL_EXPECTED_STATES}" -ne 14672 ]]; then
  echo "ScreenQA formal merge is frozen to 14672 states" >&2
  exit 2
fi
if [[ "${BE_SCREENQA_SHARD_COUNT}" -ne 4 ]]; then
  echo "ScreenQA formal merge is frozen to four shards" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
"${python_bin}" -m scripts.verify_screenqa_formal_manifest \
  --manifest-dir "${BE_SCREENQA_FORMAL_MANIFEST_DIR}" \
  --candidate-dir "${BE_SCREENQA_CANDIDATE_DIR}" \
  --expected-candidate-bundle-sha256 "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}" \
  --calibration-dir "${BE_SCREENQA_CALIBRATION_DIR}" \
  --expected-calibration-bundle-sha256 "${BE_SCREENQA_CALIBRATION_BUNDLE_SHA256}" \
  --expected-manifest-sha256 "${BE_SCREENQA_FORMAL_MANIFEST_SHA256}" \
  --expected-audit-sha256 "${BE_SCREENQA_FORMAL_MANIFEST_AUDIT_SHA256}"

for shard_index in 0 1 2 3; do
  shard_name=$(printf 'shard-%05d-of-00004' "${shard_index}")
  "${python_bin}" -m scripts.screenqa_formal_one_shot verify \
    --shard-dir "${BE_SCREENQA_FORMAL_RUN_ROOT}/${shard_name}" \
    --manifest "${manifest}" \
    --expected-manifest-sha256 "${BE_SCREENQA_FORMAL_MANIFEST_SHA256}" \
    --manifest-audit-sha256 "${BE_SCREENQA_FORMAL_MANIFEST_AUDIT_SHA256}" \
    --candidate-bundle-sha256 "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}" \
    --calibration-bundle-sha256 "${BE_SCREENQA_CALIBRATION_BUNDLE_SHA256}" \
    --code-revision "${BE_SCREENQA_EXPECTED_CODE_REVISION}" \
    --shard-index "${shard_index}" \
    --shard-count "${BE_SCREENQA_SHARD_COUNT}" \
    --expected-total-states "${BE_SCREENQA_FORMAL_EXPECTED_STATES}"
done

if [[ -e "${completion}" ]]; then
  if [[ ! -s "${output_dir}/SHA256SUMS" ]]; then
    cd "${output_dir}"
    sha256sum rollouts.jsonl rollouts.diagnostic.json rollouts.merge.json \
      rollouts.audit.json formal-bank.complete.json > SHA256SUMS
  fi
  cd "${output_dir}"
  sha256sum --check SHA256SUMS
  echo "ScreenQA one-shot formal bank was already finalized; no merge or generation performed"
  exit 0
fi

existing_derived=0
for path in "${output}" "${diagnostic}" "${merge_audit}"; do
  if [[ -e "${path}" ]]; then
    existing_derived=$((existing_derived + 1))
  fi
done
if [[ "${existing_derived}" -eq 0 ]]; then
  "${python_bin}" "${repo_dir}/scripts/merge_qwen_rollout_shards.py" \
    --manifest "${manifest}" \
    --expected-manifest-sha256 "${BE_SCREENQA_FORMAL_MANIFEST_SHA256}" \
    --run-root "${BE_SCREENQA_FORMAL_RUN_ROOT}" \
    --shard-count "${BE_SCREENQA_SHARD_COUNT}" \
    --expected-code-revision "${BE_SCREENQA_EXPECTED_CODE_REVISION}" \
    --expected-scorer screenqa \
    --require-resume-audit \
    --bootstrap-resamples 5000 \
    --bootstrap-seed 20260831 \
    --output "${output}"
elif [[ "${existing_derived}" -ne 3 ]]; then
  echo "incomplete derived ScreenQA formal merge exists; raw one-shot shards remain intact" >&2
  exit 2
fi

rollouts_sha256=$(sha256sum "${output}")
rollouts_sha256=${rollouts_sha256%% *}
merge_audit_sha256=$(sha256sum "${merge_audit}")
merge_audit_sha256=${merge_audit_sha256%% *}
"${python_bin}" -m scripts.verify_screenqa_formal_rollouts \
  --formal-manifest-dir "${BE_SCREENQA_FORMAL_MANIFEST_DIR}" \
  --candidate-dir "${BE_SCREENQA_CANDIDATE_DIR}" \
  --expected-candidate-bundle-sha256 "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}" \
  --calibration-dir "${BE_SCREENQA_CALIBRATION_DIR}" \
  --expected-calibration-bundle-sha256 "${BE_SCREENQA_CALIBRATION_BUNDLE_SHA256}" \
  --expected-manifest-sha256 "${BE_SCREENQA_FORMAL_MANIFEST_SHA256}" \
  --expected-manifest-audit-sha256 "${BE_SCREENQA_FORMAL_MANIFEST_AUDIT_SHA256}" \
  --run-root "${BE_SCREENQA_FORMAL_RUN_ROOT}" \
  --rollouts "${output}" \
  --expected-rollouts-sha256 "${rollouts_sha256}" \
  --merge-audit "${merge_audit}" \
  --expected-merge-audit-sha256 "${merge_audit_sha256}" \
  --expected-bank-code-revision "${BE_SCREENQA_EXPECTED_CODE_REVISION}" \
  --output "${rollout_audit}" \
  --resume

"${python_bin}" - "${completion}" "${output}" "${diagnostic}" \
  "${merge_audit}" "${rollout_audit}" \
  "${BE_SCREENQA_FORMAL_MANIFEST_SHA256}" \
  "${BE_SCREENQA_FORMAL_MANIFEST_AUDIT_SHA256}" \
  "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}" \
  "${BE_SCREENQA_CALIBRATION_BUNDLE_SHA256}" \
  "${BE_SCREENQA_EXPECTED_CODE_REVISION}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


completion, rollouts, diagnostic, merge, audit = map(Path, sys.argv[1:6])
payload = {
    "schema_version": 1,
    "passed": True,
    "scientific_status": (
        "ScreenQA one-shot formal sibling bank; frozen calibrated policy and "
        "implementation; no target-derived tuning"
    ),
    "one_shot_formal_bank_complete": True,
    "formal_outcomes_used_for_tuning": False,
    "states": 14672,
    "records": 73360,
    "completed_shards": 4,
    "rollouts_sha256": sha(rollouts),
    "diagnostic_sha256": sha(diagnostic),
    "merge_audit_sha256": sha(merge),
    "rollout_audit_sha256": sha(audit),
    "manifest_sha256": sys.argv[6],
    "manifest_audit_sha256": sys.argv[7],
    "candidate_bundle_sha256": sys.argv[8],
    "calibration_bundle_sha256": sys.argv[9],
    "code_revision": sys.argv[10],
}
serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
if completion.exists():
    if completion.read_text(encoding="utf-8") != serialized:
        raise SystemExit("existing ScreenQA formal bank completion marker changed")
else:
    completion.parent.mkdir(parents=True, exist_ok=True)
    with completion.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
print(serialized, end="")
PY

cd "${output_dir}"
sha256sum rollouts.jsonl rollouts.diagnostic.json rollouts.merge.json \
  rollouts.audit.json formal-bank.complete.json > SHA256SUMS
sha256sum --check SHA256SUMS
