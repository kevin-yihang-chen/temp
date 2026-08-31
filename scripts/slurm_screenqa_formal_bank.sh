#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --job-name=be-screenqa-formal-bank
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-formal-bank-%A-%a.out
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
: "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
audit_python=/userhome/cs3/yihangc/anaconda3/bin/python
manifest="${BE_SCREENQA_FORMAL_MANIFEST_DIR}/manifest.jsonl"
shard_index="${SLURM_ARRAY_TASK_ID}"
shard_name=$(printf 'shard-%05d-of-%05d' "${shard_index}" "${BE_SCREENQA_SHARD_COUNT}")
shard_dir="${BE_SCREENQA_FORMAL_RUN_ROOT}/${shard_name}"
rollouts="${shard_dir}/rollouts.jsonl"

if [[ "${BE_SCREENQA_FORMAL_EXPECTED_STATES}" -ne 14672 ]]; then
  echo "ScreenQA formal collection is frozen to 14672 states" >&2
  exit 2
fi
if [[ "${BE_SCREENQA_SHARD_COUNT}" -ne 4 ]]; then
  echo "ScreenQA formal collection is frozen to four shards" >&2
  exit 2
fi
if [[ "${shard_index}" -lt 0 || "${shard_index}" -ge "${BE_SCREENQA_SHARD_COUNT}" ]]; then
  echo "ScreenQA formal shard index is out of range" >&2
  exit 2
fi
tracked_status=$(git -C "${repo_dir}" status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before ScreenQA formal collection" >&2
  exit 2
fi
actual_code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_code_revision}" != "${BE_SCREENQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "ScreenQA formal collection code revision mismatch" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
"${audit_python}" -m scripts.verify_screenqa_formal_manifest \
  --manifest-dir "${BE_SCREENQA_FORMAL_MANIFEST_DIR}" \
  --candidate-dir "${BE_SCREENQA_CANDIDATE_DIR}" \
  --expected-candidate-bundle-sha256 "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}" \
  --calibration-dir "${BE_SCREENQA_CALIBRATION_DIR}" \
  --expected-calibration-bundle-sha256 "${BE_SCREENQA_CALIBRATION_BUNDLE_SHA256}" \
  --expected-manifest-sha256 "${BE_SCREENQA_FORMAL_MANIFEST_SHA256}" \
  --expected-audit-sha256 "${BE_SCREENQA_FORMAL_MANIFEST_AUDIT_SHA256}"

ledger_args=(
  --shard-dir "${shard_dir}"
  --manifest "${manifest}"
  --expected-manifest-sha256 "${BE_SCREENQA_FORMAL_MANIFEST_SHA256}"
  --manifest-audit-sha256 "${BE_SCREENQA_FORMAL_MANIFEST_AUDIT_SHA256}"
  --candidate-bundle-sha256 "${BE_SCREENQA_CANDIDATE_BUNDLE_SHA256}"
  --calibration-bundle-sha256 "${BE_SCREENQA_CALIBRATION_BUNDLE_SHA256}"
  --code-revision "${BE_SCREENQA_EXPECTED_CODE_REVISION}"
  --shard-index "${shard_index}"
  --shard-count "${BE_SCREENQA_SHARD_COUNT}"
  --expected-total-states "${BE_SCREENQA_FORMAL_EXPECTED_STATES}"
)
if [[ -e "${shard_dir}/formal-complete.json" ]]; then
  "${audit_python}" -m scripts.screenqa_formal_one_shot verify "${ledger_args[@]}"
  echo "ScreenQA formal shard was already completed under the exact contract; no generation performed"
  exit 0
fi
"${audit_python}" -m scripts.screenqa_formal_one_shot open "${ledger_args[@]}"

export BE_CODE_REVISION="${actual_code_revision}"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

collect_args=(
  -m beyond_entropy collect-qwen
  --manifest "${manifest}"
  --expected-manifest-sha256 "${BE_SCREENQA_FORMAL_MANIFEST_SHA256}"
  --output "${rollouts}"
  --resume
  --checkpoint-interval 32
  --shard-count "${BE_SCREENQA_SHARD_COUNT}"
  --shard-index "${shard_index}"
  --model Qwen/Qwen2.5-VL-3B-Instruct
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3
  --scorer screenqa
  --candidate-count 4
  --proposer ug-grid
  --visual-crop-ratio 2.0
  --visual-cost 1.0
  --generation-seeds 0
  --bootstrap-resamples 500
  --bootstrap-seed 20260831
  --scientific-status "ScreenQA one-shot formal sibling bank; frozen calibrated policy and implementation; no target-derived tuning"
  --max-new-tokens 32
  --min-pixels 200704
  --max-pixels 602112
  --attention-implementation sdpa
  --system-prompt "You are a helpful assistant."
)

"${python_bin}" "${collect_args[@]}"
first_sha=$(sha256sum "${rollouts}")
first_sha=${first_sha%% *}
if [[ ! -e "${shard_dir}/rollouts.first-pass.provenance.json" ]]; then
  cp "${rollouts%.jsonl}.provenance.json" \
    "${shard_dir}/rollouts.first-pass.provenance.json"
fi

# This second invocation is a no-generation proof: every state is already present.
"${python_bin}" "${collect_args[@]}"
second_sha=$(sha256sum "${rollouts}")
second_sha=${second_sha%% *}
if [[ "${first_sha}" != "${second_sha}" ]]; then
  echo "ScreenQA exact-contract resume changed completed formal rollout bytes" >&2
  exit 2
fi

"${audit_python}" - "${rollouts}" "${first_sha}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

rollouts = Path(sys.argv[1])
expected_sha256 = sys.argv[2]
provenance = json.loads(rollouts.with_suffix(".provenance.json").read_text())
records = sum(1 for line in rollouts.open(encoding="utf-8") if line.strip())
actual_sha256 = hashlib.sha256(rollouts.read_bytes()).hexdigest()
passed = (
    actual_sha256 == expected_sha256
    and provenance["output_sha256"] == actual_sha256
    and provenance["resumed_from_records"] == records
    and provenance["completed_examples"] == provenance["examples"]
)
audit = {
    "passed": passed,
    "rollouts_sha256_before_resume": expected_sha256,
    "rollouts_sha256_after_resume": actual_sha256,
    "records": records,
    "examples": provenance["examples"],
    "resumed_from_records": provenance["resumed_from_records"],
}
serialized = json.dumps(audit, indent=2, sort_keys=True) + "\n"
path = rollouts.parent / "resume.audit.json"
if path.exists():
    if path.read_text(encoding="utf-8") != serialized:
        raise SystemExit("existing ScreenQA formal resume audit changed")
else:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
if not passed:
    raise SystemExit("ScreenQA formal exact-contract resume audit failed")
print(serialized, end="")
PY

"${audit_python}" -m scripts.screenqa_formal_one_shot complete "${ledger_args[@]}"
