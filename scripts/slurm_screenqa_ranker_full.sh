#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --job-name=be-screenqa-ranker
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-ranker-full-%A-%a.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SCREENQA_MANIFEST_DIR:?missing BE_SCREENQA_MANIFEST_DIR}"
: "${BE_SCREENQA_MANIFEST_SHA256:?missing BE_SCREENQA_MANIFEST_SHA256}"
: "${BE_SCREENQA_MANIFEST_AUDIT_SHA256:?missing BE_SCREENQA_MANIFEST_AUDIT_SHA256}"
: "${BE_SCREENQA_EXPECTED_STATES:?missing BE_SCREENQA_EXPECTED_STATES}"
: "${BE_SCREENQA_RUN_ROOT:?missing BE_SCREENQA_RUN_ROOT}"
: "${BE_SCREENQA_EXPECTED_CODE_REVISION:?missing BE_SCREENQA_EXPECTED_CODE_REVISION}"
: "${BE_SCREENQA_SHARD_COUNT:?missing BE_SCREENQA_SHARD_COUNT}"
: "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
manifest="${BE_SCREENQA_MANIFEST_DIR}/manifest.jsonl"
shard_index="${SLURM_ARRAY_TASK_ID}"
shard_name=$(printf 'shard-%05d-of-%05d' "${shard_index}" "${BE_SCREENQA_SHARD_COUNT}")
shard_dir="${BE_SCREENQA_RUN_ROOT}/${shard_name}"
rollouts="${shard_dir}/rollouts.jsonl"

if [[ "${BE_SCREENQA_SHARD_COUNT}" -ne 4 ]]; then
  echo "ScreenQA ranker collection is frozen to four shards" >&2
  exit 2
fi
if [[ "${shard_index}" -lt 0 || "${shard_index}" -ge "${BE_SCREENQA_SHARD_COUNT}" ]]; then
  echo "ScreenQA ranker shard index is out of range" >&2
  exit 2
fi
tracked_status=$(git -C "${repo_dir}" status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before ScreenQA ranker collection" >&2
  exit 2
fi
actual_code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_code_revision}" != "${BE_SCREENQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "ScreenQA ranker collection code revision mismatch" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
/userhome/cs3/yihangc/anaconda3/bin/python \
  "${repo_dir}/scripts/verify_screenqa_ranker_manifest.py" \
  --manifest-dir "${BE_SCREENQA_MANIFEST_DIR}" \
  --expected-manifest-sha256 "${BE_SCREENQA_MANIFEST_SHA256}" \
  --expected-audit-sha256 "${BE_SCREENQA_MANIFEST_AUDIT_SHA256}" \
  --expected-states "${BE_SCREENQA_EXPECTED_STATES}"

export BE_CODE_REVISION="${actual_code_revision}"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${shard_dir}"
collect_args=(
  -m beyond_entropy collect-qwen
  --manifest "${manifest}"
  --expected-manifest-sha256 "${BE_SCREENQA_MANIFEST_SHA256}"
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
  --scientific-status "ScreenQA frozen ranker-training sibling bank; outcomes may fit the sole candidate only"
  --max-new-tokens 32
  --min-pixels 200704
  --max-pixels 602112
  --attention-implementation sdpa
  --system-prompt "You are a helpful assistant."
)

cd "${repo_dir}"
"${python_bin}" "${collect_args[@]}"
first_sha=$(sha256sum "${rollouts}")
first_sha=${first_sha%% *}
cp "${rollouts%.jsonl}.provenance.json" "${shard_dir}/rollouts.first-pass.provenance.json"

"${python_bin}" "${collect_args[@]}"
second_sha=$(sha256sum "${rollouts}")
second_sha=${second_sha%% *}
if [[ "${first_sha}" != "${second_sha}" ]]; then
  echo "ScreenQA resume changed completed ranker rollout bytes" >&2
  exit 2
fi

"${python_bin}" - "${rollouts}" "${first_sha}" <<'PY'
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
(rollouts.parent / "resume.audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if not passed:
    raise SystemExit("ScreenQA ranker resume audit failed")
print(json.dumps(audit, indent=2, sort_keys=True))
PY
