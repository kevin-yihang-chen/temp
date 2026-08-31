#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=be-screenqa-paper-analysis
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-formal-paper-analysis-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SCREENQA_FORMAL_EVALUATION_DIR:?missing BE_SCREENQA_FORMAL_EVALUATION_DIR}"
: "${BE_SCREENQA_FORMAL_PAPER_ANALYSIS_DIR:?missing BE_SCREENQA_FORMAL_PAPER_ANALYSIS_DIR}"
: "${BE_SCREENQA_FORMAL_PAPER_PROTOCOL:?missing BE_SCREENQA_FORMAL_PAPER_PROTOCOL}"
: "${BE_SCREENQA_FORMAL_PAPER_PROTOCOL_SHA256:?missing BE_SCREENQA_FORMAL_PAPER_PROTOCOL_SHA256}"
: "${BE_SCREENQA_EXPECTED_CODE_REVISION:?missing BE_SCREENQA_EXPECTED_CODE_REVISION}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/bin/python

if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA formal paper analysis" >&2
  exit 2
fi
if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_SCREENQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "ScreenQA formal paper-analysis revision mismatch" >&2
  exit 2
fi
actual_protocol_sha256=$(sha256sum "${BE_SCREENQA_FORMAL_PAPER_PROTOCOL}")
actual_protocol_sha256=${actual_protocol_sha256%% *}
if [[ "${actual_protocol_sha256}" != "${BE_SCREENQA_FORMAL_PAPER_PROTOCOL_SHA256}" ]]; then
  echo "ScreenQA formal paper-analysis protocol SHA-256 mismatch" >&2
  exit 2
fi
for path in \
  "${BE_SCREENQA_FORMAL_EVALUATION_DIR}/report.json" \
  "${BE_SCREENQA_FORMAL_EVALUATION_DIR}/formal-result.complete.json" \
  "${BE_SCREENQA_FORMAL_EVALUATION_DIR}/SHA256SUMS"; do
  if [[ ! -s "${path}" ]]; then
    echo "ScreenQA formal evaluation is incomplete: ${path}" >&2
    exit 2
  fi
done
cd "${BE_SCREENQA_FORMAL_EVALUATION_DIR}"
sha256sum --check SHA256SUMS

verify_analysis() {
  "${python_bin}" - "${BE_SCREENQA_FORMAL_PAPER_ANALYSIS_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("schema") != "screenqa_formal_paper_analysis_manifest_v1":
    raise SystemExit("ScreenQA paper-analysis manifest schema mismatch")
for name, expected in manifest["files"].items():
    path = root / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"ScreenQA paper-analysis hash mismatch: {path}")
report = json.loads((root / "report.json").read_text(encoding="utf-8"))
if report.get("schema") != "screenqa_formal_paper_analysis_v1":
    raise SystemExit("ScreenQA paper-analysis report schema mismatch")
if report.get("outcome_use", {}).get("formal_outcomes_used_for_tuning") is not False:
    raise SystemExit("ScreenQA paper analysis permits formal tuning")
PY
}

if [[ -e "${BE_SCREENQA_FORMAL_PAPER_ANALYSIS_DIR}" ]]; then
  verify_analysis
  echo "ScreenQA formal paper analysis was already completed; no reanalysis performed"
  exit 0
fi

export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${repo_dir}"
"${python_bin}" -m scripts.analyze_screenqa_formal_paper --self-test
"${python_bin}" -m scripts.analyze_screenqa_formal_paper \
  --formal-evaluation-dir "${BE_SCREENQA_FORMAL_EVALUATION_DIR}" \
  --protocol "${BE_SCREENQA_FORMAL_PAPER_PROTOCOL}" \
  --expected-protocol-sha256 "${BE_SCREENQA_FORMAL_PAPER_PROTOCOL_SHA256}" \
  --output-dir "${BE_SCREENQA_FORMAL_PAPER_ANALYSIS_DIR}"
verify_analysis
