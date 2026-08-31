#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=be-docvqa-reserve-export
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-docvqa-reserve-export-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_RESERVE_FREEZE BE_RESERVE_FREEZE_SHA256 BE_RESERVE_EXPECTED_CODE_REVISION
  BE_RESERVE_MANIFEST_DIR BE_RESERVE_MANIFEST_AUDIT
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then echo "missing ${name}" >&2; exit 2; fi
done
repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_RESERVE_EXPECTED_CODE_REVISION}" ]]; then
  echo "reserve export code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before reserve export" >&2
  exit 2
fi
export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
"${python_bin}" scripts/export_docvqa_reserve_toolgate.py \
  --freeze "${BE_RESERVE_FREEZE}" \
  --expected-freeze-sha256 "${BE_RESERVE_FREEZE_SHA256}" \
  --output-dir "${BE_RESERVE_MANIFEST_DIR}" \
  --audit-output "${BE_RESERVE_MANIFEST_AUDIT}"
