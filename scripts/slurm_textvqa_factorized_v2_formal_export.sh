#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-textvqa-fv2-formal-export-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_FV2_POLICY_FREEZE:?missing BE_FV2_POLICY_FREEZE}"
: "${BE_FV2_POLICY_FREEZE_SHA256:?missing BE_FV2_POLICY_FREEZE_SHA256}"
: "${BE_CODE_REVISION:?missing BE_CODE_REVISION}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
allocation="${repo_dir}/data/textvqa-train-factorized-v2/allocation.json"
allocation_audit="${repo_dir}/data/textvqa-train-factorized-v2/allocation.audit.json"
output_dir="${repo_dir}/data/textvqa-train-factorized-v2/formal-test"
audit_output="${repo_dir}/data/textvqa-train-factorized-v2/formal-test.audit.json"

if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_CODE_REVISION}" ]]; then
  echo "formal export revision differs from the frozen submission" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before formal export" >&2
  exit 2
fi
export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
"${python_bin}" scripts/export_textvqa_factorized_v2_formal.py \
  --allocation "${allocation}" \
  --allocation-audit "${allocation_audit}" \
  --policy-freeze "${BE_FV2_POLICY_FREEZE}" \
  --expected-policy-freeze-sha256 "${BE_FV2_POLICY_FREEZE_SHA256}" \
  --output-dir "${output_dir}" \
  --audit-output "${audit_output}"
