#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-docvqa-fv2-formal-export-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_DOCVQA_POLICY_FREEZE:?missing BE_DOCVQA_POLICY_FREEZE}"
: "${BE_DOCVQA_POLICY_FREEZE_SHA256:?missing BE_DOCVQA_POLICY_FREEZE_SHA256}"
: "${BE_DOCVQA_EXPECTED_CODE_REVISION:?missing BE_DOCVQA_EXPECTED_CODE_REVISION}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
allocation="${repo_dir}/data/docvqa-train-factorized-v2/allocation.json"
allocation_audit="${repo_dir}/data/docvqa-train-factorized-v2/allocation.audit.json"
protocol="${repo_dir}/docs/docvqa_train_factorized_v2_preregistration.md"
output_dir="${repo_dir}/data/docvqa-train-factorized-v2/formal-test"
audit_output="${repo_dir}/data/docvqa-train-factorized-v2/formal-test.audit.json"

if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_DOCVQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "DocVQA formal export revision differs from policy freeze" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before DocVQA formal export" >&2
  exit 2
fi
export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
"${python_bin}" scripts/export_docvqa_train_factorized_v2_formal.py \
  --allocation "${allocation}" \
  --allocation-audit "${allocation_audit}" \
  --protocol "${protocol}" \
  --policy-freeze "${BE_DOCVQA_POLICY_FREEZE}" \
  --expected-policy-freeze-sha256 "${BE_DOCVQA_POLICY_FREEZE_SHA256}" \
  --output-dir "${output_dir}" \
  --audit-output "${audit_output}"
