#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-textvqa-train-scale-formal-export-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_FORMAL_POLICY_FREEZE_SHA256:?missing BE_FORMAL_POLICY_FREEZE_SHA256}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
allocation="${repo_dir}/data/textvqa-train-scale-v1/allocation.json"
policy_freeze="${repo_dir}/artifacts/textvqa-train-scale-v1/pairwise-attention-primary-v1/policy-freeze.json"
output_dir="${repo_dir}/data/textvqa-train-scale-v1/formal-test"
audit_output="${repo_dir}/data/textvqa-train-scale-v1/formal-test.audit.json"
allocation_sha256=da6d41584bf4f3bfb91426fa9fa3bcb61a659846147c279eaab2aedb776e1657

export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${repo_dir}"
"${python_bin}" scripts/export_textvqa_train_scale_formal.py \
  --allocation "${allocation}" \
  --expected-allocation-sha256 "${allocation_sha256}" \
  --policy-freeze "${policy_freeze}" \
  --expected-policy-freeze-sha256 "${BE_FORMAL_POLICY_FREEZE_SHA256}" \
  --output-dir "${output_dir}" \
  --audit-output "${audit_output}"
