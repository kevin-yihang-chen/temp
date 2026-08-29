#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-docvqa-train-factorized-v2-allocation-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_DOCVQA_SOURCE_DIR:?missing BE_DOCVQA_SOURCE_DIR}"
: "${BE_DOCVQA_ALLOCATION:?missing BE_DOCVQA_ALLOCATION}"
: "${BE_DOCVQA_ALLOCATION_AUDIT:?missing BE_DOCVQA_ALLOCATION_AUDIT}"
: "${BE_DOCVQA_PROTOCOL:?missing BE_DOCVQA_PROTOCOL}"
: "${BE_DOCVQA_EXPECTED_CODE_REVISION:?missing BE_DOCVQA_EXPECTED_CODE_REVISION}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python

tracked_status=$(git -C "${repo_dir}" status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before DocVQA identity allocation" >&2
  exit 2
fi
actual_code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_code_revision}" != "${BE_DOCVQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "DocVQA allocation code revision mismatch" >&2
  exit 2
fi
if [[ ! -f "${BE_DOCVQA_PROTOCOL}" ]]; then
  echo "DocVQA preregistration is missing" >&2
  exit 2
fi
for prior_root in "${repo_dir}/data"; do
  if [[ ! -d "${prior_root}" ]]; then
    echo "DocVQA prior-manifest root is missing: ${prior_root}" >&2
    exit 2
  fi
done

parquet_args=()
for shard_index in {00..11}; do
  shard="${BE_DOCVQA_SOURCE_DIR}/DocVQA/train-000${shard_index}-of-00012.parquet"
  if [[ ! -s "${shard}" ]]; then
    echo "DocVQA train shard is missing or empty: ${shard}" >&2
    exit 2
  fi
  parquet_args+=(--parquet-file "${shard}")
done
if [[ "${#parquet_args[@]}" -ne 24 ]]; then
  echo "DocVQA allocation did not resolve exactly 12 Parquet shards" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${repo_dir}"
"${python_bin}" scripts/allocate_docvqa_train_factorized_v2.py \
  "${parquet_args[@]}" \
  --prior-manifest-root "${repo_dir}/data" \
  --protocol "${BE_DOCVQA_PROTOCOL}" \
  --allocation-output "${BE_DOCVQA_ALLOCATION}" \
  --audit-output "${BE_DOCVQA_ALLOCATION_AUDIT}" \
  --resume

"${python_bin}" scripts/verify_docvqa_train_factorized_v2_allocation.py \
  --allocation "${BE_DOCVQA_ALLOCATION}" \
  --audit "${BE_DOCVQA_ALLOCATION_AUDIT}" \
  --protocol "${BE_DOCVQA_PROTOCOL}"
