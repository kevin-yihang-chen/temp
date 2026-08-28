#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
slurm_script="${repo_dir}/scripts/slurm_textvqa_factorized_oof.sh"
artifact_root="${repo_dir}/artifacts/textvqa-train-scale-v1/factorized-oof-tail-sweep-v1/hybrid-context-semantic"

cd "${repo_dir}"
tracked_status=$(git status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before submitting the tail sweep" >&2
  exit 2
fi

for alpha in 0.1 10 100 1000; do
  slug=${alpha//./p}
  output_dir="${artifact_root}/alpha-${slug}"
  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing tail sweep output: ${output_dir}" >&2
    exit 2
  fi
  sbatch \
    --job-name="be-factor-tail-a${slug}" \
    --export="ALL,BE_FACTOR_FEATURE_MODE=hybrid-context-semantic,BE_FACTOR_OUTPUT_DIR=${output_dir},BE_FACTOR_ALPHA=${alpha}" \
    "${slurm_script}"
done
