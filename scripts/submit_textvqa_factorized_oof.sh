#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
slurm_script="${repo_dir}/scripts/slurm_textvqa_factorized_oof.sh"
artifact_root="${repo_dir}/artifacts/textvqa-train-scale-v1/factorized-oof-development-v1"

cd "${repo_dir}"
tracked_status=$(git status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before submitting factorized OOF jobs" >&2
  exit 2
fi

for mode in semantic-context hybrid-context-semantic; do
  slug=${mode//-/_}
  output_dir="${artifact_root}/${mode}"
  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing factorized output: ${output_dir}" >&2
    exit 2
  fi
  sbatch \
    --job-name="be-factor-${slug}" \
    --export="ALL,BE_FACTOR_FEATURE_MODE=${mode},BE_FACTOR_OUTPUT_DIR=${output_dir}" \
    "${slurm_script}"
done
