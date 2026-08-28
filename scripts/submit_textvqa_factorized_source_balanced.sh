#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
slurm_script="${repo_dir}/scripts/slurm_textvqa_factorized_oof.sh"
artifact_root="${repo_dir}/artifacts/textvqa-train-scale-v1/factorized-oof-source-balanced-v2"

cd "${repo_dir}"
tracked_status=$(git status --porcelain --untracked-files=no)
if [[ -n "${tracked_status}" ]]; then
  echo "Tracked worktree must be clean before source-balanced OOF submission" >&2
  exit 2
fi

for mode in hybrid-context-semantic semantic-context; do
  for alpha in 1 10; do
    slug=${mode//-/_}
    output_dir="${artifact_root}/${mode}/alpha-${alpha}"
    if [[ -e "${output_dir}" ]]; then
      echo "Refusing to overwrite existing output: ${output_dir}" >&2
      exit 2
    fi
    sbatch \
      --job-name="be-sbw-${slug}-a${alpha}" \
      --export="ALL,BE_FACTOR_FEATURE_MODE=${mode},BE_FACTOR_OUTPUT_DIR=${output_dir},BE_FACTOR_ALPHA=${alpha}" \
      "${slurm_script}"
  done
done
