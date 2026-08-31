#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=be-screenqa-submit-semantic-fit
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-submit-semantic-fit-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_SCREENQA_SEMANTIC_DEFERRED_SHA256:?missing BE_SCREENQA_SEMANTIC_DEFERRED_SHA256}"
: "${BE_SCREENQA_SEMANTIC_FIT_SUBMITTER_SHA256:?missing BE_SCREENQA_SEMANTIC_FIT_SUBMITTER_SHA256}"
: "${BE_SCREENQA_FEATURE_JOB_ID:?missing BE_SCREENQA_FEATURE_JOB_ID}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
deferred="${repo_dir}/scripts/slurm_deferred_screenqa_semantic_fit_submit.sh"
submitter="${repo_dir}/scripts/submit_screenqa_semantic_fit.sh"

actual_deferred_sha256=$(sha256sum "${deferred}")
actual_deferred_sha256=${actual_deferred_sha256%% *}
actual_submitter_sha256=$(sha256sum "${submitter}")
actual_submitter_sha256=${actual_submitter_sha256%% *}
if [[ "${actual_deferred_sha256}" != "${BE_SCREENQA_SEMANTIC_DEFERRED_SHA256}" ]]; then
  echo "ScreenQA deferred semantic fit worker SHA-256 mismatch" >&2
  exit 2
fi
if [[ "${actual_submitter_sha256}" != "${BE_SCREENQA_SEMANTIC_FIT_SUBMITTER_SHA256}" ]]; then
  echo "ScreenQA semantic fit submitter SHA-256 mismatch" >&2
  exit 2
fi
if [[ "${BE_SCREENQA_FEATURE_JOB_ID}" != 197065 ]]; then
  echo "ScreenQA deferred semantic feature job mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before deferred ScreenQA semantic fit" >&2
  exit 2
fi
cd "${repo_dir}"
"${submitter}"
