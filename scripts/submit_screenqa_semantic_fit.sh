#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
semantic_root="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-training/semantic-oof-v2"
feature_dir="${semantic_root}/features"
features="${feature_dir}/features-question-region-attention-label-free.pt"
label_free_audit="${feature_dir}/label-free-audit.json"
feature_bundle="${feature_dir}/SHA256SUMS"
activation="${semantic_root}/activation.audit.json"
fit_dir="${semantic_root}/hybrid-context-semantic-oof-v2"
candidate_dir="${semantic_root}/candidate-v2"
rollouts="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-rollouts-v1/merged/rollouts.jsonl"
protocol="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/ranker-development-protocol-v2-semantic.md"
worker="${repo_dir}/scripts/slurm_screenqa_semantic_fit.sh"
feature_job_id=197065
feature_code_revision=4e71fcba0d222147fa6f34658bf1f874a5c17d87
rollouts_sha256=0437d2a499adccb1b4e19eb0160583789cee00edf244718ecae9e290108bb8c9
protocol_sha256=925feba44324bf4e09aec5a7c162cc2f034bfb1e06cbae18fbaf1714d28d3a46

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid ScreenQA notification email" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA semantic fit submission" >&2
  exit 2
fi
for path in "${features}" "${label_free_audit}" "${feature_bundle}" "${activation}" "${rollouts}" "${protocol}" "${worker}"; do
  if [[ ! -s "${path}" ]]; then
    echo "missing completed ScreenQA semantic fit input: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${fit_dir}" || -e "${candidate_dir}" || -e "${semantic_root}/semantic-fit.audit.json" ]]; then
  echo "ScreenQA semantic fit output already exists" >&2
  exit 2
fi
(
  cd "${feature_dir}"
  sha256sum --check SHA256SUMS
)
actual_rollouts_sha256=$(sha256sum "${rollouts}")
actual_rollouts_sha256=${actual_rollouts_sha256%% *}
if [[ "${actual_rollouts_sha256}" != "${rollouts_sha256}" ]]; then
  echo "ScreenQA semantic fit rollout hash mismatch" >&2
  exit 2
fi
actual_protocol_sha256=$(sha256sum "${protocol}")
actual_protocol_sha256=${actual_protocol_sha256%% *}
if [[ "${actual_protocol_sha256}" != "${protocol_sha256}" ]]; then
  echo "ScreenQA semantic fit protocol hash mismatch" >&2
  exit 2
fi
if [[ "$(jq -r '.semantic_code_revision // ""' "${activation}")" != "${feature_code_revision}" ]]; then
  echo "ScreenQA semantic feature revision mismatch" >&2
  exit 2
fi

features_sha256=$(sha256sum "${features}")
features_sha256=${features_sha256%% *}
label_free_audit_sha256=$(sha256sum "${label_free_audit}")
label_free_audit_sha256=${label_free_audit_sha256%% *}
feature_bundle_sha256=$(sha256sum "${feature_bundle}")
feature_bundle_sha256=${feature_bundle_sha256%% *}
activation_sha256=$(sha256sum "${activation}")
activation_sha256=${activation_sha256%% *}
worker_sha256=$(sha256sum "${worker}")
worker_sha256=${worker_sha256%% *}
fit_code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
export_args="ALL,BE_SCREENQA_RANKER_ROLLOUTS=${rollouts},BE_SCREENQA_RANKER_ROLLOUTS_SHA256=${rollouts_sha256},BE_SCREENQA_SEMANTIC_FEATURES=${features},BE_SCREENQA_SEMANTIC_FEATURES_SHA256=${features_sha256},BE_SCREENQA_LABEL_FREE_AUDIT=${label_free_audit},BE_SCREENQA_LABEL_FREE_AUDIT_SHA256=${label_free_audit_sha256},BE_SCREENQA_FEATURE_BUNDLE=${feature_bundle},BE_SCREENQA_FEATURE_BUNDLE_SHA256=${feature_bundle_sha256},BE_SCREENQA_SEMANTIC_ACTIVATION=${activation},BE_SCREENQA_SEMANTIC_ACTIVATION_SHA256=${activation_sha256},BE_SCREENQA_V2_PROTOCOL=${protocol},BE_SCREENQA_V2_PROTOCOL_SHA256=${protocol_sha256},BE_SCREENQA_SEMANTIC_FIT_DIR=${fit_dir},BE_SCREENQA_SEMANTIC_CANDIDATE_DIR=${candidate_dir},BE_SCREENQA_FEATURE_CODE_REVISION=${feature_code_revision},BE_SCREENQA_FIT_CODE_REVISION=${fit_code_revision},BE_SCREENQA_SEMANTIC_FIT_WORKER_SHA256=${worker_sha256},BE_SCREENQA_FEATURE_JOB_ID=${feature_job_id}"
submission=$(
  /usr/local/slurm/bin/sbatch \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${worker}"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA semantic fit job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'screenqa_semantic_fit_job_id=%s feature_job_id=%s gpu_count=0 fit_code_revision=%s\n' \
  "${job_id}" "${feature_job_id}" "${fit_code_revision}"
