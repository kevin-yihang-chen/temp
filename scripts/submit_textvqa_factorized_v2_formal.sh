#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
policy_freeze="${repo_dir}/artifacts/textvqa-train-factorized-v2/fixed-sequence-calibrated/policy-freeze.json"
model="${repo_dir}/artifacts/textvqa-train-factorized-v2/fixed-sequence-calibrated/model.json"
formal_dir="${repo_dir}/data/textvqa-train-factorized-v2/formal-test"
manifest="${formal_dir}/manifest.jsonl"
manifest_provenance="${formal_dir}/manifest.provenance.json"
formal_audit="${repo_dir}/data/textvqa-train-factorized-v2/formal-test.audit.json"
run_dir="${repo_dir}/artifacts/textvqa-train-factorized-v2/formal-test/qwen3b-c4-seed0"
feature_dir="${repo_dir}/artifacts/textvqa-train-factorized-v2/formal-test/attention-semantic-v1"
report="${repo_dir}/artifacts/textvqa-train-factorized-v2/formal-test/evaluation/report.json"
scientific_status="one-shot factorized-v2 TextVQA formal sibling bank frozen before export no target tuning"

if [[ ! -r "${mail_file}" ]]; then
  echo "missing private Slurm email file" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "invalid notification email" >&2
  exit 2
fi
for path in "${policy_freeze}" "${model}" "${manifest}" "${manifest_provenance}" "${formal_audit}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing frozen formal input: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${run_dir}/rollouts.jsonl" || -e "${feature_dir}/features-question-region-attention-label-free.pt" || -e "${report}" ]]; then
  echo "refusing to reuse existing factorized-v2 formal outcomes" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before formal submission" >&2
  exit 2
fi

policy_freeze_sha256=$(sha256sum "${policy_freeze}")
policy_freeze_sha256=${policy_freeze_sha256%% *}
model_sha256=$(sha256sum "${model}")
model_sha256=${model_sha256%% *}
manifest_sha256=$(sha256sum "${manifest}")
manifest_sha256=${manifest_sha256%% *}
manifest_provenance_sha256=$(sha256sum "${manifest_provenance}")
manifest_provenance_sha256=${manifest_provenance_sha256%% *}
formal_audit_sha256=$(sha256sum "${formal_audit}")
formal_audit_sha256=${formal_audit_sha256%% *}
expected_states=$(wc -l < "${manifest}")
if [[ "$(jq -r '.formal.unique_sources' "${formal_audit}")" -ne 5953 ]]; then
  echo "formal audit does not contain exactly 5,953 sources" >&2
  exit 2
fi
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

PYTHONPATH="${repo_dir}/src" /userhome/cs3/yihangc/anaconda3/bin/python \
  "${repo_dir}/scripts/verify_factorized_v2_formal_gate.py" \
  --policy-freeze "${policy_freeze}" \
  --expected-policy-freeze-sha256 "${policy_freeze_sha256}" \
  --model "${model}" \
  --expected-model-sha256 "${model_sha256}" \
  --manifest "${manifest}" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --manifest-provenance "${manifest_provenance}" \
  --expected-manifest-provenance-sha256 "${manifest_provenance_sha256}" \
  --audit "${formal_audit}" \
  --expected-audit-sha256 "${formal_audit_sha256}"

common_export="ALL,BE_FV2_POLICY_FREEZE=${policy_freeze},BE_FV2_POLICY_FREEZE_SHA256=${policy_freeze_sha256},BE_FV2_MODEL=${model},BE_FV2_MODEL_SHA256=${model_sha256},BE_FV2_MANIFEST=${manifest},BE_FV2_MANIFEST_SHA256=${manifest_sha256},BE_FV2_MANIFEST_PROVENANCE=${manifest_provenance},BE_FV2_MANIFEST_PROVENANCE_SHA256=${manifest_provenance_sha256},BE_FV2_FORMAL_AUDIT=${formal_audit},BE_FV2_FORMAL_AUDIT_SHA256=${formal_audit_sha256},BE_FV2_EXPECTED_STATES=${expected_states},BE_FV2_SCIENTIFIC_STATUS=${scientific_status},BE_CODE_REVISION=${code_revision}"

rollout_submission=$(
  sbatch \
    --job-name=be-tvqa-fv2-formal-rollout \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${common_export},BE_FV2_RUN_DIR=${run_dir}" \
    "${repo_dir}/scripts/slurm_textvqa_factorized_v2_formal_rollout.sh"
)
rollout_job_id=${rollout_submission##* }
if [[ ! "${rollout_job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse rollout job ID: ${rollout_submission}" >&2
  exit 2
fi
printf '%s\n' "${rollout_submission}"

feature_submission=$(
  sbatch \
    --dependency="afterok:${rollout_job_id}" \
    --job-name=be-tvqa-fv2-formal-features \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${common_export},BE_FV2_ROLLOUTS=${run_dir}/rollouts.jsonl,BE_FV2_FEATURE_DIR=${feature_dir}" \
    "${repo_dir}/scripts/slurm_textvqa_factorized_v2_formal_features.sh"
)
feature_job_id=${feature_submission##* }
if [[ ! "${feature_job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse feature job ID: ${feature_submission}" >&2
  exit 2
fi
printf '%s\n' "${feature_submission}"

evaluation_submission=$(
  sbatch \
    --dependency="afterok:${feature_job_id}" \
    --job-name=be-tvqa-fv2-formal-evaluate \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${common_export},BE_FV2_ROLLOUTS=${run_dir}/rollouts.jsonl,BE_FV2_FEATURE_DIR=${feature_dir},BE_FV2_REPORT=${report}" \
    "${repo_dir}/scripts/slurm_textvqa_factorized_v2_formal_evaluate.sh"
)
evaluation_job_id=${evaluation_submission##* }
if [[ ! "${evaluation_job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse evaluation job ID: ${evaluation_submission}" >&2
  exit 2
fi
printf '%s\n' "${evaluation_submission}"
printf 'rollout_job_id=%s feature_job_id=%s evaluation_job_id=%s\n' \
  "${rollout_job_id}" "${feature_job_id}" "${evaluation_job_id}"
