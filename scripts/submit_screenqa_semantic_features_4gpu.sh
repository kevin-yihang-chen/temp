#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
fit_root="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-training/low-capacity-oof-v1"
candidate_dir="${fit_root}/candidate-v1"
candidate_audit="${candidate_dir}/candidate.audit.json"
rollouts="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-rollouts-v1/merged/rollouts.jsonl"
input_audit="${fit_root}/ranker-rollouts.audit.json"
v1_protocol="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/ranker-development-protocol-v1.md"
v2_protocol="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/ranker-development-protocol-v2-semantic.md"
semantic_root="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-training/semantic-oof-v2"
activation_audit="${semantic_root}/activation.audit.json"
feature_dir="${semantic_root}/features"
runner="${repo_dir}/scripts/slurm_screenqa_semantic_features_4gpu.sh"
preparer="${repo_dir}/scripts/prepare_semantic_feature_batch_shards.py"
merger="${repo_dir}/scripts/merge_semantic_feature_shards.py"
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
rollouts_sha256=0437d2a499adccb1b4e19eb0160583789cee00edf244718ecae9e290108bb8c9
input_audit_sha256=0651debaeb5e742f6823e7321e8bfe8184a398a468e42e36dd033f68af74563c
v1_protocol_sha256=c6118d8a013a171c3eecad374a3271e3bf00dfd199864d3efaab27c7b44e36b7
v2_protocol_sha256=925feba44324bf4e09aec5a7c162cc2f034bfb1e06cbae18fbaf1714d28d3a46
preparer_sha256=28f3e3b06007cb9a14e7cdef0ec7a631a67581cb2a6618dd4249aec2d1da22f1
merger_sha256=3b1051ea28b07a5aefd70c4c347c43410c1023cc35eed739216dc0d0d1d3ff30
resume_mode=${BE_SCREENQA_SEMANTIC_RESUME:-0}
calibration_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/calibration-manifest-v1"
formal_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/formal-manifest-v1"
reserve_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/reserve-manifest-v1"
untouched_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/untouched-manifest-v1"

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
  echo "tracked worktree must be clean before ScreenQA semantic submission" >&2
  exit 2
fi
if [[ "${resume_mode}" != 0 && "${resume_mode}" != 1 ]]; then
  echo "ScreenQA semantic resume flag must be 0 or 1" >&2
  exit 2
fi
for path in "${candidate_audit}" "${candidate_dir}/SHA256SUMS" "${rollouts}" "${input_audit}" "${v1_protocol}" "${v2_protocol}" "${runner}" "${preparer}" "${merger}"; do
  if [[ ! -s "${path}" ]]; then
    echo "missing ScreenQA semantic input: ${path}" >&2
    exit 2
  fi
done
if [[ -d "${feature_dir}" && "${resume_mode}" != 1 ]]; then
  if [[ -n "$(find "${feature_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "existing ScreenQA semantic features require audited resume" >&2
    exit 2
  fi
fi

check_hash() {
  local path=$1 expected=$2 name=$3 actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ScreenQA semantic ${name} SHA-256 mismatch" >&2
    exit 2
  fi
}
check_hash "${rollouts}" "${rollouts_sha256}" "rollouts"
check_hash "${input_audit}" "${input_audit_sha256}" "input audit"
check_hash "${v1_protocol}" "${v1_protocol_sha256}" "v1 protocol"
check_hash "${v2_protocol}" "${v2_protocol_sha256}" "v2 protocol"
check_hash "${preparer}" "${preparer_sha256}" "shard preparer"
check_hash "${merger}" "${merger_sha256}" "shard merger"
(
  cd "${candidate_dir}"
  sha256sum --check SHA256SUMS
)

candidate_audit_sha256=$(sha256sum "${candidate_audit}")
candidate_audit_sha256=${candidate_audit_sha256%% *}
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
mkdir -p "${semantic_root}"
activation_args=(
  --candidate-dir "${candidate_dir}"
  --expected-candidate-audit-sha256 "${candidate_audit_sha256}"
  --ranker-rollouts "${rollouts}"
  --expected-ranker-rollouts-sha256 "${rollouts_sha256}"
  --ranker-input-audit "${input_audit}"
  --expected-ranker-input-audit-sha256 "${input_audit_sha256}"
  --v1-protocol "${v1_protocol}"
  --expected-v1-protocol-sha256 "${v1_protocol_sha256}"
  --v2-protocol "${v2_protocol}"
  --expected-v2-protocol-sha256 "${v2_protocol_sha256}"
  --sealed-output-dir "${calibration_dir}"
  --sealed-output-dir "${formal_dir}"
  --sealed-output-dir "${reserve_dir}"
  --sealed-output-dir "${untouched_dir}"
  --expected-code-revision "${code_revision}"
  --output "${activation_audit}"
)
if [[ -e "${activation_audit}" ]]; then activation_args+=(--resume); fi
PYTHONPATH="${repo_dir}/src" "${python_bin}" scripts/verify_screenqa_semantic_activation.py "${activation_args[@]}"
activation_sha256=$(sha256sum "${activation_audit}")
activation_sha256=${activation_sha256%% *}
runner_sha256=$(sha256sum "${runner}")
runner_sha256=${runner_sha256%% *}

export_args="ALL,BE_SCREENQA_CANDIDATE_DIR=${candidate_dir},BE_SCREENQA_CANDIDATE_AUDIT_SHA256=${candidate_audit_sha256},BE_SCREENQA_RANKER_ROLLOUTS=${rollouts},BE_SCREENQA_RANKER_ROLLOUTS_SHA256=${rollouts_sha256},BE_SCREENQA_RANKER_INPUT_AUDIT=${input_audit},BE_SCREENQA_RANKER_INPUT_AUDIT_SHA256=${input_audit_sha256},BE_SCREENQA_V1_PROTOCOL=${v1_protocol},BE_SCREENQA_V1_PROTOCOL_SHA256=${v1_protocol_sha256},BE_SCREENQA_V2_PROTOCOL=${v2_protocol},BE_SCREENQA_V2_PROTOCOL_SHA256=${v2_protocol_sha256},BE_SCREENQA_SEMANTIC_ACTIVATION=${activation_audit},BE_SCREENQA_SEMANTIC_ACTIVATION_SHA256=${activation_sha256},BE_SCREENQA_SEMANTIC_FEATURE_DIR=${feature_dir},BE_SCREENQA_EXPECTED_CODE_REVISION=${code_revision},BE_SCREENQA_SEMANTIC_RUNNER_SHA256=${runner_sha256},BE_SCREENQA_SHARD_PREPARER_SHA256=${preparer_sha256},BE_SCREENQA_SHARD_MERGER_SHA256=${merger_sha256},BE_SCREENQA_SEMANTIC_RESUME=${resume_mode}"
submission=$(
  /usr/local/slurm/bin/sbatch \
    --mail-user="${notify_email}" \
    --mail-type=ALL \
    --export="${export_args}" \
    "${runner}"
)
job_id=${submission##* }
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse ScreenQA semantic feature job ID: ${submission}" >&2
  exit 2
fi
printf '%s\n' "${submission}"
printf 'screenqa_semantic_feature_job_id=%s gpu_count=4 code_revision=%s activation_sha256=%s\n' \
  "${job_id}" "${code_revision}" "${activation_sha256}"
