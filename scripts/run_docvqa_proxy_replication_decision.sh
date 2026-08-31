#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
root="${repo_dir}/artifacts/docvqa-train-factorized-v2/proxy-to-outcome-cross-domain-v1/full-v1"
report="${root}/analysis/report.json"
audit_completion="${root}/analysis/audit.complete.json"
output_dir="${root}/decision"
protocol="${repo_dir}/artifacts/docvqa-train-factorized-v2/ops/proxy-to-outcome-cross-domain-protocol-v1.md"
decision_contract="${repo_dir}/artifacts/docvqa-train-factorized-v2/ops/proxy-replication-decision-implementation-v1.md"
decision_module="${repo_dir}/src/beyond_entropy/proxy_replication_decision.py"
decision_cli="${repo_dir}/scripts/decide_proxy_replication.py"
protocol_sha256=106879da7d15db351a4145e5a06c43fc3f33803182d1ca4e6f08362b076f8cbe
decision_contract_sha256=721ab2a7dd2749f77c95fee2f316e865aca259e975bb9f44747c12d6eb404a66
decision_module_sha256=3f1fa418f53ee6fecc3e0889bedff164c09bdb0e716dc17cf38a0f0df1f79727
decision_cli_sha256=61bbcd5392eceb65837d95ffc25c23f8b4e29690eb4683037afe5dc176232204
decision_code_revision=9b2116f793e0e68d16b31ee4a5b96db6fff0105c

check_hash() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "DocVQA replication decision ${label} hash mismatch" >&2
    exit 2
  fi
}

if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before DocVQA replication decision" >&2
  exit 2
fi
for path in "${report}" "${audit_completion}" "${protocol}" \
  "${decision_contract}" "${decision_module}" "${decision_cli}"; do
  if [[ ! -s "${path}" ]]; then
    echo "missing DocVQA replication decision input: ${path}" >&2
    exit 2
  fi
done
check_hash "${protocol}" "${protocol_sha256}" protocol
check_hash "${decision_contract}" "${decision_contract_sha256}" contract
check_hash "${decision_module}" "${decision_module_sha256}" module
check_hash "${decision_cli}" "${decision_cli_sha256}" CLI

report_sha256=$(sha256sum "${report}" | cut -d ' ' -f 1)
if [[ "$(jq -r '.report_sha256' "${audit_completion}")" != "${report_sha256}" ]]; then
  echo "DocVQA audit completion/report hash mismatch" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
"${python_bin}" "${decision_cli}" \
  --report "${report}" \
  --protocol "${protocol}" \
  --output-dir "${output_dir}" \
  --expected-report-sha256 "${report_sha256}" \
  --expected-protocol-sha256 "${protocol_sha256}" \
  --expected-study-label "DocVQA ranker development" \
  --code-revision "${decision_code_revision}"

decision_json="${output_dir}/decision.json"
decision_completion="${output_dir}/decision.complete.json"
decision=$(jq -r '.decision' "${decision_json}")
if [[ "${decision}" != replicated_alignment \
  && "${decision}" != partial_alignment \
  && "${decision}" != non_replication ]]; then
  echo "DocVQA replication decision has an invalid status" >&2
  exit 2
fi
if [[ "$(jq -r '.selection.score_threshold_selected' "${decision_json}")" != false \
  || "$(jq -r '.selection.call_rate_selected' "${decision_json}")" != false \
  || "$(jq -r '.selection.protected_outcome_used' "${decision_json}")" != false \
  || "$(jq -r '.decision' "${decision_completion}")" != "${decision}" ]]; then
  echo "DocVQA replication decision output contract failed" >&2
  exit 2
fi
printf 'docvqa_proxy_replication_decision=%s decision_sha256=%s report_sha256=%s\n' \
  "${decision}" "$(sha256sum "${decision_json}" | cut -d ' ' -f 1)" \
  "${report_sha256}"
