#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
root="${repo_dir}/artifacts/screenqa-train-factorized-v1/backbone-7b-diagnostic-v1/full-h800-v1"
report="${root}/analysis/report.json"
analysis_completion="${root}/analysis/audit.complete.json"
protocol="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-diagnostic-protocol-v1.md"
decision_contract="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-decision-implementation-v1.md"
decision_module="${repo_dir}/src/beyond_entropy/backbone_replication_decision.py"
decision_cli="${repo_dir}/scripts/decide_backbone_replication.py"
output_dir="${root}/decision"

protocol_sha256=1cd70d11168e12a2855ec01e8a869d89b82c4e87c3d864c566ed7db02bb61474
decision_contract_sha256=b03a830b6af1e6b4d8f2e7ca99a52cc0a0eeba0df01a0dd8334ca8b73ec934de
decision_module_sha256=a69f4b098a2e3a7879728085b5efd1e2d68e90c941f690240a616cd9b0a48486
decision_cli_sha256=08111d528284bb18cc422d5f6113e11bcd869b741445271d807b010c87abd6fd

sha256_of() {
  sha256sum "$1" | cut -d ' ' -f 1
}

if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before the frozen Qwen-7B decision" >&2
  exit 2
fi
for path in "${report}" "${analysis_completion}" "${protocol}" \
  "${decision_contract}" "${decision_module}" "${decision_cli}"; do
  if [[ ! -s "${path}" ]]; then
    echo "missing frozen Qwen-7B decision input: ${path}" >&2
    exit 2
  fi
done
if [[ "$(sha256_of "${protocol}")" != "${protocol_sha256}" \
  || "$(sha256_of "${decision_contract}")" != "${decision_contract_sha256}" \
  || "$(sha256_of "${decision_module}")" != "${decision_module_sha256}" \
  || "$(sha256_of "${decision_cli}")" != "${decision_cli_sha256}" ]]; then
  echo "frozen Qwen-7B decision implementation hash mismatch" >&2
  exit 2
fi
report_sha256=$(sha256_of "${report}")
if [[ "$(jq -r '.schema' "${analysis_completion}")" != visual_action_proxy_outcome_audit_completion_v1 \
  || "$(jq -r '.report_sha256' "${analysis_completion}")" != "${report_sha256}" \
  || "$(jq -r '.protocol_sha256' "${analysis_completion}")" != "${protocol_sha256}" \
  || "$(jq -r '.study_label' "${analysis_completion}")" != "ScreenQA Qwen2.5-VL-7B opened development" ]]; then
  echo "Qwen-7B analysis completion does not bind the frozen report" >&2
  exit 2
fi
if [[ -e "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite an existing Qwen-7B decision" >&2
  exit 2
fi

code_revision=$(git -C "${repo_dir}" rev-parse HEAD)
export PYTHONPATH="${repo_dir}/src"
"${python_bin}" "${decision_cli}" \
  --report "${report}" \
  --protocol "${protocol}" \
  --output-dir "${output_dir}" \
  --expected-report-sha256 "${report_sha256}" \
  --expected-protocol-sha256 "${protocol_sha256}" \
  --expected-study-label "ScreenQA Qwen2.5-VL-7B opened development" \
  --code-revision "${code_revision}"

decision="${output_dir}/decision.json"
completion="${output_dir}/decision.complete.json"
if [[ "$(jq -r '.schema' "${decision}")" != visual_action_backbone_replication_decision_v1 \
  || "$(jq -r '.selection.score_threshold_selected' "${decision}")" != false \
  || "$(jq -r '.selection.call_rate_selected' "${decision}")" != false \
  || "$(jq -r '.selection.protected_outcome_used' "${decision}")" != false \
  || "$(jq -r '.schema' "${completion}")" != visual_action_backbone_replication_completion_v1 \
  || "$(jq -r '.report_sha256' "${completion}")" != "${report_sha256}" ]]; then
  echo "Qwen-7B frozen decision output contract failed" >&2
  exit 2
fi
printf 'screenqa_qwen7b_decision=%s completion_sha256=%s\n' \
  "$(jq -r '.decision' "${decision}")" "$(sha256_of "${completion}")"
