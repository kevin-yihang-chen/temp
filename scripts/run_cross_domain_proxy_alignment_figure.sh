#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/bin/python
screenqa_report="${repo_dir}/artifacts/screenqa-train-factorized-v1/proxy-to-outcome-audit-v1/full-v1/analysis/report.json"
screenqa_report_sha256=438a5e64072826480aa41a5ccf78224bb4b8191a04976bf26760f4278181276a
docvqa_root="${repo_dir}/artifacts/docvqa-train-factorized-v2/proxy-to-outcome-cross-domain-v1/full-v1"
docvqa_report="${docvqa_root}/analysis/report.json"
docvqa_audit_completion="${docvqa_root}/analysis/audit.complete.json"
decision="${docvqa_root}/decision/decision.json"
decision_completion="${docvqa_root}/decision/decision.complete.json"
output_dir="${repo_dir}/artifacts/docvqa-train-factorized-v2/paper-assets/proxy-alignment-cross-domain-v1"
renderer="${repo_dir}/scripts/render_proxy_alignment_figure.py"
renderer_module="${repo_dir}/src/beyond_entropy/proxy_alignment_figure.py"
renderer_sha256=781e6d38fac011d68326253461e34ff42d48a646e8d0e0adc6ba6c019b35f891
renderer_module_sha256=d081431005abb7b28c8e2caaf18a00a4a2c3defaba931a4a11f9353145b4186f
figure_code_revision=7934df176d09da42983a9cb24762da7d19ff7349

check_hash() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "cross-domain proxy figure ${label} hash mismatch" >&2
    exit 2
  fi
}

if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before rendering the cross-domain figure" >&2
  exit 2
fi
for path in "${screenqa_report}" "${docvqa_report}" \
  "${docvqa_audit_completion}" "${decision}" "${decision_completion}" \
  "${renderer}" "${renderer_module}"; do
  if [[ ! -s "${path}" ]]; then
    echo "missing cross-domain proxy figure input: ${path}" >&2
    exit 2
  fi
done
check_hash "${screenqa_report}" "${screenqa_report_sha256}" "ScreenQA report"
check_hash "${renderer}" "${renderer_sha256}" renderer
check_hash "${renderer_module}" "${renderer_module_sha256}" "renderer module"

docvqa_report_sha256=$(sha256sum "${docvqa_report}" | cut -d ' ' -f 1)
if [[ "$(jq -r '.report_sha256' "${docvqa_audit_completion}")" != "${docvqa_report_sha256}" ]]; then
  echo "DocVQA audit completion/report hash mismatch" >&2
  exit 2
fi
decision_sha256=$(sha256sum "${decision}" | cut -d ' ' -f 1)
decision_status=$(jq -r '.decision' "${decision}")
if [[ "$(jq -r '.decision_json_sha256' "${decision_completion}")" != "${decision_sha256}" \
  || "$(jq -r '.report_sha256' "${decision_completion}")" != "${docvqa_report_sha256}" \
  || "$(jq -r '.decision' "${decision_completion}")" != "${decision_status}" ]]; then
  echo "DocVQA decision completion contract mismatch" >&2
  exit 2
fi
if [[ "${decision_status}" != replicated_alignment \
  && "${decision_status}" != partial_alignment \
  && "${decision_status}" != non_replication ]]; then
  echo "invalid DocVQA replication decision" >&2
  exit 2
fi
if [[ "$(jq -r '.selection.score_threshold_selected' "${decision}")" != false \
  || "$(jq -r '.selection.call_rate_selected' "${decision}")" != false \
  || "$(jq -r '.selection.protected_outcome_used' "${decision}")" != false ]]; then
  echo "DocVQA decision selected a forbidden deployment parameter" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
"${python_bin}" "${renderer}" \
  --report "ScreenQA=${screenqa_report}" \
  --report "DocVQA=${docvqa_report}" \
  --expected-sha256 "ScreenQA=${screenqa_report_sha256}" \
  --expected-sha256 "DocVQA=${docvqa_report_sha256}" \
  --output-dir "${output_dir}" \
  --basename proxy-alignment-cross-domain-v1 \
  --code-revision "${figure_code_revision}"

provenance="${output_dir}/proxy-alignment-cross-domain-v1.provenance.json"
if [[ "$(jq -r '.reports | length' "${provenance}")" -ne 2 \
  || "$(jq -r '[.reports[].label] == ["ScreenQA", "DocVQA"]' "${provenance}")" != true \
  || "$(jq -r '.selection.threshold_selected' "${provenance}")" != false \
  || "$(jq -r '.selection.call_rate_selected' "${provenance}")" != false \
  || "$(jq -r '.selection.protected_outcome_used' "${provenance}")" != false \
  || "$(jq -r '.implementation.module_sha256' "${provenance}")" != "${renderer_module_sha256}" \
  || "$(jq -r '.implementation.cli_sha256' "${provenance}")" != "${renderer_sha256}" ]]; then
  echo "cross-domain proxy figure provenance contract failed" >&2
  exit 2
fi
printf 'cross_domain_proxy_figure_decision=%s provenance_sha256=%s docvqa_report_sha256=%s\n' \
  "${decision_status}" "$(sha256sum "${provenance}" | cut -d ' ' -f 1)" \
  "${docvqa_report_sha256}"
