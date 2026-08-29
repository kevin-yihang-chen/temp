#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
candidate="${repo_dir}/artifacts/textvqa-train-factorized-v2/frozen-candidate/model.json"
allocation="${repo_dir}/data/textvqa-train-factorized-v2/allocation.json"
allocation_audit="${repo_dir}/data/textvqa-train-factorized-v2/allocation.audit.json"
calibration_dir="${repo_dir}/artifacts/textvqa-train-factorized-v2/fixed-sequence-calibrated"
manifest_provenance="${repo_dir}/data/textvqa-train-factorized-v2/risk-calibration/manifest.provenance.json"
rollout_audit="${repo_dir}/artifacts/textvqa-train-factorized-v2/risk-calibration/attention-semantic-v1/rollouts.audit.json"
protocol="${repo_dir}/docs/textvqa_factorized_fixed_sequence_preregistration.md"

if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before policy freeze" >&2
  exit 2
fi
export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
"${python_bin}" scripts/freeze_factorized_v2_formal_policy.py \
  --candidate "${candidate}" \
  --allocation "${allocation}" \
  --allocation-audit "${allocation_audit}" \
  --calibration "${calibration_dir}/calibration.json" \
  --model "${calibration_dir}/model.json" \
  --manifest-provenance "${manifest_provenance}" \
  --rollout-audit "${rollout_audit}" \
  --protocol "${protocol}" \
  --output "${calibration_dir}/policy-freeze.json"
