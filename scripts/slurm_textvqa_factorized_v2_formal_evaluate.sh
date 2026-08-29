#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-textvqa-fv2-formal-evaluate-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_FV2_POLICY_FREEZE BE_FV2_POLICY_FREEZE_SHA256 BE_FV2_MODEL
  BE_FV2_MODEL_SHA256 BE_FV2_MANIFEST BE_FV2_MANIFEST_SHA256
  BE_FV2_MANIFEST_PROVENANCE BE_FV2_MANIFEST_PROVENANCE_SHA256
  BE_FV2_FORMAL_AUDIT BE_FV2_FORMAL_AUDIT_SHA256 BE_FV2_EXPECTED_STATES
  BE_FV2_ROLLOUTS BE_FV2_FEATURE_DIR BE_FV2_SCIENTIFIC_STATUS
  BE_FV2_REPORT BE_CODE_REVISION
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
rollout_audit="${BE_FV2_FEATURE_DIR}/rollouts.audit.json"
features="${BE_FV2_FEATURE_DIR}/features-question-region-attention-label-free.pt"

if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_CODE_REVISION}" ]]; then
  echo "formal evaluator revision differs from the frozen submission" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before formal evaluation" >&2
  exit 2
fi
if [[ -e "${BE_FV2_REPORT}" ]]; then
  echo "refusing to overwrite one-shot formal report" >&2
  exit 2
fi
rendered_report="${BE_FV2_REPORT%.json}.md"
if [[ -e "${rendered_report}" ]]; then
  echo "refusing to overwrite rendered formal report" >&2
  exit 2
fi

rollouts_sha256=$(sha256sum "${BE_FV2_ROLLOUTS}")
rollouts_sha256=${rollouts_sha256%% *}
rollout_audit_sha256=$(sha256sum "${rollout_audit}")
rollout_audit_sha256=${rollout_audit_sha256%% *}
features_sha256=$(sha256sum "${features}")
features_sha256=${features_sha256%% *}

export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${repo_dir}"
"${python_bin}" scripts/evaluate_factorized_textvqa_formal.py \
  --policy-freeze "${BE_FV2_POLICY_FREEZE}" \
  --expected-policy-freeze-sha256 "${BE_FV2_POLICY_FREEZE_SHA256}" \
  --model "${BE_FV2_MODEL}" \
  --expected-model-sha256 "${BE_FV2_MODEL_SHA256}" \
  --manifest "${BE_FV2_MANIFEST}" \
  --expected-manifest-sha256 "${BE_FV2_MANIFEST_SHA256}" \
  --manifest-provenance "${BE_FV2_MANIFEST_PROVENANCE}" \
  --expected-manifest-provenance-sha256 "${BE_FV2_MANIFEST_PROVENANCE_SHA256}" \
  --formal-audit "${BE_FV2_FORMAL_AUDIT}" \
  --expected-formal-audit-sha256 "${BE_FV2_FORMAL_AUDIT_SHA256}" \
  --rollouts "${BE_FV2_ROLLOUTS}" \
  --expected-rollouts-sha256 "${rollouts_sha256}" \
  --rollout-audit "${rollout_audit}" \
  --expected-rollout-audit-sha256 "${rollout_audit_sha256}" \
  --features "${features}" \
  --expected-features-sha256 "${features_sha256}" \
  --expected-states "${BE_FV2_EXPECTED_STATES}" \
  --expected-scientific-status "${BE_FV2_SCIENTIFIC_STATUS}" \
  --output "${BE_FV2_REPORT}" \
  --bootstrap-resamples 20000 \
  --bootstrap-confidence 0.975 \
  --bootstrap-seed 20260828

report_sha256=$(sha256sum "${BE_FV2_REPORT}")
report_sha256=${report_sha256%% *}
"${python_bin}" scripts/render_factorized_textvqa_formal.py \
  --report "${BE_FV2_REPORT}" \
  --expected-report-sha256 "${report_sha256}" \
  --output "${rendered_report}"
