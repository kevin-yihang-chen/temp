#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-docvqa-fv2-formal-evaluate-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_DOCVQA_POLICY_FREEZE BE_DOCVQA_POLICY_FREEZE_SHA256 BE_DOCVQA_MODEL
  BE_DOCVQA_MODEL_SHA256 BE_DOCVQA_MANIFEST BE_DOCVQA_MANIFEST_SHA256
  BE_DOCVQA_MANIFEST_PROVENANCE BE_DOCVQA_MANIFEST_PROVENANCE_SHA256
  BE_DOCVQA_FORMAL_AUDIT BE_DOCVQA_FORMAL_AUDIT_SHA256
  BE_DOCVQA_EXPECTED_STATES BE_DOCVQA_ROLLOUTS BE_DOCVQA_FEATURE_DIR
  BE_DOCVQA_REPORT BE_DOCVQA_EXPECTED_CODE_REVISION
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
rollout_audit="${BE_DOCVQA_FEATURE_DIR}/rollouts.audit.json"
features="${BE_DOCVQA_FEATURE_DIR}/features-question-region-attention-label-free.pt"
rendered_report="${BE_DOCVQA_REPORT%.json}.md"

if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_DOCVQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "DocVQA formal evaluator revision differs from policy freeze" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before DocVQA formal evaluation" >&2
  exit 2
fi
if [[ -e "${BE_DOCVQA_REPORT}" || -e "${rendered_report}" ]]; then
  echo "refusing to overwrite DocVQA one-shot formal result" >&2
  exit 2
fi
for path in "${BE_DOCVQA_ROLLOUTS}" "${rollout_audit}" "${features}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing completed DocVQA formal input: ${path}" >&2
    exit 2
  fi
done

rollouts_sha256=$(sha256sum "${BE_DOCVQA_ROLLOUTS}")
rollouts_sha256=${rollouts_sha256%% *}
rollout_audit_sha256=$(sha256sum "${rollout_audit}")
rollout_audit_sha256=${rollout_audit_sha256%% *}
features_sha256=$(sha256sum "${features}")
features_sha256=${features_sha256%% *}
export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${repo_dir}"
"${python_bin}" scripts/evaluate_docvqa_train_factorized_v2_formal.py \
  --policy-freeze "${BE_DOCVQA_POLICY_FREEZE}" \
  --expected-policy-freeze-sha256 "${BE_DOCVQA_POLICY_FREEZE_SHA256}" \
  --model "${BE_DOCVQA_MODEL}" \
  --expected-model-sha256 "${BE_DOCVQA_MODEL_SHA256}" \
  --manifest "${BE_DOCVQA_MANIFEST}" \
  --expected-manifest-sha256 "${BE_DOCVQA_MANIFEST_SHA256}" \
  --manifest-provenance "${BE_DOCVQA_MANIFEST_PROVENANCE}" \
  --expected-manifest-provenance-sha256 "${BE_DOCVQA_MANIFEST_PROVENANCE_SHA256}" \
  --formal-audit "${BE_DOCVQA_FORMAL_AUDIT}" \
  --expected-formal-audit-sha256 "${BE_DOCVQA_FORMAL_AUDIT_SHA256}" \
  --rollouts "${BE_DOCVQA_ROLLOUTS}" \
  --expected-rollouts-sha256 "${rollouts_sha256}" \
  --rollout-audit "${rollout_audit}" \
  --expected-rollout-audit-sha256 "${rollout_audit_sha256}" \
  --features "${features}" \
  --expected-features-sha256 "${features_sha256}" \
  --expected-states "${BE_DOCVQA_EXPECTED_STATES}" \
  --output "${BE_DOCVQA_REPORT}" \
  --bootstrap-resamples 20000 \
  --bootstrap-confidence 0.975 \
  --bootstrap-seed 20260829

report_sha256=$(sha256sum "${BE_DOCVQA_REPORT}")
report_sha256=${report_sha256%% *}
"${python_bin}" scripts/render_docvqa_train_factorized_v2_formal.py \
  --report "${BE_DOCVQA_REPORT}" \
  --expected-report-sha256 "${report_sha256}" \
  --output "${rendered_report}"
