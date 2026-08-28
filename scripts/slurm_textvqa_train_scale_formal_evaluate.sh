#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-textvqa-train-scale-formal-evaluate-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

: "${BE_FORMAL_MODEL_SHA256:?missing BE_FORMAL_MODEL_SHA256}"
: "${BE_FORMAL_MANIFEST_SHA256:?missing BE_FORMAL_MANIFEST_SHA256}"
: "${BE_FORMAL_ROLLOUTS_SHA256:?missing BE_FORMAL_ROLLOUTS_SHA256}"
: "${BE_FORMAL_FEATURES_SHA256:?missing BE_FORMAL_FEATURES_SHA256}"
: "${BE_FORMAL_PROTOCOL_SHA256:?missing BE_FORMAL_PROTOCOL_SHA256}"
: "${BE_FORMAL_EVALUATOR_MODULE_SHA256:?missing BE_FORMAL_EVALUATOR_MODULE_SHA256}"
: "${BE_FORMAL_EVALUATOR_SCRIPT_SHA256:?missing BE_FORMAL_EVALUATOR_SCRIPT_SHA256}"
: "${BE_FORMAL_EXPECTED_STATES:?missing BE_FORMAL_EXPECTED_STATES}"
: "${BE_FORMAL_POLICY_FREEZE_SHA256:?missing BE_FORMAL_POLICY_FREEZE_SHA256}"
: "${BE_FORMAL_AUDIT_SHA256:?missing BE_FORMAL_AUDIT_SHA256}"

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
model="${repo_dir}/artifacts/textvqa-train-scale-v1/pairwise-attention-primary-v1/risk-calibrated/model.json"
policy_freeze="${repo_dir}/artifacts/textvqa-train-scale-v1/pairwise-attention-primary-v1/policy-freeze.json"
manifest="${repo_dir}/data/textvqa-train-scale-v1/formal-test/manifest.jsonl"
audit="${repo_dir}/data/textvqa-train-scale-v1/formal-test.audit.json"
rollouts="${repo_dir}/artifacts/textvqa-train-scale-v1/formal-test/qwen3b-c4-seed0/rollouts.jsonl"
feature_dir="${repo_dir}/artifacts/textvqa-train-scale-v1/formal-test/attention-semantic-v1"
features="${feature_dir}/features-question-region-attention-label-free.pt"
protocol="${repo_dir}/docs/scaled_textvqa_risk_control_preregistration.md"
output="${repo_dir}/artifacts/textvqa-train-scale-v1/pairwise-attention-primary-v1/formal-evaluation.json"
scientific_status="scaled TextVQA one-shot formal sibling bank; frozen risk-controlled policy; no target-derived tuning"

export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${repo_dir}"
test -s "${feature_dir}/rollouts.audit.json"
test -s "${feature_dir}/label-free-audit.json"
if [[ -e "${output}" ]]; then
  echo "One-shot formal evaluation output already exists" >&2
  exit 2
fi
"${python_bin}" scripts/verify_scaled_textvqa_formal_gate.py \
  --policy-freeze "${policy_freeze}" \
  --expected-policy-freeze-sha256 "${BE_FORMAL_POLICY_FREEZE_SHA256}" \
  --model "${model}" \
  --expected-model-sha256 "${BE_FORMAL_MODEL_SHA256}" \
  --manifest "${manifest}" \
  --expected-manifest-sha256 "${BE_FORMAL_MANIFEST_SHA256}" \
  --audit "${audit}" \
  --expected-audit-sha256 "${BE_FORMAL_AUDIT_SHA256}"
"${python_bin}" scripts/evaluate_scaled_textvqa_action_value.py \
  --model "${model}" \
  --expected-model-sha256 "${BE_FORMAL_MODEL_SHA256}" \
  --manifest "${manifest}" \
  --expected-manifest-sha256 "${BE_FORMAL_MANIFEST_SHA256}" \
  --rollouts "${rollouts}" \
  --expected-rollouts-sha256 "${BE_FORMAL_ROLLOUTS_SHA256}" \
  --features "${features}" \
  --expected-features-sha256 "${BE_FORMAL_FEATURES_SHA256}" \
  --expected-states "${BE_FORMAL_EXPECTED_STATES}" \
  --expected-scientific-status "${scientific_status}" \
  --protocol "${protocol}" \
  --expected-protocol-sha256 "${BE_FORMAL_PROTOCOL_SHA256}" \
  --expected-evaluator-module-sha256 "${BE_FORMAL_EVALUATOR_MODULE_SHA256}" \
  --expected-evaluator-script-sha256 "${BE_FORMAL_EVALUATOR_SCRIPT_SHA256}" \
  --output "${output}" \
  --bootstrap-resamples 20000 \
  --bootstrap-confidence 0.975 \
  --bootstrap-seed 20260828
