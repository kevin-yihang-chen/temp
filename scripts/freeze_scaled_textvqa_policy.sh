#!/usr/bin/env bash

set -euo pipefail

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
primary_dir="${repo_dir}/artifacts/textvqa-train-scale-v1/pairwise-attention-primary-v1"
output="${primary_dir}/policy-freeze.json"

export PYTHONPATH="${repo_dir}/src"
cd "${repo_dir}"
"${python_bin}" scripts/freeze_scaled_textvqa_policy.py \
  --ranker-model "${primary_dir}/ranker-development/model.json" \
  --ranker-report "${primary_dir}/ranker-development/report.json" \
  --calibrated-model "${primary_dir}/risk-calibrated/model.json" \
  --calibration-report "${primary_dir}/risk-calibrated/calibration.json" \
  --ranker-rollouts "${repo_dir}/artifacts/textvqa-train-scale-v1/ranker-training/qwen3b-c4-seed0/rollouts.jsonl" \
  --ranker-features "${repo_dir}/artifacts/textvqa-train-scale-v1/ranker-training/attention-semantic-v1/features-question-region-attention-label-free.pt" \
  --calibration-rollouts "${repo_dir}/artifacts/textvqa-train-scale-v1/risk-calibration/qwen3b-c4-seed0/rollouts.jsonl" \
  --calibration-features "${repo_dir}/artifacts/textvqa-train-scale-v1/risk-calibration/attention-semantic-v1/features-question-region-attention-label-free.pt" \
  --protocol "${repo_dir}/docs/scaled_textvqa_risk_control_preregistration.md" \
  --output "${output}"
sha256sum "${output}"
