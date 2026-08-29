#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-textvqa-fv2-formal-rollout-%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_FV2_POLICY_FREEZE BE_FV2_POLICY_FREEZE_SHA256 BE_FV2_MODEL
  BE_FV2_MODEL_SHA256 BE_FV2_MANIFEST BE_FV2_MANIFEST_SHA256
  BE_FV2_MANIFEST_PROVENANCE BE_FV2_MANIFEST_PROVENANCE_SHA256
  BE_FV2_FORMAL_AUDIT BE_FV2_FORMAL_AUDIT_SHA256 BE_FV2_EXPECTED_STATES
  BE_FV2_RUN_DIR BE_FV2_SCIENTIFIC_STATUS BE_CODE_REVISION
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
rollouts="${BE_FV2_RUN_DIR}/rollouts.jsonl"

if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_CODE_REVISION}" ]]; then
  echo "formal rollout revision differs from the frozen submission" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before formal rollout" >&2
  exit 2
fi
export PYTHONPATH="${repo_dir}/src"
/userhome/cs3/yihangc/anaconda3/bin/python \
  "${repo_dir}/scripts/verify_factorized_v2_formal_gate.py" \
  --policy-freeze "${BE_FV2_POLICY_FREEZE}" \
  --expected-policy-freeze-sha256 "${BE_FV2_POLICY_FREEZE_SHA256}" \
  --model "${BE_FV2_MODEL}" \
  --expected-model-sha256 "${BE_FV2_MODEL_SHA256}" \
  --manifest "${BE_FV2_MANIFEST}" \
  --expected-manifest-sha256 "${BE_FV2_MANIFEST_SHA256}" \
  --manifest-provenance "${BE_FV2_MANIFEST_PROVENANCE}" \
  --expected-manifest-provenance-sha256 "${BE_FV2_MANIFEST_PROVENANCE_SHA256}" \
  --audit "${BE_FV2_FORMAL_AUDIT}" \
  --expected-audit-sha256 "${BE_FV2_FORMAL_AUDIT_SHA256}"

if [[ "$(wc -l < "${BE_FV2_MANIFEST}")" -ne "${BE_FV2_EXPECTED_STATES}" ]]; then
  echo "formal manifest state count mismatch" >&2
  exit 2
fi
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "${repo_dir}"
mkdir -p "${BE_FV2_RUN_DIR}"
"${python_bin}" -m beyond_entropy collect-qwen \
  --manifest "${BE_FV2_MANIFEST}" \
  --expected-manifest-sha256 "${BE_FV2_MANIFEST_SHA256}" \
  --output "${rollouts}" \
  --resume \
  --checkpoint-interval 32 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --scorer textvqa \
  --candidate-count 4 \
  --proposer ug-grid \
  --visual-crop-ratio 2.0 \
  --visual-cost 1.0 \
  --generation-seeds 0 \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 20260828 \
  --scientific-status "${BE_FV2_SCIENTIFIC_STATUS}" \
  --max-new-tokens 32 \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --attention-implementation sdpa \
  --system-prompt "You are a helpful assistant."
