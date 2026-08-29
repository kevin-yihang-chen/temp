#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-docvqa-fv2-formal-rollout-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_DOCVQA_POLICY_FREEZE BE_DOCVQA_POLICY_FREEZE_SHA256 BE_DOCVQA_MODEL
  BE_DOCVQA_MODEL_SHA256 BE_DOCVQA_MANIFEST BE_DOCVQA_MANIFEST_SHA256
  BE_DOCVQA_MANIFEST_PROVENANCE BE_DOCVQA_MANIFEST_PROVENANCE_SHA256
  BE_DOCVQA_FORMAL_AUDIT BE_DOCVQA_FORMAL_AUDIT_SHA256
  BE_DOCVQA_EXPECTED_STATES BE_DOCVQA_RUN_DIR
  BE_DOCVQA_EXPECTED_CODE_REVISION
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
rollouts="${BE_DOCVQA_RUN_DIR}/rollouts.jsonl"

if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_DOCVQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "DocVQA formal rollout revision differs from policy freeze" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before DocVQA formal rollout" >&2
  exit 2
fi
export PYTHONPATH="${repo_dir}/src"
"${python_bin}" "${repo_dir}/scripts/verify_docvqa_train_factorized_v2_formal_gate.py" \
  --policy-freeze "${BE_DOCVQA_POLICY_FREEZE}" \
  --expected-policy-freeze-sha256 "${BE_DOCVQA_POLICY_FREEZE_SHA256}" \
  --model "${BE_DOCVQA_MODEL}" \
  --expected-model-sha256 "${BE_DOCVQA_MODEL_SHA256}" \
  --manifest "${BE_DOCVQA_MANIFEST}" \
  --expected-manifest-sha256 "${BE_DOCVQA_MANIFEST_SHA256}" \
  --manifest-provenance "${BE_DOCVQA_MANIFEST_PROVENANCE}" \
  --expected-manifest-provenance-sha256 "${BE_DOCVQA_MANIFEST_PROVENANCE_SHA256}" \
  --audit "${BE_DOCVQA_FORMAL_AUDIT}" \
  --expected-audit-sha256 "${BE_DOCVQA_FORMAL_AUDIT_SHA256}"
if [[ "$(wc -l < "${BE_DOCVQA_MANIFEST}")" -ne "${BE_DOCVQA_EXPECTED_STATES}" ]]; then
  echo "DocVQA formal manifest state count mismatch" >&2
  exit 2
fi

export BE_CODE_REVISION="${BE_DOCVQA_EXPECTED_CODE_REVISION}"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "${repo_dir}"
mkdir -p "${BE_DOCVQA_RUN_DIR}"
"${python_bin}" -m beyond_entropy collect-qwen \
  --manifest "${BE_DOCVQA_MANIFEST}" \
  --expected-manifest-sha256 "${BE_DOCVQA_MANIFEST_SHA256}" \
  --output "${rollouts}" \
  --resume \
  --checkpoint-interval 32 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --scorer docvqa \
  --candidate-count 4 \
  --proposer ug-grid \
  --visual-crop-ratio 2.0 \
  --visual-cost 1.0 \
  --generation-seeds 0 \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 20260829 \
  --scientific-status "one-shot DocVQA-train factorized-v2 formal sibling bank; frozen policy and implementation; no target-derived tuning" \
  --max-new-tokens 32 \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --attention-implementation sdpa \
  --system-prompt "You are a helpful assistant."
