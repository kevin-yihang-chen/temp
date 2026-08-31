#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --job-name=be-docvqa-reserve
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-docvqa-reserve-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_RESERVE_FREEZE BE_RESERVE_FREEZE_SHA256 BE_RESERVE_EXPECTED_CODE_REVISION
  BE_RESERVE_MANIFEST BE_RESERVE_MANIFEST_AUDIT BE_RESERVE_RUN_ROOT
  BE_RESERVE_FEATURE_DIR BE_RESERVE_SCORE_DIR BE_RESERVE_RESULT
  BE_RESERVE_RESUME
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then echo "missing ${name}" >&2; exit 2; fi
done
if [[ "${BE_RESERVE_RESUME}" != 0 && "${BE_RESERVE_RESUME}" != 1 ]]; then
  echo "BE_RESERVE_RESUME must be 0 or 1" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
model_revision=66285546d2b821cf421d4f5eb2576359d3770cd3
scientific_status="one-shot outcome-sealed DocVQA reserve ToolGate comparator sibling bank"
rollout_preparer="${repo_dir}/artifacts/docvqa-train-factorized-v2/ops/prepare_docvqa_formal_multigpu_shards.py"
rollout_merger="${repo_dir}/artifacts/docvqa-train-factorized-v2/ops/merge_docvqa_formal_multigpu_rollouts.py"
feature_preparer="${repo_dir}/scripts/prepare_semantic_feature_batch_shards.py"
feature_merger="${repo_dir}/scripts/merge_semantic_feature_shards.py"
rollout_plan_dir="${BE_RESERVE_RUN_ROOT}/multigpu-plan"
rollout_plan="${rollout_plan_dir}/plan.json"
rollout_shards="${BE_RESERVE_RUN_ROOT}/shards"
rollouts="${BE_RESERVE_RUN_ROOT}/rollouts.jsonl"
feature_shard_root="${BE_RESERVE_FEATURE_DIR}/multigpu-batch-aligned-shards"
feature_plan_dir="${feature_shard_root}/plan"
feature_plan="${feature_plan_dir}/plan.json"
base_features="${BE_RESERVE_FEATURE_DIR}/features-label-free.pt"
multimodal_features="${BE_RESERVE_FEATURE_DIR}/features-multimodal-label-free.pt"
attention_features="${BE_RESERVE_FEATURE_DIR}/features-question-region-attention-label-free.pt"
label_free_audit="${BE_RESERVE_FEATURE_DIR}/label-free-audit.json"
scores="${BE_RESERVE_SCORE_DIR}/policy-scores.jsonl"
score_report="${BE_RESERVE_SCORE_DIR}/score-report.json"

if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_RESERVE_EXPECTED_CODE_REVISION}" ]]; then
  echo "reserve pipeline code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before reserve pipeline" >&2
  exit 2
fi
if [[ -e "${BE_RESERVE_RESULT}" ]]; then
  echo "refusing to reuse one-shot reserve result" >&2
  exit 2
fi
if [[ "${BE_RESERVE_RESUME}" != 1 ]]; then
  for path in "${BE_RESERVE_RUN_ROOT}" "${BE_RESERVE_FEATURE_DIR}" "${BE_RESERVE_SCORE_DIR}"; do
    if [[ -d "${path}" && -n "$(find "${path}" -mindepth 1 -print -quit)" ]]; then
      echo "existing reserve partial outputs require audited resume: ${path}" >&2
      exit 2
    fi
  done
fi

sha256_of() { sha256sum "$1" | cut -d ' ' -f 1; }
manifest_sha256=$(sha256_of "${BE_RESERVE_MANIFEST}")
audit_sha256=$(sha256_of "${BE_RESERVE_MANIFEST_AUDIT}")
export PYTHONPATH="${repo_dir}/src"
export BE_CODE_REVISION="${BE_RESERVE_EXPECTED_CODE_REVISION}"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "${repo_dir}"
"${python_bin}" scripts/verify_docvqa_reserve_gate.py \
  --freeze "${BE_RESERVE_FREEZE}" \
  --expected-freeze-sha256 "${BE_RESERVE_FREEZE_SHA256}" \
  --manifest "${BE_RESERVE_MANIFEST}" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --audit "${BE_RESERVE_MANIFEST_AUDIT}" \
  --expected-audit-sha256 "${audit_sha256}"

mkdir -p "${rollout_plan_dir}" "${rollout_shards}"
rollout_prepare_args=(
  --manifest "${BE_RESERVE_MANIFEST}"
  --expected-manifest-sha256 "${manifest_sha256}"
  --output-dir "${rollout_plan_dir}"
  --shard-count 4
  --seed 20260829
  --formal-scientific-status "${scientific_status}"
)
if [[ -e "${rollout_plan}" ]]; then rollout_prepare_args+=(--resume); fi
"${python_bin}" "${rollout_preparer}" "${rollout_prepare_args[@]}"
if [[ "$(jq -r '.outcome_fields_used_for_assignment | length' "${rollout_plan}")" -ne 0 \
  || "$(jq -r '.source_count' "${rollout_plan}")" -ne 688 \
  || "$(jq -r '.shard_count' "${rollout_plan}")" -ne 4 ]]; then
  echo "reserve rollout shard plan contract failed" >&2
  exit 2
fi

run_rollout_shard() {
  local index=$1 manifest manifest_hash output_dir
  manifest=$(jq -r --argjson index "${index}" '.shards[] | select(.shard_index == $index) | .manifest' "${rollout_plan}")
  manifest_hash=$(jq -r --argjson index "${index}" '.shards[] | select(.shard_index == $index) | .manifest_sha256' "${rollout_plan}")
  output_dir=$(printf '%s/shard-%02d' "${rollout_shards}" "${index}")
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${index}" "${python_bin}" -m beyond_entropy collect-qwen \
    --manifest "${manifest}" \
    --expected-manifest-sha256 "${manifest_hash}" \
    --output "${output_dir}/rollouts.jsonl" \
    --resume \
    --checkpoint-interval 32 \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --model-revision "${model_revision}" \
    --scorer docvqa \
    --candidate-count 4 \
    --proposer ug-grid \
    --visual-crop-ratio 2.0 \
    --visual-cost 1.0 \
    --generation-seeds 0 \
    --bootstrap-resamples 100 \
    --bootstrap-seed 20260829 \
    --scientific-status "${scientific_status}" \
    --max-new-tokens 32 \
    --min-pixels 200704 \
    --max-pixels 602112 \
    --attention-implementation sdpa \
    --system-prompt "You are a helpful assistant."
}

run_parallel() {
  local function_name=$1 label=$2
  local pids=() failed=0 index pid
  for index in 0 1 2 3; do "${function_name}" "${index}" & pids+=("$!"); done
  for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
  if [[ "${failed}" -ne 0 ]]; then echo "reserve ${label} shard failed" >&2; return 1; fi
}

run_parallel run_rollout_shard rollout
rollout_plan_sha256=$(sha256_of "${rollout_plan}")
rollout_merge_args=(
  --manifest "${BE_RESERVE_MANIFEST}"
  --expected-manifest-sha256 "${manifest_sha256}"
  --plan "${rollout_plan}"
  --expected-plan-sha256 "${rollout_plan_sha256}"
  --shard-output-root "${rollout_shards}"
  --output "${rollouts}"
  --expected-code-revision "${BE_RESERVE_EXPECTED_CODE_REVISION}"
  --expected-model-revision "${model_revision}"
  --expected-scientific-status "${scientific_status}"
)
if [[ -e "${rollouts}" ]]; then rollout_merge_args+=(--resume); fi
"${python_bin}" "${rollout_merger}" "${rollout_merge_args[@]}"
rollouts_sha256=$(sha256_of "${rollouts}")

mkdir -p "${BE_RESERVE_FEATURE_DIR}" "${feature_plan_dir}"
feature_prepare_args=(
  --rollouts "${rollouts}"
  --expected-rollouts-sha256 "${rollouts_sha256}"
  --output-dir "${feature_plan_dir}"
  --shards 4
  --batch-size 4
  --expected-code-revision "${BE_RESERVE_EXPECTED_CODE_REVISION}"
  --expected-candidate-count 4
)
if [[ -e "${feature_plan}" ]]; then feature_prepare_args+=(--resume); fi
"${python_bin}" "${feature_preparer}" "${feature_prepare_args[@]}"
if [[ "$(jq -r '.assignment_outcome_fields_used' "${feature_plan}")" != false \
  || "$(jq -r '.shard_count' "${feature_plan}")" -ne 4 ]]; then
  echo "reserve semantic shard plan contract failed" >&2
  exit 2
fi

feature_rollout() { printf '%s/rollouts.shard-%02d.jsonl' "${feature_plan_dir}" "$1"; }
feature_dir() { printf '%s/shard-%02d' "${feature_shard_root}" "$1"; }
run_base() {
  local index=$1 input output_dir output input_hash resume_args=()
  input=$(feature_rollout "${index}"); output_dir=$(feature_dir "${index}")
  output="${output_dir}/features-label-free.pt"; input_hash=$(sha256_of "${input}")
  mkdir -p "${output_dir}"; if [[ -e "${output}" ]]; then resume_args=(--resume); fi
  CUDA_VISIBLE_DEVICES="${index}" "${python_bin}" -m beyond_entropy extract-qwen-features \
    --rollouts "${input}" --expected-rollouts-sha256 "${input_hash}" --output "${output}" \
    --model Qwen/Qwen2.5-VL-3B-Instruct --model-revision "${model_revision}" \
    --min-pixels 200704 --max-pixels 602112 --attention-implementation sdpa \
    --question-feature-mode input_mean --checkpoint-interval 32 --exclude-outcomes "${resume_args[@]}"
}
run_multimodal() {
  local index=$1 input output_dir source output resume_args=()
  input=$(feature_rollout "${index}"); output_dir=$(feature_dir "${index}")
  source="${output_dir}/features-label-free.pt"; output="${output_dir}/features-multimodal-label-free.pt"
  if [[ -e "${output}" ]]; then resume_args=(--resume); fi
  CUDA_VISIBLE_DEVICES="${index}" "${python_bin}" scripts/reembed_contextual_questions.py \
    --source-features "${source}" --rollouts "${input}" --output "${output}" \
    --model Qwen/Qwen2.5-VL-3B-Instruct --revision "${model_revision}" --device-map cuda:0 \
    --dtype bfloat16 --attention-implementation sdpa --mode multimodal-original \
    --batch-size 4 --checkpoint-interval 64 "${resume_args[@]}"
}
run_attention() {
  local index=$1 input output_dir source output resume_args=()
  input=$(feature_rollout "${index}"); output_dir=$(feature_dir "${index}")
  source="${output_dir}/features-multimodal-label-free.pt"; output="${output_dir}/features-question-region-attention-label-free.pt"
  if [[ -e "${output}" ]]; then resume_args=(--resume); fi
  CUDA_VISIBLE_DEVICES="${index}" "${python_bin}" scripts/extract_question_region_attention.py \
    --source-features "${source}" --rollouts "${input}" --output "${output}" \
    --model Qwen/Qwen2.5-VL-3B-Instruct --revision "${model_revision}" --device-map cuda:0 \
    --dtype bfloat16 --top-layers 4 --checkpoint-interval 32 "${resume_args[@]}"
}
merge_features() {
  local stage=$1 shard_name=$2 output=$3 report=$4 source=${5:-}
  local args=(--stage "${stage}" --full-rollouts "${rollouts}" --expected-full-rollouts-sha256 "${rollouts_sha256}" --expected-code-revision "${BE_RESERVE_EXPECTED_CODE_REVISION}" --output "${output}" --report "${report}")
  local index plan_hash source_hash
  plan_hash=$(sha256_of "${feature_plan}")
  args+=(--shard-plan "${feature_plan}" --expected-shard-plan-sha256 "${plan_hash}")
  for index in 0 1 2 3; do args+=(--shard-rollouts "$(feature_rollout "${index}")"); done
  for index in 0 1 2 3; do args+=(--shard-features "$(feature_dir "${index}")/${shard_name}"); done
  if [[ -n "${source}" ]]; then source_hash=$(sha256_of "${source}"); args+=(--source-features "${source}" --expected-source-features-sha256 "${source_hash}"); fi
  if [[ -e "${output}" || -e "${report}" ]]; then args+=(--resume); fi
  "${python_bin}" "${feature_merger}" "${args[@]}"
}

run_parallel run_base base-feature
merge_features base features-label-free.pt "${base_features}" "${feature_shard_root}/merge-base.json"
run_parallel run_multimodal multimodal-feature
merge_features multimodal features-multimodal-label-free.pt "${multimodal_features}" "${feature_shard_root}/merge-multimodal.json" "${base_features}"
run_parallel run_attention attention-feature
merge_features attention features-question-region-attention-label-free.pt "${attention_features}" "${feature_shard_root}/merge-attention.json" "${multimodal_features}"

"${python_bin}" scripts/audit_label_free_semantic_features.py \
  --features "${attention_features}" --rollouts "${rollouts}" > "${label_free_audit}.tmp"
if [[ -e "${label_free_audit}" ]]; then
  cmp -s "${label_free_audit}.tmp" "${label_free_audit}" || { echo "reserve label-free audit changed" >&2; exit 2; }
  rm "${label_free_audit}.tmp"
else
  mv "${label_free_audit}.tmp" "${label_free_audit}"
fi
features_sha256=$(sha256_of "${attention_features}")
mkdir -p "${BE_RESERVE_SCORE_DIR}" "$(dirname "${BE_RESERVE_RESULT}")"
if [[ -e "${scores}" || -e "${score_report}" ]]; then
  if [[ "${BE_RESERVE_RESUME}" != 1 || ! -s "${scores}" || ! -s "${score_report}" ]]; then
    echo "reserve policy scores are partial or resume is disabled" >&2
    exit 2
  fi
else
  "${python_bin}" scripts/score_docvqa_reserve_toolgate.py \
    --freeze "${BE_RESERVE_FREEZE}" --expected-freeze-sha256 "${BE_RESERVE_FREEZE_SHA256}" \
    --manifest "${BE_RESERVE_MANIFEST}" --expected-manifest-sha256 "${manifest_sha256}" \
    --reserve-audit "${BE_RESERVE_MANIFEST_AUDIT}" --expected-reserve-audit-sha256 "${audit_sha256}" \
    --rollouts "${rollouts}" --expected-rollouts-sha256 "${rollouts_sha256}" \
    --features "${attention_features}" --expected-features-sha256 "${features_sha256}" \
    --scores-output "${scores}" --report-output "${score_report}"
fi
scores_sha256=$(sha256_of "${scores}")
score_report_sha256=$(sha256_of "${score_report}")
"${python_bin}" scripts/evaluate_docvqa_reserve_toolgate.py \
  --freeze "${BE_RESERVE_FREEZE}" --expected-freeze-sha256 "${BE_RESERVE_FREEZE_SHA256}" \
  --rollouts "${rollouts}" --expected-rollouts-sha256 "${rollouts_sha256}" \
  --scores "${scores}" --expected-scores-sha256 "${scores_sha256}" \
  --score-report "${score_report}" --expected-score-report-sha256 "${score_report_sha256}" \
  --output "${BE_RESERVE_RESULT}" --bootstrap-resamples 20000 --bootstrap-seed 20260829
printf 'reserve_result_sha256=%s\n' "$(sha256_of "${BE_RESERVE_RESULT}")"
