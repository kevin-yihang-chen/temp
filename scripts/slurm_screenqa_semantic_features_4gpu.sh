#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=be-screenqa-semantic-features
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-semantic-features-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_SCREENQA_CANDIDATE_DIR BE_SCREENQA_CANDIDATE_AUDIT_SHA256
  BE_SCREENQA_RANKER_ROLLOUTS BE_SCREENQA_RANKER_ROLLOUTS_SHA256
  BE_SCREENQA_RANKER_INPUT_AUDIT BE_SCREENQA_RANKER_INPUT_AUDIT_SHA256
  BE_SCREENQA_V1_PROTOCOL BE_SCREENQA_V1_PROTOCOL_SHA256
  BE_SCREENQA_V2_PROTOCOL BE_SCREENQA_V2_PROTOCOL_SHA256
  BE_SCREENQA_SEMANTIC_ACTIVATION BE_SCREENQA_SEMANTIC_ACTIVATION_SHA256
  BE_SCREENQA_SEMANTIC_FEATURE_DIR BE_SCREENQA_EXPECTED_CODE_REVISION
  BE_SCREENQA_SEMANTIC_RUNNER_SHA256 BE_SCREENQA_SHARD_PREPARER_SHA256
  BE_SCREENQA_SHARD_MERGER_SHA256 BE_SCREENQA_SEMANTIC_RESUME
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
runner="${repo_dir}/scripts/slurm_screenqa_semantic_features_4gpu.sh"
preparer="${repo_dir}/scripts/prepare_semantic_feature_batch_shards.py"
merger="${repo_dir}/scripts/merge_semantic_feature_shards.py"
model_revision=66285546d2b821cf421d4f5eb2576359d3770cd3
shard_root="${BE_SCREENQA_SEMANTIC_FEATURE_DIR}/multigpu-batch-aligned-shards"
plan_dir="${shard_root}/plan"
plan="${plan_dir}/plan.json"
base_features="${BE_SCREENQA_SEMANTIC_FEATURE_DIR}/features-label-free.pt"
multimodal_features="${BE_SCREENQA_SEMANTIC_FEATURE_DIR}/features-multimodal-label-free.pt"
attention_features="${BE_SCREENQA_SEMANTIC_FEATURE_DIR}/features-question-region-attention-label-free.pt"
label_free_audit="${BE_SCREENQA_SEMANTIC_FEATURE_DIR}/label-free-audit.json"
resume_mode="${BE_SCREENQA_SEMANTIC_RESUME}"
calibration_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/calibration-manifest-v1"
formal_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/formal-manifest-v1"
reserve_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/reserve-manifest-v1"
untouched_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/untouched-manifest-v1"

check_hash() {
  local path=$1
  local expected=$2
  local name=$3
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ScreenQA semantic ${name} SHA-256 mismatch" >&2
    exit 2
  fi
}

check_hash "${runner}" "${BE_SCREENQA_SEMANTIC_RUNNER_SHA256}" "runner"
check_hash "${preparer}" "${BE_SCREENQA_SHARD_PREPARER_SHA256}" "shard preparer"
check_hash "${merger}" "${BE_SCREENQA_SHARD_MERGER_SHA256}" "shard merger"
check_hash "${BE_SCREENQA_SEMANTIC_ACTIVATION}" "${BE_SCREENQA_SEMANTIC_ACTIVATION_SHA256}" "activation audit"
if [[ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${BE_SCREENQA_EXPECTED_CODE_REVISION}" ]]; then
  echo "ScreenQA semantic feature code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA semantic features" >&2
  exit 2
fi
if [[ "${resume_mode}" != 0 && "${resume_mode}" != 1 ]]; then
  echo "ScreenQA semantic resume flag must be 0 or 1" >&2
  exit 2
fi
if [[ -d "${BE_SCREENQA_SEMANTIC_FEATURE_DIR}" && "${resume_mode}" != 1 ]]; then
  if [[ -n "$(find "${BE_SCREENQA_SEMANTIC_FEATURE_DIR}" -mindepth 1 -print -quit)" ]]; then
    echo "existing ScreenQA semantic features require audited resume" >&2
    exit 2
  fi
fi

export PYTHONPATH="${repo_dir}/src"
export BE_CODE_REVISION="${BE_SCREENQA_EXPECTED_CODE_REVISION}"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "${repo_dir}"

activation_args=(
  --candidate-dir "${BE_SCREENQA_CANDIDATE_DIR}"
  --expected-candidate-audit-sha256 "${BE_SCREENQA_CANDIDATE_AUDIT_SHA256}"
  --ranker-rollouts "${BE_SCREENQA_RANKER_ROLLOUTS}"
  --expected-ranker-rollouts-sha256 "${BE_SCREENQA_RANKER_ROLLOUTS_SHA256}"
  --ranker-input-audit "${BE_SCREENQA_RANKER_INPUT_AUDIT}"
  --expected-ranker-input-audit-sha256 "${BE_SCREENQA_RANKER_INPUT_AUDIT_SHA256}"
  --v1-protocol "${BE_SCREENQA_V1_PROTOCOL}"
  --expected-v1-protocol-sha256 "${BE_SCREENQA_V1_PROTOCOL_SHA256}"
  --v2-protocol "${BE_SCREENQA_V2_PROTOCOL}"
  --expected-v2-protocol-sha256 "${BE_SCREENQA_V2_PROTOCOL_SHA256}"
  --sealed-output-dir "${calibration_dir}"
  --sealed-output-dir "${formal_dir}"
  --sealed-output-dir "${reserve_dir}"
  --sealed-output-dir "${untouched_dir}"
  --expected-code-revision "${BE_SCREENQA_EXPECTED_CODE_REVISION}"
  --output "${BE_SCREENQA_SEMANTIC_ACTIVATION}"
  --resume
)
"${python_bin}" scripts/verify_screenqa_semantic_activation.py "${activation_args[@]}"
check_hash "${BE_SCREENQA_SEMANTIC_ACTIVATION}" "${BE_SCREENQA_SEMANTIC_ACTIVATION_SHA256}" "activation audit after verification"

mkdir -p "${BE_SCREENQA_SEMANTIC_FEATURE_DIR}"
prepare_args=(
  --rollouts "${BE_SCREENQA_RANKER_ROLLOUTS}"
  --expected-rollouts-sha256 "${BE_SCREENQA_RANKER_ROLLOUTS_SHA256}"
  --output-dir "${plan_dir}"
  --shards 4
  --batch-size 4
  --expected-code-revision "${BE_SCREENQA_EXPECTED_CODE_REVISION}"
  --expected-candidate-count 4
)
if [[ -d "${plan_dir}" ]]; then prepare_args+=(--resume); fi
"${python_bin}" "${preparer}" "${prepare_args[@]}"
if [[ "$(jq -r '.assignment_unit // ""' "${plan}")" != global_sorted_decision_batch \
  || "$(jq -r '.assignment_outcome_fields_used' "${plan}")" != false \
  || "$(jq -r '.batch_size // 0' "${plan}")" -ne 4 \
  || "$(jq -r '.shard_count // 0' "${plan}")" -ne 4 \
  || "$(jq -r '.decisions // 0' "${plan}")" -ne 14511 \
  || "$(jq -r '.records // 0' "${plan}")" -ne 72555 ]]; then
  echo "ScreenQA semantic shard plan contract mismatch" >&2
  exit 2
fi

shard_rollout() {
  printf '%s/rollouts.shard-%02d.jsonl' "${plan_dir}" "$1"
}

shard_feature_dir() {
  printf '%s/shard-%02d' "${shard_root}" "$1"
}

verify_shard_rollout() {
  local index=$1
  local path expected actual
  path=$(shard_rollout "${index}")
  expected=$(jq -r --argjson index "${index}" '.shards[] | select(.index == $index) | .rollouts_sha256' "${plan}")
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ScreenQA semantic rollout shard ${index} SHA-256 mismatch" >&2
    return 2
  fi
}

run_base_shard() {
  local index=$1 rollouts output_dir output rollout_sha
  local resume_args=()
  verify_shard_rollout "${index}"
  rollouts=$(shard_rollout "${index}")
  output_dir=$(shard_feature_dir "${index}")
  output="${output_dir}/features-label-free.pt"
  rollout_sha=$(sha256sum "${rollouts}")
  rollout_sha=${rollout_sha%% *}
  mkdir -p "${output_dir}"
  if [[ -e "${output}" ]]; then resume_args=(--resume); fi
  CUDA_VISIBLE_DEVICES="${index}" "${python_bin}" -m beyond_entropy extract-qwen-features \
    --rollouts "${rollouts}" \
    --expected-rollouts-sha256 "${rollout_sha}" \
    --output "${output}" \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --model-revision "${model_revision}" \
    --min-pixels 200704 \
    --max-pixels 602112 \
    --attention-implementation sdpa \
    --question-feature-mode input_mean \
    --checkpoint-interval 32 \
    --exclude-outcomes \
    "${resume_args[@]}"
}

run_multimodal_shard() {
  local index=$1 rollouts output_dir source output
  local resume_args=()
  verify_shard_rollout "${index}"
  rollouts=$(shard_rollout "${index}")
  output_dir=$(shard_feature_dir "${index}")
  source="${output_dir}/features-label-free.pt"
  output="${output_dir}/features-multimodal-label-free.pt"
  if [[ -e "${output}" ]]; then resume_args=(--resume); fi
  CUDA_VISIBLE_DEVICES="${index}" "${python_bin}" scripts/reembed_contextual_questions.py \
    --source-features "${source}" \
    --rollouts "${rollouts}" \
    --output "${output}" \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --revision "${model_revision}" \
    --device-map cuda:0 \
    --dtype bfloat16 \
    --attention-implementation sdpa \
    --mode multimodal-original \
    --batch-size 4 \
    --checkpoint-interval 64 \
    "${resume_args[@]}"
}

run_attention_shard() {
  local index=$1 rollouts output_dir source output
  local resume_args=()
  verify_shard_rollout "${index}"
  rollouts=$(shard_rollout "${index}")
  output_dir=$(shard_feature_dir "${index}")
  source="${output_dir}/features-multimodal-label-free.pt"
  output="${output_dir}/features-question-region-attention-label-free.pt"
  if [[ -e "${output}" ]]; then resume_args=(--resume); fi
  CUDA_VISIBLE_DEVICES="${index}" "${python_bin}" scripts/extract_question_region_attention.py \
    --source-features "${source}" \
    --rollouts "${rollouts}" \
    --output "${output}" \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --revision "${model_revision}" \
    --device-map cuda:0 \
    --dtype bfloat16 \
    --top-layers 4 \
    --checkpoint-interval 32 \
    "${resume_args[@]}"
}

run_parallel_stage() {
  local function_name=$1 label=$2
  local pids=() failed=0 index pid
  for index in 0 1 2 3; do
    "${function_name}" "${index}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "one or more ScreenQA ${label} semantic shards failed" >&2
    return 1
  fi
}

merge_stage() {
  local stage=$1 shard_name=$2 output=$3 report=$4 source=${5:-}
  local args=(
    --stage "${stage}"
    --full-rollouts "${BE_SCREENQA_RANKER_ROLLOUTS}"
    --expected-full-rollouts-sha256 "${BE_SCREENQA_RANKER_ROLLOUTS_SHA256}"
    --expected-code-revision "${BE_SCREENQA_EXPECTED_CODE_REVISION}"
    --output "${output}"
    --report "${report}"
  )
  local index plan_sha source_sha
  plan_sha=$(sha256sum "${plan}")
  plan_sha=${plan_sha%% *}
  args+=(--shard-plan "${plan}" --expected-shard-plan-sha256 "${plan_sha}")
  for index in 0 1 2 3; do args+=(--shard-rollouts "$(shard_rollout "${index}")"); done
  for index in 0 1 2 3; do args+=(--shard-features "$(shard_feature_dir "${index}")/${shard_name}"); done
  if [[ -n "${source}" ]]; then
    source_sha=$(sha256sum "${source}")
    source_sha=${source_sha%% *}
    args+=(--source-features "${source}" --expected-source-features-sha256 "${source_sha}")
  fi
  if [[ -e "${output}" || -e "${report}" ]]; then args+=(--resume); fi
  "${python_bin}" "${merger}" "${args[@]}"
}

run_parallel_stage run_base_shard base
merge_stage base features-label-free.pt "${base_features}" "${shard_root}/merge-base.json"
run_parallel_stage run_multimodal_shard multimodal
merge_stage multimodal features-multimodal-label-free.pt "${multimodal_features}" "${shard_root}/merge-multimodal.json" "${base_features}"
run_parallel_stage run_attention_shard attention
merge_stage attention features-question-region-attention-label-free.pt "${attention_features}" "${shard_root}/merge-attention.json" "${multimodal_features}"

label_free_tmp="${label_free_audit}.tmp"
if [[ -e "${label_free_tmp}" ]]; then
  echo "ScreenQA label-free audit staging file exists" >&2
  exit 2
fi
"${python_bin}" scripts/audit_label_free_semantic_features.py \
  --features "${attention_features}" \
  --rollouts "${BE_SCREENQA_RANKER_ROLLOUTS}" > "${label_free_tmp}"
if [[ -e "${label_free_audit}" ]]; then
  if [[ "${resume_mode}" != 1 ]] || ! cmp -s "${label_free_tmp}" "${label_free_audit}"; then
    echo "existing ScreenQA label-free audit differs from recomputation" >&2
    exit 2
  fi
  rm "${label_free_tmp}"
else
  mv "${label_free_tmp}" "${label_free_audit}"
fi
attention_sha256=$(sha256sum "${attention_features}")
attention_sha256=${attention_sha256%% *}
if [[ "$(jq -r '.decisions // 0' "${label_free_audit}")" -ne 14511 \
  || "$(jq -r 'if has("outcomes_included_metadata") then .outcomes_included_metadata else true end' "${label_free_audit}")" != false \
  || "$(jq -r '.features_sha256 // ""' "${label_free_audit}")" != "${attention_sha256}" \
  || "$(jq -r '.rollouts_sha256 // ""' "${label_free_audit}")" != "${BE_SCREENQA_RANKER_ROLLOUTS_SHA256}" ]]; then
  echo "ScreenQA label-free semantic feature audit failed" >&2
  exit 2
fi
(
  cd "${BE_SCREENQA_SEMANTIC_FEATURE_DIR}"
  sha256sum \
    features-label-free.pt \
    features-multimodal-label-free.pt \
    features-question-region-attention-label-free.pt \
    label-free-audit.json > SHA256SUMS
  sha256sum --check SHA256SUMS
)
printf 'screenqa_semantic_features_sha256=%s\n' "${attention_sha256}"
printf 'screenqa_semantic_label_free_audit_sha256=%s\n' "$(sha256sum "${label_free_audit}" | cut -d ' ' -f 1)"
