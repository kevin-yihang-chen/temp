#!/usr/bin/env bash
#SBATCH --partition=q-hgpu-small
#SBATCH --gres=gpu:h800:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --job-name=be-infovqa-lit-attn
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-infovqa-lit-attn-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: worker REVISION WORKER_SHA PROTOCOL_SHA BLIND_AUDIT_SHA RESUME SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_protocol_sha256=$3
expected_blind_audit_sha256=$4
resume_mode=$5
submit_epoch=$6
if [[ "${resume_mode}" != 0 && "${resume_mode}" != 1 ]]; then
  echo "literature-attention resume must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "literature-attention submit epoch must be an integer" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
worker="${repo}/scripts/slurm_infographicvqa_literature_attention_h800.sh"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-literature-attention-where-protocol-20260902-pending.md"
blind_audit="${repo}/artifacts/docvqa-train-factorized-v2/ops/attention-crop-literature-blind-audit-20260902-pending.md"
extractor="${repo}/scripts/extract_infographicvqa_literature_attention.py"
merger="${repo}/scripts/merge_infographicvqa_literature_attention_shards.py"
auditor="${repo}/scripts/audit_infographicvqa_literature_attention_features.py"
generation_root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
rollout_root="${generation_root}/rollout-shards"
base_feature_root="${generation_root}/feature-shards"
merged_rollouts="${generation_root}/merged-rollouts/rollouts.jsonl"
merged_base_features="${generation_root}/merged-features/features-label-free.pt"
output_root="${generation_root}/literature-attention-where-v1"
feature_shards="${output_root}/feature-shards"
merged_dir="${output_root}/merged-features"
execution_dir="${output_root}/execution"
merged_features="${merged_dir}/features-literature-attention-label-free.pt"
merge_report="${merged_dir}/merge-report.json"
feature_audit="${merged_dir}/literature-attention-feature-audit.json"
complete="${output_root}/complete.json"
model=Qwen/Qwen2.5-VL-7B-Instruct
model_revision=cc594898137f460bfe9f0759e9844b3ce807cfb5
merged_rollouts_sha256=9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e
merged_base_features_sha256=d0508726a50b4c1e54778392d08329b242a680fc13292cac1ebec8b42a175300
feature_hashes=(
  2ef27cfe17b5d8d36bd4410850a24e982b23f7686a9fa064c48db73c1ba0f3da
  ed643ad1d4b82500db3dd3cec6f7d6d01412cef90e7f55690ad2e57b70cabdeb
  6cf284fec70ad2873ff05a1ef17ab0958ab57d92bc1b340c03a749fdb470a69b
  4eb20a4d9ca35b693889406eb82c74f1c635e93ab07ec74917fe0976773d948e
)
rollout_hashes=(
  1e130c22a2b1ba85e41c12a18dbef66e717fafc4a996f6dde90577a839d1b6da
  506928e10c66bd47caab2be0631a63a0f63abb83e520a8769c826b84f5fe35b9
  a486e8d5e34b7ac7846b074782738826ee1bd81a1ec5833c4a18011a0597c8da
  e8bd28e33f7d72135728fb567ea9d3407fb325c7e0e49af8f860fb37c4606d83
)

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "literature-attention ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "literature-attention tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "literature-attention tracked worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${protocol}" "${expected_protocol_sha256}" protocol
require_hash "${blind_audit}" "${expected_blind_audit_sha256}" blind-audit
require_hash "${merged_rollouts}" "${merged_rollouts_sha256}" merged-rollouts
require_hash "${merged_base_features}" "${merged_base_features_sha256}" merged-base-features
for index in 0 1 2 3; do
  shard_name=$(printf 'shard-%05d-of-00004' "${index}")
  require_hash "${rollout_root}/${shard_name}/rollouts.jsonl" \
    "${rollout_hashes[${index}]}" "rollout-shard-${index}"
  require_hash "${base_feature_root}/shard-${index}/features-label-free.pt" \
    "${feature_hashes[${index}]}" "base-feature-shard-${index}"
done
if [[ ! -d /userhome/cs3/yihangc/Data/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/${model_revision} ]]; then
  echo "literature-attention pinned Qwen-7B cache is absent" >&2
  exit 2
fi
if [[ -d "${output_root}" && -n "$(find "${output_root}" -mindepth 1 -print -quit)" \
  && "${resume_mode}" != 1 ]]; then
  echo "existing literature-attention outputs require explicit resume" >&2
  exit 2
fi
available_kb=$(df -Pk "${repo}" | awk 'NR==2 {print $4}')
if [[ -z "${available_kb}" || "${available_kb}" -lt 16777216 ]]; then
  echo "literature-attention requires at least 16 GiB free disk" >&2
  exit 2
fi

export PYTHONPATH="${repo}/src"
export BE_CODE_REVISION="${expected_revision}"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HTTP_PROXY HTTPS_PROXY ALL_PROXY \
  http_proxy https_proxy all_proxy

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Slurm exposed no GPUs to literature-attention" >&2
  exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_gpus[@]}" -ne 2 ]]; then
  echo "literature-attention requires exactly two H800 devices" >&2
  exit 2
fi
mkdir -p "${feature_shards}" "${merged_dir}" "${execution_dir}"

job_start_epoch=$(date +%s)
queue_wait_seconds=$((job_start_epoch - submit_epoch))
if [[ "${queue_wait_seconds}" -lt 0 ]]; then
  echo "literature-attention submit epoch is in the future" >&2
  exit 2
fi
echo "literature-attention start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "tracked revision: ${expected_revision}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

run_feature_shard() {
  local index=$1 gpu_slot=$2 shard_name source_features rollouts output_dir output
  local resume_args=()
  shard_name=$(printf 'shard-%05d-of-00004' "${index}")
  source_features="${base_feature_root}/shard-${index}/features-label-free.pt"
  rollouts="${rollout_root}/${shard_name}/rollouts.jsonl"
  output_dir="${feature_shards}/shard-${index}"
  output="${output_dir}/features-literature-attention-label-free.pt"
  mkdir -p "${output_dir}"
  if [[ -e "${output}" ]]; then resume_args=(--resume); fi
  CUDA_VISIBLE_DEVICES="${allocated_gpus[${gpu_slot}]}" \
    "${python_bin}" "${extractor}" \
      --source-features "${source_features}" \
      --rollouts "${rollouts}" \
      --output "${output}" \
      --model "${model}" \
      --revision "${model_revision}" \
      --device-map cuda:0 \
      --dtype bfloat16 \
      --checkpoint-interval 256 \
      "${resume_args[@]}" \
      > "${output_dir}/attention.log" 2>&1
}

extraction_start=$(date +%s)
failed=0
for wave_start in 0 2; do
  pids=()
  run_feature_shard "${wave_start}" 0 & pids+=("$!")
  run_feature_shard "$((wave_start + 1))" 1 & pids+=("$!")
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  if [[ "${failed}" -ne 0 ]]; then break; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more literature-attention feature shards failed" >&2
  exit 1
fi
extraction_seconds=$(( $(date +%s) - extraction_start ))

merge_args=(
  --full-rollouts "${merged_rollouts}"
  --expected-full-rollouts-sha256 "${merged_rollouts_sha256}"
  --source-features "${merged_base_features}"
  --expected-source-features-sha256 "${merged_base_features_sha256}"
  --expected-code-revision "${expected_revision}"
  --output "${merged_features}"
  --report "${merge_report}"
)
for index in 0 1 2 3; do
  shard_name=$(printf 'shard-%05d-of-00004' "${index}")
  merge_args+=(--shard-rollouts "${rollout_root}/${shard_name}/rollouts.jsonl")
done
for index in 0 1 2 3; do
  merge_args+=(--shard-features "${feature_shards}/shard-${index}/features-literature-attention-label-free.pt")
done
if [[ -e "${merged_features}" || -e "${merge_report}" ]]; then
  merge_args+=(--resume)
fi
merge_start=$(date +%s)
"${python_bin}" "${merger}" "${merge_args[@]}"
merge_seconds=$(( $(date +%s) - merge_start ))
merged_sha256=$(sha "${merged_features}")
if [[ "$(jq -r '.passed' "${merge_report}")" != true \
  || "$(jq -r '.decisions' "${merge_report}")" -ne 23946 \
  || "$(jq -r '.sources' "${merge_report}")" -ne 2204 \
  || "$(jq -r '.source_disjoint' "${merge_report}")" != true \
  || "$(jq -r '.outcomes_included' "${merge_report}")" != false \
  || "$(jq -r '.validation_or_test_inputs_used' "${merge_report}")" != false \
  || "$(jq -r '.output_sha256' "${merge_report}")" != "${merged_sha256}" ]]; then
  echo "literature-attention merge contract failed" >&2
  exit 2
fi

"${python_bin}" "${auditor}" \
  --features "${merged_features}" \
  --expected-features-sha256 "${merged_sha256}" \
  --rollouts "${merged_rollouts}" \
  --expected-rollouts-sha256 "${merged_rollouts_sha256}" \
  --expected-code-revision "${expected_revision}" \
  --expected-model-revision "${model_revision}" \
  --source-features-sha256 "${merged_base_features_sha256}" \
  > "${feature_audit}.tmp"
if [[ -e "${feature_audit}" ]]; then
  cmp -s "${feature_audit}.tmp" "${feature_audit}" || {
    echo "literature-attention feature audit changed on resume" >&2
    exit 2
  }
  rm "${feature_audit}.tmp"
else
  mv "${feature_audit}.tmp" "${feature_audit}"
fi
if ! jq -e '
  .passed == true and
  .population == {decisions:23946,images:4406,sources:2204} and
  .outcomes_included == false and
  .candidate_actions_executed == false and
  .validation_or_test_inputs_used == false and
  .vicrop_relative_bank.score_sum_max_absolute_error <= 0.000001 and
  .laser_contrastive_all_head_bank.score_sum_max_absolute_error <= 0.000001 and
  ([.vicrop_relative_bank.selected_action_counts[]] | add) == 23946 and
  ([.laser_contrastive_all_head_bank.selected_action_counts[]] | add) == 23946
' "${feature_audit}" >/dev/null; then
  echo "literature-attention feature audit failed" >&2
  exit 2
fi

job_end_epoch=$(date +%s)
audit_sha256=$(sha "${feature_audit}")
merge_report_sha256=$(sha "${merge_report}")
execution="${execution_dir}/job-${SLURM_JOB_ID}.json"
jq -n \
  --arg schema infographicvqa_literature_attention_where_feature_execution_v1 \
  --arg job_id "${SLURM_JOB_ID}" --arg code_revision "${expected_revision}" \
  --arg protocol_sha256 "${expected_protocol_sha256}" \
  --arg blind_audit_sha256 "${expected_blind_audit_sha256}" \
  --arg features_sha256 "${merged_sha256}" --arg audit_sha256 "${audit_sha256}" \
  --arg merge_report_sha256 "${merge_report_sha256}" \
  --argjson submitted_epoch "${submit_epoch}" --argjson started_epoch "${job_start_epoch}" \
  --argjson ended_epoch "${job_end_epoch}" --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson extraction_seconds "${extraction_seconds}" --argjson merge_seconds "${merge_seconds}" \
  '{schema:$schema,job_id:$job_id,slurm_state:"COMPLETED",exit_code:"0:0",restarts:0,
    code_revision:$code_revision,protocol_sha256:$protocol_sha256,
    blind_audit_sha256:$blind_audit_sha256,submitted_epoch:$submitted_epoch,
    started_epoch:$started_epoch,ended_epoch:$ended_epoch,
    queue_wait_seconds:$queue_wait_seconds,extraction_seconds:$extraction_seconds,
    merge_seconds:$merge_seconds,gpu_type:"NVIDIA H800",gpu_count:2,prefills_per_decision:3,
    validation_or_test_inputs_used:false,outcomes_included:false,
    merged_features_sha256:$features_sha256,audit_sha256:$audit_sha256,
    merge_report_sha256:$merge_report_sha256}' > "${execution}.tmp"
mv "${execution}.tmp" "${execution}"
execution_sha256=$(sha "${execution}")
jq -n \
  --arg schema infographicvqa_literature_attention_where_feature_complete_v1 \
  --arg features_sha256 "${merged_sha256}" --arg audit_sha256 "${audit_sha256}" \
  --arg execution_sha256 "${execution_sha256}" \
  '{schema:$schema,passed:true,decisions:23946,sources:2204,images:4406,
    validation_or_test_inputs_used:false,outcomes_included:false,
    merged_features_sha256:$features_sha256,audit_sha256:$audit_sha256,
    execution_sha256:$execution_sha256}' > "${complete}.tmp"
if [[ -e "${complete}" ]]; then
  cmp -s "${complete}.tmp" "${complete}" || {
    echo "literature-attention completion marker changed on resume" >&2
    exit 2
  }
  rm "${complete}.tmp"
else
  mv "${complete}.tmp" "${complete}"
fi
echo "literature-attention complete: $(date --iso-8601=seconds)"
echo "merged literature attention SHA-256: ${merged_sha256}"
echo "feature audit SHA-256: ${audit_sha256}"
