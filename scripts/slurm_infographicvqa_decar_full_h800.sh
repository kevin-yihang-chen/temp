#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=08:15:00
#SBATCH --job-name=be-infovqa-decar-full
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-infovqa-decar-full-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: worker EXPECTED_REVISION WORKER_SHA256 FREEZE_SHA256 RESUME SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_freeze_sha256=$3
resume_mode=$4
submit_epoch=$5
if [[ "${resume_mode}" != 0 && "${resume_mode}" != 1 ]]; then
  echo "DECAR full resume must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "DECAR full submit epoch must be an integer" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
manifest="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-manifest-v1/task-manifest.jsonl"
image_manifest="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-manifest-v1/image-manifest.jsonl"
materialization_complete="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-manifest-v1/complete.json"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-method-protocol-v1.md"
materialization_result="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-full-materialization-result-v1.md"
pilot_result="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-pilot-result-v2.md"
feature_correction="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-feature-implementation-correction-v1.md"
freeze="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-full-generation-freeze-v1.md"
worker="${repo}/scripts/slurm_infographicvqa_decar_full_h800.sh"
rollout_merger="${repo}/scripts/merge_qwen_rollout_shards.py"
nll_scorer="${repo}/scripts/score_visual_action_answer_nll.py"
nll_merger="${repo}/scripts/merge_visual_action_answer_nll.py"
feature_merger="${repo}/scripts/merge_semantic_feature_shards.py"
feature_auditor="${repo}/scripts/audit_label_free_semantic_features.py"
decar_input_auditor="${repo}/scripts/audit_infographicvqa_decar_inputs.py"
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
rollout_root="${root}/rollout-shards"
merged_rollout_dir="${root}/merged-rollouts"
nll_root="${root}/nll-shards"
merged_nll_dir="${root}/merged-nll"
feature_root="${root}/feature-shards"
merged_feature_dir="${root}/merged-features"
execution_dir="${root}/execution"
model=Qwen/Qwen2.5-VL-7B-Instruct
model_revision=cc594898137f460bfe9f0759e9844b3ce807cfb5
manifest_sha256=b78a024cb623b17bb8cb73416b3c62f78b140e2e3c3b9737e1dde38bdfe3d254
image_manifest_sha256=0916a6b5a32e15c4f5b3bf920e1ecd4f304aeb97ae186e0e3e846391e2304203
shard_namespace=infovqa-decar-full-shard-v1-06817

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "DECAR full ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "DECAR full tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "DECAR full tracked worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${freeze}" "${expected_freeze_sha256}" generation-freeze
require_hash "${manifest}" "${manifest_sha256}" task-manifest
require_hash "${image_manifest}" "${image_manifest_sha256}" image-manifest
require_hash "${materialization_complete}" b873b5bffc3ebf2f64e353afbfdd058608165069cab6d0387412f56e20be921b materialization-complete
require_hash "${protocol}" d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342 protocol
require_hash "${materialization_result}" 86d943966e9e0e43ce50338483f85155ebbe493e7108725ec2ad3d1fcda75a94 materialization-result
require_hash "${pilot_result}" 827fa8b15510cbdf2f1b9925978f866b582e2cc3ac9d333b0fc4fe6c21e89b8a corrected-pilot-result
require_hash "${feature_correction}" 22a7e9046dcd7e949aee2d725a068f7b9cd0b5a3476130e8e9c50818bf158d46 feature-correction
if [[ "$(wc -l < "${manifest}")" -ne 23946 || "$(wc -l < "${image_manifest}")" -ne 4406 ]]; then
  echo "DECAR full manifest population changed" >&2
  exit 2
fi
shard_population=$(
  PYTHONPATH="${repo}/src" "${python_bin}" - "${manifest}" "${shard_namespace}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

from beyond_entropy.sharding import stable_shard_index

weights = Counter()
with Path(sys.argv[1]).open(encoding="utf-8") as handle:
    for line in handle:
        weights[json.loads(line)["source_id"]] += 1
questions = [0, 0, 0, 0]
sources = [0, 0, 0, 0]
for source_id, weight in weights.items():
    index = stable_shard_index(source_id, 4, namespace=sys.argv[2])
    questions[index] += weight
    sources[index] += 1
print(" ".join(map(str, questions)) + "|" + " ".join(map(str, sources)))
PY
)
if [[ "${shard_population}" != "6014 6036 5910 5986|538 597 547 522" ]]; then
  echo "DECAR full frozen source-shard population changed: ${shard_population}" >&2
  exit 2
fi
if [[ ! -d /userhome/cs3/yihangc/Data/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/${model_revision} ]]; then
  echo "DECAR full pinned Qwen-7B cache is absent" >&2
  exit 2
fi
if [[ -d "${root}" && -n "$(find "${root}" -mindepth 1 -print -quit)" && "${resume_mode}" != 1 ]]; then
  echo "existing DECAR full outputs require explicit resume" >&2
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
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Slurm exposed no GPUs to DECAR full" >&2
  exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_gpus[@]}" -ne 4 ]]; then
  echo "DECAR full requires exactly four H800 devices" >&2
  exit 2
fi
mkdir -p "${rollout_root}" "${merged_rollout_dir}" "${nll_root}" \
  "${merged_nll_dir}" "${feature_root}" "${merged_feature_dir}" "${execution_dir}"

job_start_epoch=$(date +%s)
queue_wait_seconds=$((job_start_epoch - submit_epoch))
if [[ "${queue_wait_seconds}" -lt 0 ]]; then
  echo "DECAR full submit epoch is in the future" >&2
  exit 2
fi
echo "DECAR full start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "tracked revision: ${expected_revision}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

run_collect_shard() {
  local index=$1 label=$2 shard_name shard_dir
  shard_name=$(printf 'shard-%05d-of-00004' "${index}")
  shard_dir="${rollout_root}/${shard_name}"
  mkdir -p "${shard_dir}"
  CUDA_VISIBLE_DEVICES="${allocated_gpus[${index}]}" \
    "${python_bin}" -m beyond_entropy collect-qwen \
      --manifest "${manifest}" --expected-manifest-sha256 "${manifest_sha256}" \
      --shard-count 4 --shard-index "${index}" --shard-key source_id \
      --shard-namespace "${shard_namespace}" \
      --output "${shard_dir}/rollouts.jsonl" --resume --checkpoint-interval 8 \
      --model "${model}" --model-revision "${model_revision}" \
      --scorer docvqa --candidate-count 4 --proposer ug-grid \
      --visual-crop-ratio 2.0 --visual-cost 1.0 \
      --generation-seeds 0 --bootstrap-resamples 100 --bootstrap-seed 20260917 \
      --scientific-status "registered InfographicVQA full-train DECAR OOF sibling bank; validation/test sealed" \
      --max-new-tokens 32 --min-pixels 200704 --max-pixels 602112 \
      --device-map cuda:0 --dtype bfloat16 --attention-implementation sdpa \
      --system-prompt "You are a helpful assistant." \
      > "${shard_dir}/${label}.log" 2>&1
}

run_parallel_collect() {
  local label=$1 failed=0 index pid
  local pids=()
  for index in 0 1 2 3; do run_collect_shard "${index}" "${label}" & pids+=("$!"); done
  for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
  [[ "${failed}" -eq 0 ]]
}

rollout_start=$(date +%s)
run_parallel_collect collect
for index in 0 1 2 3; do
  shard_name=$(printf 'shard-%05d-of-00004' "${index}")
  shard_dir="${rollout_root}/${shard_name}"
  first_sha_file="${shard_dir}/rollouts.first-pass.sha256"
  if [[ ! -e "${first_sha_file}" ]]; then
    cp "${shard_dir}/rollouts.provenance.json" "${shard_dir}/rollouts.first-pass.provenance.json"
    sha "${shard_dir}/rollouts.jsonl" > "${first_sha_file}"
  fi
done
run_parallel_collect resume
for index in 0 1 2 3; do
  shard_name=$(printf 'shard-%05d-of-00004' "${index}")
  shard_dir="${rollout_root}/${shard_name}"
  expected_sha=$(<"${shard_dir}/rollouts.first-pass.sha256")
  actual_sha=$(sha "${shard_dir}/rollouts.jsonl")
  records=$(wc -l < "${shard_dir}/rollouts.jsonl")
  resumed=$(jq -r '.resumed_from_records' "${shard_dir}/rollouts.provenance.json")
  if [[ "${actual_sha}" != "${expected_sha}" || "${resumed}" -ne "${records}" \
    || "$(jq -r '.shard_key' "${shard_dir}/rollouts.provenance.json")" != source_id \
    || "$(jq -r '.shard_namespace' "${shard_dir}/rollouts.provenance.json")" != "${shard_namespace}" ]]; then
    echo "DECAR full rollout shard ${index} source/resume audit failed" >&2
    exit 2
  fi
  jq -n --arg before "${expected_sha}" --arg after "${actual_sha}" \
    --argjson records "${records}" --argjson resumed "${resumed}" \
    '{passed:true,records:$records,resumed_from_records:$resumed,rollouts_sha256_before_resume:$before,rollouts_sha256_after_resume:$after}' \
    > "${shard_dir}/resume.audit.json.tmp"
  mv "${shard_dir}/resume.audit.json.tmp" "${shard_dir}/resume.audit.json"
done
rollout_seconds=$(( $(date +%s) - rollout_start ))

merged_rollouts="${merged_rollout_dir}/rollouts.jsonl"
if [[ ! -e "${merged_rollouts}" ]]; then
  "${python_bin}" "${rollout_merger}" \
    --manifest "${manifest}" --expected-manifest-sha256 "${manifest_sha256}" \
    --run-root "${rollout_root}" --shard-count 4 --shard-key source_id \
    --shard-namespace "${shard_namespace}" \
    --output "${merged_rollouts}" --expected-code-revision "${expected_revision}" \
    --expected-scorer docvqa --require-resume-audit \
    --bootstrap-resamples 100 --bootstrap-seed 20260917
fi
merged_rollouts_sha256=$(sha "${merged_rollouts}")
if [[ "$(wc -l < "${merged_rollouts}")" -ne 119730 \
  || "$(jq -r '.selected_states' "${merged_rollout_dir}/rollouts.merge.json")" -ne 23946 \
  || "$(jq -r '.shard_key' "${merged_rollout_dir}/rollouts.merge.json")" != source_id \
  || "$(jq -r '.shard_namespace' "${merged_rollout_dir}/rollouts.merge.json")" != "${shard_namespace}" \
  || "$(jq -r '.merged_rollouts_sha256' "${merged_rollout_dir}/rollouts.merge.json")" != "${merged_rollouts_sha256}" ]]; then
  echo "DECAR full merged rollout contract failed" >&2
  exit 2
fi

run_nll_shard() {
  local index=$1 label=$2 output
  output="${nll_root}/answer-nll-shard-${index}-of-4.jsonl"
  CUDA_VISIBLE_DEVICES="${allocated_gpus[${index}]}" \
    "${python_bin}" "${nll_scorer}" \
      --manifest "${manifest}" --rollouts "${merged_rollouts}" --output "${output}" \
      --expected-manifest-sha256 "${manifest_sha256}" \
      --expected-rollouts-sha256 "${merged_rollouts_sha256}" \
      --shard-count 4 --shard-index "${index}" --shard-key source_id \
      --shard-namespace "${shard_namespace}" \
      --checkpoint-interval 8 --resume \
      --model "${model}" --model-revision "${model_revision}" \
      --device-map cuda:0 --dtype bfloat16 --attention-implementation sdpa \
      --min-pixels 200704 --max-pixels 602112 \
      --system-prompt "You are a helpful assistant." --code-revision "${expected_revision}" \
      --scientific-status "registered InfographicVQA full-train DECAR teacher likelihood; validation/test sealed" \
      > "${nll_root}/shard-${index}-${label}.log" 2>&1
}

run_parallel_nll() {
  local label=$1 failed=0 index pid
  local pids=()
  for index in 0 1 2 3; do run_nll_shard "${index}" "${label}" & pids+=("$!"); done
  for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
  [[ "${failed}" -eq 0 ]]
}

nll_start=$(date +%s)
run_parallel_nll score
for index in 0 1 2 3; do
  output="${nll_root}/answer-nll-shard-${index}-of-4.jsonl"
  provenance="${output%.jsonl}.provenance.json"
  if [[ ! -e "${output%.jsonl}.first-pass.sha256" ]]; then
    cp "${provenance}" "${output%.jsonl}.first-pass.provenance.json"
    sha "${output}" > "${output%.jsonl}.first-pass.sha256"
  fi
done
run_parallel_nll resume
for index in 0 1 2 3; do
  output="${nll_root}/answer-nll-shard-${index}-of-4.jsonl"
  provenance="${output%.jsonl}.provenance.json"
  expected_sha=$(<"${output%.jsonl}.first-pass.sha256")
  decisions=$(jq -r '.decisions' "${provenance}")
  if [[ "$(sha "${output}")" != "${expected_sha}" \
    || "$(jq -r '.resumed_from_decisions' "${provenance}")" -ne "${decisions}" \
    || "$(jq -r '.records' "${provenance}")" -ne $((decisions * 5)) \
    || "$(jq -r '.shard_key' "${provenance}")" != source_id \
    || "$(jq -r '.shard_namespace' "${provenance}")" != "${shard_namespace}" \
    || "$(jq -r '.raw_targets_written' "${provenance}")" != false ]]; then
    echo "DECAR full NLL shard ${index} source/resume/leakage audit failed" >&2
    exit 2
  fi
done
nll_seconds=$(( $(date +%s) - nll_start ))

merged_nll="${merged_nll_dir}/answer-nll.jsonl"
if [[ ! -e "${merged_nll}" ]]; then
  "${python_bin}" "${nll_merger}" \
    --shard "${nll_root}/answer-nll-shard-0-of-4.jsonl" \
    --shard "${nll_root}/answer-nll-shard-1-of-4.jsonl" \
    --shard "${nll_root}/answer-nll-shard-2-of-4.jsonl" \
    --shard "${nll_root}/answer-nll-shard-3-of-4.jsonl" \
    --output "${merged_nll}" --expected-shard-count 4 \
    --expected-decisions 23946 --expected-records 119730 --expected-sources 2204
fi
merged_nll_sha256=$(sha "${merged_nll}")
merged_nll_provenance="${merged_nll%.jsonl}.provenance.json"
if [[ "$(wc -l < "${merged_nll}")" -ne 119730 \
  || "$(jq -r '.output_sha256' "${merged_nll_provenance}")" != "${merged_nll_sha256}" \
  || "$(jq -r '.shard_key' "${merged_nll_provenance}")" != source_id \
  || "$(jq -r '.shard_namespace' "${merged_nll_provenance}")" != "${shard_namespace}" \
  || "$(jq -r '.source_shards_disjoint' "${merged_nll_provenance}")" != true \
  || "$(jq -r '.raw_targets_written' "${merged_nll_provenance}")" != false ]]; then
  echo "DECAR full merged NLL contract failed" >&2
  exit 2
fi

run_feature_shard() {
  local index=$1 shard_name shard_rollouts output resume_args=()
  shard_name=$(printf 'shard-%05d-of-00004' "${index}")
  shard_rollouts="${rollout_root}/${shard_name}/rollouts.jsonl"
  output="${feature_root}/shard-${index}/features-label-free.pt"
  mkdir -p "${feature_root}/shard-${index}"
  if [[ -e "${output}" ]]; then resume_args=(--resume); fi
  CUDA_VISIBLE_DEVICES="${allocated_gpus[${index}]}" \
    "${python_bin}" -m beyond_entropy extract-qwen-features \
      --rollouts "${shard_rollouts}" --expected-rollouts-sha256 "$(sha "${shard_rollouts}")" \
      --output "${output}" --model "${model}" --model-revision "${model_revision}" \
      --device-map cuda:0 --dtype bfloat16 --attention-implementation sdpa \
      --min-pixels 200704 --max-pixels 602112 \
      --question-feature-mode contextual_text_mean --checkpoint-interval 8 \
      --exclude-outcomes "${resume_args[@]}" \
      > "${feature_root}/shard-${index}/features.log" 2>&1
}

feature_start=$(date +%s)
pids=()
failed=0
for index in 0 1 2 3; do run_feature_shard "${index}" & pids+=("$!"); done
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more DECAR full semantic feature shards failed" >&2
  exit 1
fi

feature_args=(
  --stage base --full-rollouts "${merged_rollouts}"
  --expected-full-rollouts-sha256 "${merged_rollouts_sha256}"
  --expected-code-revision "${expected_revision}"
  --output "${merged_feature_dir}/features-label-free.pt"
  --report "${merged_feature_dir}/merge-report.json"
)
for index in 0 1 2 3; do
  shard_name=$(printf 'shard-%05d-of-00004' "${index}")
  feature_args+=(--shard-rollouts "${rollout_root}/${shard_name}/rollouts.jsonl")
done
for index in 0 1 2 3; do
  feature_args+=(--shard-features "${feature_root}/shard-${index}/features-label-free.pt")
done
if [[ -e "${merged_feature_dir}/features-label-free.pt" || -e "${merged_feature_dir}/merge-report.json" ]]; then
  feature_args+=(--resume)
fi
"${python_bin}" "${feature_merger}" "${feature_args[@]}"
label_free_audit="${merged_feature_dir}/label-free-audit.json"
"${python_bin}" "${feature_auditor}" \
  --features "${merged_feature_dir}/features-label-free.pt" \
  --rollouts "${merged_rollouts}" > "${label_free_audit}.tmp"
if [[ -e "${label_free_audit}" ]]; then
  cmp -s "${label_free_audit}.tmp" "${label_free_audit}" || {
    echo "DECAR full label-free audit changed on resume" >&2; exit 2;
  }
  rm "${label_free_audit}.tmp"
else
  mv "${label_free_audit}.tmp" "${label_free_audit}"
fi
if [[ "$(jq -r '.decisions' "${label_free_audit}")" -ne 23946 \
  || "$(jq -r '.outcomes_included_metadata' "${label_free_audit}")" != false \
  || "$(jq -r '.outcome_fields_present | length' "${label_free_audit}")" -ne 0 ]]; then
  echo "DECAR full label-free feature audit failed" >&2
  exit 2
fi
merged_feature_sha256=$(sha "${merged_feature_dir}/features-label-free.pt")
decar_input_audit="${merged_feature_dir}/decar-input-audit.json"
"${python_bin}" "${decar_input_auditor}" \
  --rollouts "${merged_rollouts}" --expected-rollouts-sha256 "${merged_rollouts_sha256}" \
  --answer-nll "${merged_nll}" --expected-answer-nll-sha256 "${merged_nll_sha256}" \
  --features "${merged_feature_dir}/features-label-free.pt" \
  --expected-features-sha256 "${merged_feature_sha256}" \
  --expected-decisions 23946 --expected-sources 2204 > "${decar_input_audit}.tmp"
if [[ -e "${decar_input_audit}" ]]; then
  cmp -s "${decar_input_audit}.tmp" "${decar_input_audit}" || {
    echo "DECAR full joined input audit changed on resume" >&2; exit 2;
  }
  rm "${decar_input_audit}.tmp"
else
  mv "${decar_input_audit}.tmp" "${decar_input_audit}"
fi
if ! jq -e '
  .passed == true and .decisions == 23946 and .sources == 2204 and
  .images == 4406 and .actions_per_decision == 5 and .scalar_dim == 16 and
  .generated_token_statistics_complete == true and
  .label_free_feature_storage == true and
  .inference_feature_outcomes_included == false and
  .scientific_endpoints_reported == false
' "${decar_input_audit}" >/dev/null; then
  echo "DECAR full joined input audit failed" >&2
  exit 2
fi
decar_input_audit_sha256=$(sha "${decar_input_audit}")
feature_seconds=$(( $(date +%s) - feature_start ))

for index in 0 1 2 3; do
  shard_name=$(printf 'shard-%05d-of-00004' "${index}")
  for pair in \
    "${rollout_root}/${shard_name}/rollouts.first-pass.provenance.json:rollout" \
    "${nll_root}/answer-nll-shard-${index}-of-4.first-pass.provenance.json:nll"; do
    provenance=${pair%%:*}
    label=${pair##*:}
    if ! jq -e '
      .runtime_measurement.accelerator_name == "NVIDIA H800" and
      .runtime_measurement.compute_capability == [9,0] and
      .runtime_measurement.requested_dtype == "bfloat16" and
      .runtime_measurement.parameter_dtype == "torch.bfloat16" and
      .runtime_measurement.actual_attention_implementation == "sdpa" and
      (.runtime_measurement.peak_allocated_bytes > 0)
    ' "${provenance}" >/dev/null; then
      echo "DECAR full ${label} H800 runtime contract failed for shard ${index}" >&2
      exit 2
    fi
  done
done

job_end_epoch=$(date +%s)
execution="${execution_dir}/job-${SLURM_JOB_ID}.json"
jq -n \
  --arg schema infographicvqa_decar_full_generation_execution_v1 \
  --arg job_id "${SLURM_JOB_ID}" --arg code_revision "${expected_revision}" \
  --arg accelerator "NVIDIA H800" --arg manifest_sha256 "${manifest_sha256}" \
  --arg image_manifest_sha256 "${image_manifest_sha256}" \
  --arg rollouts_sha256 "${merged_rollouts_sha256}" \
  --arg nll_sha256 "${merged_nll_sha256}" --arg features_sha256 "${merged_feature_sha256}" \
  --arg decar_input_audit_sha256 "${decar_input_audit_sha256}" \
  --arg shard_namespace "${shard_namespace}" \
  --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson rollout_seconds "${rollout_seconds}" --argjson nll_seconds "${nll_seconds}" \
  --argjson feature_seconds "${feature_seconds}" \
  --argjson total_seconds "$((job_end_epoch - job_start_epoch))" \
  '{schema:$schema,job_id:$job_id,code_revision:$code_revision,accelerator:$accelerator,gpu_count:4,queue_wait_seconds:$queue_wait_seconds,shard_key:"source_id",shard_namespace:$shard_namespace,source_shards_disjoint:true,timing_seconds:{rollout_and_resume:$rollout_seconds,teacher_nll_and_resume:$nll_seconds,label_free_features_and_strict_join:$feature_seconds,total:$total_seconds},artifacts:{manifest_sha256:$manifest_sha256,image_manifest_sha256:$image_manifest_sha256,merged_rollouts_sha256:$rollouts_sha256,merged_teacher_nll_sha256:$nll_sha256,merged_label_free_features_sha256:$features_sha256,decar_input_audit_sha256:$decar_input_audit_sha256},population:{questions:23946,sources:2204,images:4406,actions_per_question:5},generated_token_statistics_complete:true,predictions_computed:false,scientific_endpoints_used_for_selection:false,validation_or_test_inputs_used:false}' \
  > "${execution}.tmp"
mv "${execution}.tmp" "${execution}"
echo "DECAR full end: $(date --iso-8601=seconds)"
printf 'infographicvqa_decar_full_complete=%s execution_sha256=%s\n' \
  "${execution}" "$(sha "${execution}")"
