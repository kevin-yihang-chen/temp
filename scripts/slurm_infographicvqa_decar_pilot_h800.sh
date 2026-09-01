#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=03:00:00
#SBATCH --job-name=be-infovqa-decar-pilot
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-infovqa-decar-pilot-%j.out
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
  echo "DECAR pilot resume must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "DECAR pilot submit epoch must be an integer" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
manifest="${repo}/artifacts/infographicvqa-train-v1/decar-v1/pilot-manifest-v1/task-manifest.jsonl"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-method-protocol-v1.md"
allocation_result="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-allocation-result-v1.md"
materialization_result="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-pilot-materialization-result-v1.md"
materialization_complete="${repo}/artifacts/infographicvqa-train-v1/decar-v1/pilot-manifest-v1/complete.json"
freeze="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-pilot-implementation-freeze-v2.md"
feature_correction="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-feature-implementation-correction-v1.md"
worker="${repo}/scripts/slurm_infographicvqa_decar_pilot_h800.sh"
rollout_merger="${repo}/scripts/merge_qwen_rollout_shards.py"
nll_scorer="${repo}/scripts/score_visual_action_answer_nll.py"
nll_merger="${repo}/scripts/merge_visual_action_answer_nll.py"
feature_merger="${repo}/scripts/merge_semantic_feature_shards.py"
feature_auditor="${repo}/scripts/audit_label_free_semantic_features.py"
decar_input_auditor="${repo}/scripts/audit_infographicvqa_decar_inputs.py"
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/pilot-qwen7b-v2"
rollout_root="${root}/rollout-shards"
merged_rollout_dir="${root}/merged-rollouts"
nll_root="${root}/nll-shards"
merged_nll_dir="${root}/merged-nll"
feature_root="${root}/feature-shards"
merged_feature_dir="${root}/merged-features"
execution_dir="${root}/execution"
model=Qwen/Qwen2.5-VL-7B-Instruct
model_revision=cc594898137f460bfe9f0759e9844b3ce807cfb5
manifest_sha256=80067cc1446782f458665d8ddfa98745bda73b03b9eb96da3528f82f22158d29

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "DECAR pilot ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "DECAR pilot tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "DECAR pilot tracked worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${freeze}" "${expected_freeze_sha256}" implementation-freeze
require_hash "${feature_correction}" 22a7e9046dcd7e949aee2d725a068f7b9cd0b5a3476130e8e9c50818bf158d46 feature-correction
require_hash "${manifest}" "${manifest_sha256}" task-manifest
require_hash "${protocol}" d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342 protocol
require_hash "${allocation_result}" 3d0948cc6840b008cd4b19408ff002ed0756bb0d9f7f5e6b8cdb6d0af5a4da60 allocation-result
require_hash "${materialization_result}" 7b5a73fa8fad96eae542c74a3abf4a8c5687e4b0edb68a376a5a554956358345 materialization-result
require_hash "${materialization_complete}" 9b28285892d43290b898eefa9bca3abef79f40a248323c84c4bce0df5b52562a materialization-complete
if [[ "$(wc -l < "${manifest}")" -ne 512 ]]; then
  echo "DECAR pilot manifest must contain 512 questions" >&2
  exit 2
fi
if [[ ! -d /userhome/cs3/yihangc/Data/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/${model_revision} ]]; then
  echo "DECAR pilot pinned Qwen-7B cache is absent" >&2
  exit 2
fi
if [[ -d "${root}" && -n "$(find "${root}" -mindepth 1 -print -quit)" && "${resume_mode}" != 1 ]]; then
  echo "existing DECAR pilot outputs require explicit resume" >&2
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
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Slurm exposed no GPUs to DECAR pilot" >&2
  exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_gpus[@]}" -ne 4 ]]; then
  echo "DECAR pilot requires exactly four H800 devices" >&2
  exit 2
fi
mkdir -p "${rollout_root}" "${merged_rollout_dir}" "${nll_root}" \
  "${merged_nll_dir}" "${feature_root}" "${merged_feature_dir}" "${execution_dir}"

job_start_epoch=$(date +%s)
queue_wait_seconds=$((job_start_epoch - submit_epoch))
if [[ "${queue_wait_seconds}" -lt 0 ]]; then
  echo "DECAR pilot submit epoch is in the future" >&2
  exit 2
fi
echo "DECAR pilot start: $(date --iso-8601=seconds)"
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
      --manifest "${manifest}" \
      --expected-manifest-sha256 "${manifest_sha256}" \
      --shard-count 4 --shard-index "${index}" \
      --output "${shard_dir}/rollouts.jsonl" \
      --resume --checkpoint-interval 8 \
      --model "${model}" --model-revision "${model_revision}" \
      --scorer docvqa --candidate-count 4 --proposer ug-grid \
      --visual-crop-ratio 2.0 --visual-cost 1.0 \
      --generation-seeds 0 --bootstrap-resamples 100 --bootstrap-seed 20260917 \
      --scientific-status "registered InfographicVQA 512-source DECAR engineering pilot; endpoints cannot select settings" \
      --max-new-tokens 32 --min-pixels 200704 --max-pixels 602112 \
      --device-map cuda:0 --dtype bfloat16 --attention-implementation sdpa \
      --system-prompt "You are a helpful assistant." \
      > "${shard_dir}/${label}.log" 2>&1
}

run_parallel_collect() {
  local label=$1 failed=0 index pid
  local pids=()
  for index in 0 1 2 3; do
    run_collect_shard "${index}" "${label}" & pids+=("$!")
  done
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
  if [[ "${actual_sha}" != "${expected_sha}" || "${resumed}" -ne "${records}" ]]; then
    echo "DECAR rollout shard ${index} resume audit failed" >&2
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
    --run-root "${rollout_root}" --shard-count 4 --output "${merged_rollouts}" \
    --expected-code-revision "${expected_revision}" --expected-scorer docvqa \
    --require-resume-audit --bootstrap-resamples 100 --bootstrap-seed 20260917
fi
merged_rollouts_sha256=$(sha "${merged_rollouts}")
if [[ "$(wc -l < "${merged_rollouts}")" -ne 2560 \
  || "$(jq -r '.selected_states' "${merged_rollout_dir}/rollouts.merge.json")" -ne 512 \
  || "$(jq -r '.merged_rollouts_sha256' "${merged_rollout_dir}/rollouts.merge.json")" != "${merged_rollouts_sha256}" ]]; then
  echo "DECAR merged rollout contract failed" >&2
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
      --shard-count 4 --shard-index "${index}" --checkpoint-interval 8 --resume \
      --model "${model}" --model-revision "${model_revision}" \
      --device-map cuda:0 --dtype bfloat16 --attention-implementation sdpa \
      --min-pixels 200704 --max-pixels 602112 \
      --system-prompt "You are a helpful assistant." \
      --code-revision "${expected_revision}" \
      --scientific-status "registered InfographicVQA DECAR engineering-pilot teacher likelihood; no setting selection" \
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
  if [[ "$(sha "${output}")" != "${expected_sha}" \
    || "$(jq -r '.decisions' "${provenance}")" -ne 128 \
    || "$(jq -r '.records' "${provenance}")" -ne 640 \
    || "$(jq -r '.resumed_from_decisions' "${provenance}")" -ne 128 \
    || "$(jq -r '.raw_targets_written' "${provenance}")" != false ]]; then
    echo "DECAR NLL shard ${index} resume or leakage audit failed" >&2
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
    --expected-decisions 512 --expected-records 2560 --expected-sources 512
fi
merged_nll_sha256=$(sha "${merged_nll}")
merged_nll_provenance="${merged_nll%.jsonl}.provenance.json"
if [[ "$(wc -l < "${merged_nll}")" -ne 2560 \
  || ! -f "${merged_nll_provenance}" \
  || "$(jq -r '.output_sha256' "${merged_nll_provenance}")" != "${merged_nll_sha256}" \
  || "$(jq -r '.decisions' "${merged_nll_provenance}")" -ne 512 \
  || "$(jq -r '.records' "${merged_nll_provenance}")" -ne 2560 \
  || "$(jq -r '.sources' "${merged_nll_provenance}")" -ne 512 \
  || "$(jq -r '.raw_targets_written' "${merged_nll_provenance}")" != false ]]; then
  echo "DECAR merged NLL record count failed" >&2
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
      --rollouts "${shard_rollouts}" \
      --expected-rollouts-sha256 "$(sha "${shard_rollouts}")" \
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
  echo "one or more DECAR semantic feature shards failed" >&2
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
    echo "DECAR label-free audit changed on resume" >&2; exit 2;
  }
  rm "${label_free_audit}.tmp"
else
  mv "${label_free_audit}.tmp" "${label_free_audit}"
fi
if [[ "$(jq -r '.decisions' "${label_free_audit}")" -ne 512 \
  || "$(jq -r '.outcomes_included_metadata' "${label_free_audit}")" != false \
  || "$(jq -r '.outcome_fields_present | length' "${label_free_audit}")" -ne 0 ]]; then
  echo "DECAR label-free feature audit failed" >&2
  exit 2
fi
feature_seconds=$(( $(date +%s) - feature_start ))
merged_feature_sha256=$(sha "${merged_feature_dir}/features-label-free.pt")
decar_input_audit="${merged_feature_dir}/decar-input-audit.json"
"${python_bin}" "${decar_input_auditor}" \
  --rollouts "${merged_rollouts}" \
  --expected-rollouts-sha256 "${merged_rollouts_sha256}" \
  --answer-nll "${merged_nll}" \
  --expected-answer-nll-sha256 "${merged_nll_sha256}" \
  --features "${merged_feature_dir}/features-label-free.pt" \
  --expected-features-sha256 "${merged_feature_sha256}" \
  --expected-decisions 512 --expected-sources 512 \
  > "${decar_input_audit}.tmp"
if [[ -e "${decar_input_audit}" ]]; then
  cmp -s "${decar_input_audit}.tmp" "${decar_input_audit}" || {
    echo "DECAR joined input audit changed on resume" >&2; exit 2;
  }
  rm "${decar_input_audit}.tmp"
else
  mv "${decar_input_audit}.tmp" "${decar_input_audit}"
fi
if ! jq -e '
  .passed == true and
  .decisions == 512 and
  .sources == 512 and
  .actions_per_decision == 5 and
  .scalar_dim == 16 and
  .generated_token_statistics_complete == true and
  .label_free_feature_storage == true and
  .inference_feature_outcomes_included == false and
  .scientific_endpoints_reported == false
' "${decar_input_audit}" >/dev/null; then
  echo "DECAR joined input audit failed" >&2
  exit 2
fi
decar_input_audit_sha256=$(sha "${decar_input_audit}")

for index in 0 1 2 3; do
  shard_name=$(printf 'shard-%05d-of-00004' "${index}")
  first_rollout_provenance="${rollout_root}/${shard_name}/rollouts.first-pass.provenance.json"
  first_nll_provenance="${nll_root}/answer-nll-shard-${index}-of-4.first-pass.provenance.json"
  for pair in "${first_rollout_provenance}:rollout" "${first_nll_provenance}:nll"; do
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
      echo "DECAR ${label} H800 runtime contract failed for shard ${index}" >&2
      exit 2
    fi
  done
done

job_end_epoch=$(date +%s)
execution="${execution_dir}/job-${SLURM_JOB_ID}.json"
jq -n \
  --arg schema infographicvqa_decar_pilot_execution_v1 \
  --arg job_id "${SLURM_JOB_ID}" --arg code_revision "${expected_revision}" \
  --arg accelerator "NVIDIA H800" --arg manifest_sha256 "${manifest_sha256}" \
  --arg rollouts_sha256 "${merged_rollouts_sha256}" \
  --arg nll_sha256 "${merged_nll_sha256}" --arg features_sha256 "${merged_feature_sha256}" \
  --arg decar_input_audit_sha256 "${decar_input_audit_sha256}" \
  --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson rollout_seconds "${rollout_seconds}" --argjson nll_seconds "${nll_seconds}" \
  --argjson feature_seconds "${feature_seconds}" \
  --argjson total_seconds "$((job_end_epoch - job_start_epoch))" \
  '{schema:$schema,job_id:$job_id,code_revision:$code_revision,accelerator:$accelerator,gpu_count:4,queue_wait_seconds:$queue_wait_seconds,timing_seconds:{rollout_and_resume:$rollout_seconds,teacher_nll_and_resume:$nll_seconds,label_free_features:$feature_seconds,total:$total_seconds},artifacts:{manifest_sha256:$manifest_sha256,merged_rollouts_sha256:$rollouts_sha256,merged_teacher_nll_sha256:$nll_sha256,merged_label_free_features_sha256:$features_sha256,decar_input_audit_sha256:$decar_input_audit_sha256},population:{questions:512,sources:512,actions_per_question:5},generated_token_statistics_complete:true,task_endpoints_used_for_selection:false,validation_or_test_inputs_used:false}' \
  > "${execution}.tmp"
mv "${execution}.tmp" "${execution}"
echo "DECAR pilot end: $(date --iso-8601=seconds)"
printf 'infographicvqa_decar_pilot_complete=%s execution_sha256=%s\n' \
  "${execution}" "$(sha "${execution}")"
