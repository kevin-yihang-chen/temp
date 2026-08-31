#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=00:20:00
#SBATCH --job-name=be-screenqa-7b-runtime-recovery
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-7b-runtime-recovery-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_BB7_RECOVERY_EXPECTED_CODE_REVISION BE_BB7_RECOVERY_RUN_ROOT
  BE_BB7_RECOVERY_WORKER_SHA256 BE_BB7_RECOVERY_MODULE_SHA256
  BE_BB7_RECOVERY_CLI_SHA256 BE_BB7_RECOVERY_COLLECTOR_SHA256
  BE_BB7_RECOVERY_BACKEND_SHA256 BE_BB7_RECOVERY_ROLLOUT_SHA256
  BE_BB7_RECOVERY_CROPS_SHA256 BE_BB7_RECOVERY_BENCHMARKS_SHA256
  BE_BB7_RECOVERY_ROLLOUT_MERGER_MODULE_SHA256
  BE_BB7_RECOVERY_ROLLOUT_MERGER_CLI_SHA256
  BE_BB7_RECOVERY_SCORE_MODULE_SHA256 BE_BB7_RECOVERY_SCORER_SHA256
  BE_BB7_RECOVERY_SCORE_MERGER_SHA256 BE_BB7_RECOVERY_ANALYZER_SHA256
  BE_BB7_RECOVERY_AUDIT_MODULE_SHA256 BE_BB7_RECOVERY_PROTOCOL_SHA256
  BE_BB7_RECOVERY_ANALYSIS_IMPLEMENTATION_SHA256 BE_BB7_RECOVERY_SUBMIT_EPOCH
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done
if [[ ! "${BE_BB7_RECOVERY_SUBMIT_EPOCH}" =~ ^[0-9]+$ ]]; then
  echo "BE_BB7_RECOVERY_SUBMIT_EPOCH must be an integer epoch" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
root="${BE_BB7_RECOVERY_RUN_ROOT}"
manifest="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-manifest-v1/backbone-7b-source512-manifest-v1.jsonl"
rollout_root="${root}/rollout-shards"
merged_rollout_dir="${root}/merged-rollouts"
score_root="${root}/score-shards"
merged_score_dir="${root}/merged-scores"
analysis_dir="${root}/analysis"
execution_dir="${root}/execution"
replay_root="${execution_dir}/rollout-runtime-replay-${SLURM_JOB_ID}"
archive_dir="${execution_dir}/pre-runtime-replay-merge-job-199148"
worker="${repo_dir}/scripts/slurm_screenqa_backbone_7b_runtime_recovery.sh"
recovery_module="${repo_dir}/src/beyond_entropy/rollout_runtime_recovery.py"
recovery_cli="${repo_dir}/scripts/recover_rollout_runtime_provenance.py"
collector="${repo_dir}/src/beyond_entropy/cli.py"
backend="${repo_dir}/src/beyond_entropy/qwen_backend.py"
rollout_module="${repo_dir}/src/beyond_entropy/rollout.py"
crops_module="${repo_dir}/src/beyond_entropy/crops.py"
benchmarks_module="${repo_dir}/src/beyond_entropy/benchmarks.py"
rollout_merger_module="${repo_dir}/src/beyond_entropy/rollout_shards.py"
rollout_merger="${repo_dir}/scripts/merge_qwen_rollout_shards.py"
score_module="${repo_dir}/src/beyond_entropy/answer_likelihood.py"
scorer="${repo_dir}/scripts/score_visual_action_answer_nll.py"
score_merger="${repo_dir}/scripts/merge_visual_action_answer_nll.py"
analyzer="${repo_dir}/scripts/analyze_visual_action_proxy_outcomes.py"
audit_module="${repo_dir}/src/beyond_entropy/proxy_outcome_audit.py"
protocol="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-diagnostic-protocol-v1.md"
analysis_implementation="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-analysis-implementation-v1.md"
model=Qwen/Qwen2.5-VL-7B-Instruct
model_revision=cc594898137f460bfe9f0759e9844b3ce807cfb5
manifest_sha256=4af43ac80a1666c174774d1c33383adca625e1ef4fc535ffb74e627f149290d0

sha256_of() {
  sha256sum "$1" | cut -d ' ' -f 1
}

check_hash() {
  local path=$1
  local expected=$2
  local label=$3
  if [[ ! -s "${path}" || "$(sha256_of "${path}")" != "${expected}" ]]; then
    echo "Qwen-7B runtime recovery ${label} hash mismatch" >&2
    exit 2
  fi
}

actual_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_revision}" != "${BE_BB7_RECOVERY_EXPECTED_CODE_REVISION}" ]]; then
  echo "Qwen-7B runtime recovery code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before runtime recovery" >&2
  exit 2
fi
check_hash "${worker}" "${BE_BB7_RECOVERY_WORKER_SHA256}" worker
check_hash "${recovery_module}" "${BE_BB7_RECOVERY_MODULE_SHA256}" "recovery module"
check_hash "${recovery_cli}" "${BE_BB7_RECOVERY_CLI_SHA256}" "recovery CLI"
check_hash "${collector}" "${BE_BB7_RECOVERY_COLLECTOR_SHA256}" collector
check_hash "${backend}" "${BE_BB7_RECOVERY_BACKEND_SHA256}" backend
check_hash "${rollout_module}" "${BE_BB7_RECOVERY_ROLLOUT_SHA256}" rollout
check_hash "${crops_module}" "${BE_BB7_RECOVERY_CROPS_SHA256}" crops
check_hash "${benchmarks_module}" "${BE_BB7_RECOVERY_BENCHMARKS_SHA256}" benchmarks
check_hash "${rollout_merger_module}" "${BE_BB7_RECOVERY_ROLLOUT_MERGER_MODULE_SHA256}" "rollout merger module"
check_hash "${rollout_merger}" "${BE_BB7_RECOVERY_ROLLOUT_MERGER_CLI_SHA256}" "rollout merger CLI"
check_hash "${score_module}" "${BE_BB7_RECOVERY_SCORE_MODULE_SHA256}" "score module"
check_hash "${scorer}" "${BE_BB7_RECOVERY_SCORER_SHA256}" scorer
check_hash "${score_merger}" "${BE_BB7_RECOVERY_SCORE_MERGER_SHA256}" "score merger"
check_hash "${analyzer}" "${BE_BB7_RECOVERY_ANALYZER_SHA256}" analyzer
check_hash "${audit_module}" "${BE_BB7_RECOVERY_AUDIT_MODULE_SHA256}" "audit module"
check_hash "${protocol}" "${BE_BB7_RECOVERY_PROTOCOL_SHA256}" protocol
check_hash "${analysis_implementation}" "${BE_BB7_RECOVERY_ANALYSIS_IMPLEMENTATION_SHA256}" "analysis implementation"
check_hash "${manifest}" "${manifest_sha256}" manifest

if [[ -e "${analysis_dir}/report.json" || -e "${merged_score_dir}/answer-nll.jsonl" ]]; then
  echo "runtime recovery requires an incomplete pre-analysis run" >&2
  exit 2
fi
if [[ -e "${archive_dir}" ]]; then
  echo "runtime recovery merge archive already exists" >&2
  exit 2
fi
for protected in calibration formal reserve untouched validation test; do
  protected_dir="${repo_dir}/artifacts/screenqa-train-factorized-v1/${protected}-manifest-v1"
  if [[ -e "${protected_dir}" && ( ! -d "${protected_dir}" || -n "$(find "${protected_dir}" -mindepth 1 -print -quit)" ) ]]; then
    echo "protected ScreenQA role opened before runtime recovery: ${protected}" >&2
    exit 2
  fi
done

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Slurm did not expose four H800 devices" >&2
  exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_gpus[@]}" -ne 4 ]]; then
  echo "runtime recovery expected four CUDA devices" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
export BE_CODE_REVISION="${actual_revision}"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "${execution_dir}" "${merged_score_dir}"
cd "${repo_dir}"

job_start_epoch=$(date +%s)
queue_wait_seconds=$((job_start_epoch - BE_BB7_RECOVERY_SUBMIT_EPOCH))
if [[ "${queue_wait_seconds}" -lt 0 ]]; then
  echo "runtime recovery submission epoch is in the future" >&2
  exit 2
fi

"${python_bin}" "${recovery_cli}" prepare \
  --manifest "${manifest}" \
  --rollout-root "${rollout_root}" \
  --replay-root "${replay_root}" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --shard-count 4

run_probe() {
  local shard_index=$1
  local probe_manifest probe_manifest_sha256 probe_rollouts probe_dir
  probe_manifest=$(jq -r ".entries[${shard_index}].probe_manifest" "${replay_root}/plan.json")
  probe_manifest_sha256=$(jq -r ".entries[${shard_index}].probe_manifest_sha256" "${replay_root}/plan.json")
  probe_rollouts=$(jq -r ".entries[${shard_index}].probe_rollouts" "${replay_root}/plan.json")
  probe_dir=$(dirname "${probe_rollouts}")
  CUDA_VISIBLE_DEVICES="${allocated_gpus[${shard_index}]}" \
    "${python_bin}" -m beyond_entropy collect-qwen \
      --manifest "${probe_manifest}" \
      --expected-manifest-sha256 "${probe_manifest_sha256}" \
      --output "${probe_rollouts}" \
      --checkpoint-interval 1 \
      --model "${model}" \
      --model-revision "${model_revision}" \
      --scorer screenqa \
      --candidate-count 4 \
      --proposer ug-grid \
      --visual-crop-ratio 2.0 \
      --visual-cost 1.0 \
      --generation-seeds 0 \
      --bootstrap-resamples 100 \
      --bootstrap-seed 20260903 \
      --scientific-status "provenance-only deterministic H800 replay after completed Qwen-7B rollout; not candidate selection" \
      --max-new-tokens 32 \
      --min-pixels 200704 \
      --max-pixels 602112 \
      --attention-implementation sdpa \
      --system-prompt "You are a helpful assistant." \
      > "${probe_dir}/probe.log" 2>&1
}

probe_start=$(date +%s)
pids=()
for shard_index in 0 1 2 3; do
  run_probe "${shard_index}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more Qwen-7B runtime replays failed" >&2
  exit 1
fi
probe_seconds=$(( $(date +%s) - probe_start ))

"${python_bin}" "${recovery_cli}" repair \
  --plan "${replay_root}/plan.json" \
  --code-revision "${actual_revision}" \
  --prior-job-id 199141 \
  --prior-job-id 199143 \
  --prior-job-id 199148
replay_completion="${replay_root}/replay.complete.json"
if [[ "$(jq -r '.passed' "${replay_completion}")" != true \
  || "$(jq -r '.repairs | length' "${replay_completion}")" -ne 4 \
  || "$(jq -r '.original_process_peak_reconstructed' "${replay_completion}")" != false ]]; then
  echo "Qwen-7B runtime replay completion contract failed" >&2
  exit 2
fi

mkdir -p "${archive_dir}"
old_merged_sha256=$(sha256_of "${merged_rollout_dir}/rollouts.jsonl")
for name in rollouts.jsonl rollouts.diagnostic.json rollouts.merge.json; do
  if [[ ! -s "${merged_rollout_dir}/${name}" ]]; then
    echo "pre-recovery merged rollout artifact is missing: ${name}" >&2
    exit 2
  fi
  mv "${merged_rollout_dir}/${name}" "${archive_dir}/${name}"
done
jq -n \
  --arg schema qwen7b_pre_runtime_recovery_merge_archive_v1 \
  --arg rollouts_sha256 "${old_merged_sha256}" \
  --arg source_job_id 199148 \
  '{schema:$schema,source_job_id:$source_job_id,rollouts_sha256:$rollouts_sha256,recoverable:true}' \
  > "${archive_dir}/archive.json"

data_code_revision=$(jq -r '.code_revision' "${rollout_root}/shard-00000-of-00004/rollouts.provenance.json")
for shard_index in 1 2 3; do
  shard_name=$(printf 'shard-%05d-of-00004' "${shard_index}")
  if [[ "$(jq -r '.code_revision' "${rollout_root}/${shard_name}/rollouts.provenance.json")" != "${data_code_revision}" ]]; then
    echo "rollout data code revisions differ after runtime recovery" >&2
    exit 2
  fi
done
merged_rollouts="${merged_rollout_dir}/rollouts.jsonl"
"${python_bin}" "${rollout_merger}" \
  --manifest "${manifest}" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --run-root "${rollout_root}" \
  --shard-count 4 \
  --output "${merged_rollouts}" \
  --expected-code-revision "${data_code_revision}" \
  --expected-scorer screenqa \
  --require-resume-audit \
  --bootstrap-resamples 100 \
  --bootstrap-seed 20260903
merged_rollouts_sha256=$(sha256_of "${merged_rollouts}")
if [[ "${merged_rollouts_sha256}" != "${old_merged_sha256}" ]]; then
  echo "runtime recovery changed merged rollout bytes" >&2
  exit 2
fi

validate_runtime() {
  local provenance=$1
  if ! jq -e '
    .runtime_measurement.accelerator_name == "NVIDIA H800" and
    .runtime_measurement.compute_capability == [9,0] and
    .runtime_measurement.requested_dtype == "bfloat16" and
    .runtime_measurement.parameter_dtype == "torch.bfloat16" and
    .runtime_measurement.attention_implementation == "sdpa" and
    .runtime_measurement.actual_attention_implementation == "sdpa" and
    (.runtime_measurement.peak_allocated_bytes | type == "number" and . > 0) and
    (.runtime_measurement.peak_reserved_bytes | type == "number" and . > 0)
  ' "${provenance}" >/dev/null; then
    echo "runtime recovery telemetry validation failed: ${provenance}" >&2
    exit 2
  fi
}

for shard_index in 0 1 2 3; do
  shard_name=$(printf 'shard-%05d-of-00004' "${shard_index}")
  validate_runtime "${rollout_root}/${shard_name}/rollouts.provenance.json"
  output="${score_root}/answer-nll-shard-${shard_index}-of-4.jsonl"
  provenance="${output%.jsonl}.provenance.json"
  first_sha=$(<"${score_root}/answer-nll-shard-${shard_index}-of-4.first-pass.sha256")
  if [[ "$(sha256_of "${output}")" != "${first_sha}" \
    || "$(jq -r '.output_sha256' "${provenance}")" != "${first_sha}" \
    || "$(jq -r '.decisions' "${provenance}")" -ne 128 \
    || "$(jq -r '.records' "${provenance}")" -ne 640 \
    || "$(jq -r '.sources' "${provenance}")" -ne 128 \
    || "$(jq -r '.resumed_from_decisions' "${provenance}")" -ne 128 \
    || "$(jq -r '.raw_targets_written' "${provenance}")" != false \
    || "$(jq -r '.rollouts_sha256' "${provenance}")" != "${merged_rollouts_sha256}" ]]; then
    echo "complete Qwen-7B NLL shard ${shard_index} failed recovery audit" >&2
    exit 2
  fi
  validate_runtime "${provenance}"
done

merged_scores="${merged_score_dir}/answer-nll.jsonl"
merge_args=(
  --output "${merged_scores}"
  --expected-shard-count 4
  --expected-decisions 512
  --expected-records 2560
  --expected-sources 512
)
for shard_index in 0 1 2 3; do
  merge_args+=(--shard "${score_root}/answer-nll-shard-${shard_index}-of-4.jsonl")
done
"${python_bin}" "${score_merger}" "${merge_args[@]}"
merged_scores_sha256=$(sha256_of "${merged_scores}")

analysis_start=$(date +%s)
"${python_bin}" "${analyzer}" \
  --scores "${merged_scores}" \
  --protocol "${protocol}" \
  --implementation-contract "${analysis_implementation}" \
  --output-dir "${analysis_dir}" \
  --expected-scores-sha256 "${merged_scores_sha256}" \
  --expected-protocol-sha256 "${BE_BB7_RECOVERY_PROTOCOL_SHA256}" \
  --expected-implementation-contract-sha256 "${BE_BB7_RECOVERY_ANALYSIS_IMPLEMENTATION_SHA256}" \
  --expected-decisions 512 \
  --expected-sources 512 \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 20260903 \
  --bootstrap-confidence 0.95 \
  --study-label "ScreenQA Qwen2.5-VL-7B opened development" \
  --scientific-status "frozen Qwen2.5-VL-7B mechanism diagnostic on opened ScreenQA ranker development; provenance recovery disclosed; not candidate selection or independent validation" \
  --interpretation-boundary "Only the frozen 512-source opened ScreenQA ranker-development population is used. Calibration, formal, reserve, untouched, validation, and test roles remain sealed. No threshold or call rate is selected." \
  --code-revision "${actual_revision}"
analysis_seconds=$(( $(date +%s) - analysis_start ))

report="${analysis_dir}/report.json"
if [[ "$(jq -r '.population.decisions' "${report}")" -ne 512 \
  || "$(jq -r '.population.sources' "${report}")" -ne 512 \
  || "$(jq -r '.population.zoom_actions' "${report}")" -ne 2048 \
  || "$(jq -r '.population.score_records' "${report}")" -ne 2560 \
  || "$(jq -r '.bootstrap.n_resamples' "${report}")" -ne 5000 \
  || "$(jq -r '.bootstrap.seed' "${report}")" -ne 20260903 \
  || "$(jq -r '.outcome_use.protected_role_inputs_used' "${report}")" != false ]]; then
  echo "runtime-recovered Qwen-7B analysis contract failed" >&2
  exit 2
fi

job_end_epoch=$(date +%s)
execution_audit="${execution_dir}/runtime-recovery-job-${SLURM_JOB_ID}.json"
jq -n \
  --arg schema qwen7b_runtime_recovery_execution_v1 \
  --arg job_id "${SLURM_JOB_ID}" \
  --arg code_revision "${actual_revision}" \
  --arg data_code_revision "${data_code_revision}" \
  --arg replay_completion_sha256 "$(sha256_of "${replay_completion}")" \
  --arg merged_rollouts_sha256 "${merged_rollouts_sha256}" \
  --arg merged_scores_sha256 "${merged_scores_sha256}" \
  --arg report_sha256 "$(sha256_of "${report}")" \
  --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson probe_seconds "${probe_seconds}" \
  --argjson analysis_seconds "${analysis_seconds}" \
  --argjson total_seconds "$((job_end_epoch - job_start_epoch))" \
  '{schema:$schema,job_id:$job_id,prior_jobs:["199141","199143","199148"],code_revision:$code_revision,data_code_revision:$data_code_revision,accelerator:"NVIDIA H800",gpu_count:4,queue_wait_seconds:$queue_wait_seconds,timing_seconds:{runtime_replay:$probe_seconds,analysis:$analysis_seconds,total:$total_seconds},runtime_recovery:{source:"deterministic_one_state_h800_replay",exact_action_record_match_required:true,original_process_peak_reconstructed:false,completion_sha256:$replay_completion_sha256},artifacts:{merged_rollouts_sha256:$merged_rollouts_sha256,merged_scores_sha256:$merged_scores_sha256,report_sha256:$report_sha256},protected_role_inputs_used:false}' \
  > "${execution_audit}.tmp"
mv "${execution_audit}.tmp" "${execution_audit}"

printf 'screenqa_qwen7b_runtime_recovery_complete=%s report_sha256=%s decision_ready=true\n' \
  "${execution_audit}" "$(sha256_of "${report}")"
