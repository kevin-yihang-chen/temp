#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=01:00:00
#SBATCH --job-name=be-screenqa-7b-full
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-7b-full-h800-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_BB7_FULL_EXPECTED_CODE_REVISION BE_BB7_FULL_RUN_ROOT
  BE_BB7_FULL_WORKER_SHA256 BE_BB7_FULL_COLLECTOR_SHA256
  BE_BB7_FULL_BACKEND_SHA256 BE_BB7_FULL_ROLLOUT_SHA256
  BE_BB7_FULL_CROPS_SHA256 BE_BB7_FULL_BENCHMARKS_SHA256
  BE_BB7_FULL_ROLLOUT_MERGER_MODULE_SHA256
  BE_BB7_FULL_ROLLOUT_MERGER_CLI_SHA256
  BE_BB7_FULL_SCORE_MODULE_SHA256 BE_BB7_FULL_SCORER_SHA256
  BE_BB7_FULL_SCORE_MERGER_SHA256 BE_BB7_FULL_ANALYZER_SHA256
  BE_BB7_FULL_AUDIT_MODULE_SHA256 BE_BB7_FULL_PROTOCOL_SHA256
  BE_BB7_FULL_POPULATION_ACTIVATION_SHA256
  BE_BB7_FULL_ANALYSIS_IMPLEMENTATION_SHA256
  BE_BB7_FULL_HARDWARE_ACTIVATION_SHA256
  BE_BB7_FULL_SMOKE_COMPLETION_SHA256 BE_BB7_FULL_RESUME
  BE_BB7_SUBMIT_EPOCH
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done
if [[ "${BE_BB7_FULL_RESUME}" != 0 && "${BE_BB7_FULL_RESUME}" != 1 ]]; then
  echo "BE_BB7_FULL_RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${BE_BB7_SUBMIT_EPOCH}" =~ ^[0-9]+$ ]]; then
  echo "BE_BB7_SUBMIT_EPOCH must be an integer epoch" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
manifest="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-manifest-v1/backbone-7b-source512-manifest-v1.jsonl"
protocol="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-diagnostic-protocol-v1.md"
population_activation="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-population-activation-v1.md"
analysis_implementation="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-analysis-implementation-v1.md"
hardware_activation="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-full-hardware-activation-v1.md"
smoke_completion="${repo_dir}/artifacts/screenqa-train-factorized-v1/backbone-7b-diagnostic-v1/smoke-h800-v2/job-199116/smoke.complete.json"
worker="${repo_dir}/scripts/slurm_screenqa_backbone_7b_full_h800.sh"
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
audit_module="${repo_dir}/src/beyond_entropy/proxy_outcome_audit.py"
analyzer="${repo_dir}/scripts/analyze_visual_action_proxy_outcomes.py"
root="${BE_BB7_FULL_RUN_ROOT}"
rollout_root="${root}/rollout-shards"
merged_rollout_dir="${root}/merged-rollouts"
score_root="${root}/score-shards"
merged_score_dir="${root}/merged-scores"
analysis_dir="${root}/analysis"
execution_dir="${root}/execution"
model=Qwen/Qwen2.5-VL-7B-Instruct
model_revision=cc594898137f460bfe9f0759e9844b3ce807cfb5
manifest_sha256=4af43ac80a1666c174774d1c33383adca625e1ef4fc535ffb74e627f149290d0
protocol_sha256=1cd70d11168e12a2855ec01e8a869d89b82c4e87c3d864c566ed7db02bb61474
population_activation_sha256=a26b8bc6e8a7c81df3cad59f05ac3c9b35b6c340e71f5596175556cd0af6ee6e
analysis_implementation_sha256=3da107bef0fa8614e6cb088f4e54745cba9d8dcb872b5764c705c12bdf773eb1
hardware_activation_sha256=4bf3d87ad4af3166d1f163fea0878077f213ca2a23977526b70d4979dded8948
smoke_completion_sha256=e944437165523b4dab5261822abbeb002872f068d0ecd70b7af023688ae64e11
bound_collector_sha256=6512131e7a9bbe55b65f9229a044df43e0fa9c4564e4c20fca060a2a17059346
bound_backend_sha256=5ee063fb3d8abe3461186e7185960afd002848f1f31aad7b1fdbc1fc53840acb
bound_rollout_module_sha256=b4e30265e3b0d9bd69119ffd32901679ccd2b59140d7785c299e14465deff455
bound_crops_sha256=ddbd23e1f3e7930f1ae187aa325f0a26406e8cfa78fbf65a525ea43a22b138bf
bound_benchmarks_sha256=d96d95f3814209822f724f131175058dda7044f6fb70b402aae27a806d1a30fc
bound_rollout_merger_module_sha256=b480e939017774dcd5dab483eeb5864425b046468dbe2356d006408063d347b5
bound_rollout_merger_sha256=5ddd3fcbff9d21f036c75efa8591ab70e3cd9a311e7bd6d679dafcb251061744
bound_score_module_sha256=afcf8ec83e513d855532bf64b7ecc61911a21776b005220d4ec2f8a64e18f470
bound_scorer_sha256=230e1cf2d8e264d9092c0b1c390dbd29029049635911455757d52f3ad9062be4
bound_score_merger_sha256=4e5c8f2a97e9bdfed835f592e6cc9e52138134e4b6d1cfcd855c013b05f5974d
bound_analyzer_sha256=0147a7215ac4956eb908322cce880512e6961ee8ba1cf6ce4321c5084c22e266
bound_audit_module_sha256=7ad2fe4a710e60ca3d1d7f69584c9344c2eee6533e176ddb5e28063b16dae5a4

check_hash() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Qwen-7B full ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

actual_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_revision}" != "${BE_BB7_FULL_EXPECTED_CODE_REVISION}" ]]; then
  echo "Qwen-7B full code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before Qwen-7B full execution" >&2
  exit 2
fi

check_hash "${manifest}" "${manifest_sha256}" manifest
check_hash "${protocol}" "${protocol_sha256}" protocol
check_hash "${population_activation}" "${population_activation_sha256}" "population activation"
check_hash "${analysis_implementation}" "${analysis_implementation_sha256}" "analysis implementation"
check_hash "${hardware_activation}" "${hardware_activation_sha256}" "hardware activation"
check_hash "${smoke_completion}" "${smoke_completion_sha256}" "smoke completion"
check_hash "${worker}" "${BE_BB7_FULL_WORKER_SHA256}" worker
check_hash "${collector}" "${BE_BB7_FULL_COLLECTOR_SHA256}" collector
check_hash "${backend}" "${BE_BB7_FULL_BACKEND_SHA256}" backend
check_hash "${rollout_module}" "${BE_BB7_FULL_ROLLOUT_SHA256}" rollout
check_hash "${crops_module}" "${BE_BB7_FULL_CROPS_SHA256}" crops
check_hash "${benchmarks_module}" "${BE_BB7_FULL_BENCHMARKS_SHA256}" benchmarks
check_hash "${rollout_merger_module}" "${BE_BB7_FULL_ROLLOUT_MERGER_MODULE_SHA256}" "rollout merger module"
check_hash "${rollout_merger}" "${BE_BB7_FULL_ROLLOUT_MERGER_CLI_SHA256}" "rollout merger CLI"
check_hash "${score_module}" "${BE_BB7_FULL_SCORE_MODULE_SHA256}" "score module"
check_hash "${scorer}" "${BE_BB7_FULL_SCORER_SHA256}" scorer
check_hash "${score_merger}" "${BE_BB7_FULL_SCORE_MERGER_SHA256}" "score merger"
check_hash "${analyzer}" "${BE_BB7_FULL_ANALYZER_SHA256}" analyzer
check_hash "${audit_module}" "${BE_BB7_FULL_AUDIT_MODULE_SHA256}" "audit module"
for pair in \
  "${BE_BB7_FULL_COLLECTOR_SHA256}:${bound_collector_sha256}" \
  "${BE_BB7_FULL_BACKEND_SHA256}:${bound_backend_sha256}" \
  "${BE_BB7_FULL_ROLLOUT_SHA256}:${bound_rollout_module_sha256}" \
  "${BE_BB7_FULL_CROPS_SHA256}:${bound_crops_sha256}" \
  "${BE_BB7_FULL_BENCHMARKS_SHA256}:${bound_benchmarks_sha256}" \
  "${BE_BB7_FULL_ROLLOUT_MERGER_MODULE_SHA256}:${bound_rollout_merger_module_sha256}" \
  "${BE_BB7_FULL_ROLLOUT_MERGER_CLI_SHA256}:${bound_rollout_merger_sha256}" \
  "${BE_BB7_FULL_SCORE_MODULE_SHA256}:${bound_score_module_sha256}" \
  "${BE_BB7_FULL_SCORER_SHA256}:${bound_scorer_sha256}" \
  "${BE_BB7_FULL_SCORE_MERGER_SHA256}:${bound_score_merger_sha256}" \
  "${BE_BB7_FULL_ANALYZER_SHA256}:${bound_analyzer_sha256}" \
  "${BE_BB7_FULL_AUDIT_MODULE_SHA256}:${bound_audit_module_sha256}"; do
  if [[ "${pair%%:*}" != "${pair#*:}" ]]; then
    echo "Qwen-7B full submitted component differs from frozen implementation" >&2
    exit 2
  fi
done
if [[ "${BE_BB7_FULL_PROTOCOL_SHA256}" != "${protocol_sha256}" \
  || "${BE_BB7_FULL_POPULATION_ACTIVATION_SHA256}" != "${population_activation_sha256}" \
  || "${BE_BB7_FULL_ANALYSIS_IMPLEMENTATION_SHA256}" != "${analysis_implementation_sha256}" \
  || "${BE_BB7_FULL_HARDWARE_ACTIVATION_SHA256}" != "${hardware_activation_sha256}" \
  || "${BE_BB7_FULL_SMOKE_COMPLETION_SHA256}" != "${smoke_completion_sha256}" ]]; then
  echo "Qwen-7B full submitted protocol bundle mismatch" >&2
  exit 2
fi
if [[ "$(jq -r '.passed' "${smoke_completion}")" != true \
  || "$(jq -r '.accelerator_name' "${smoke_completion}")" != *H800* \
  || "$(jq -r '.outcome_use.task_endpoints_computed' "${smoke_completion}")" != false \
  || "$(jq -r '.outcome_use.hardware_selected_from_task_outcomes' "${smoke_completion}")" != false \
  || "$(jq -r '.outcome_use.protected_role_inputs_used' "${smoke_completion}")" != false ]]; then
  echo "Qwen-7B full smoke contract failed" >&2
  exit 2
fi
for protected in \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/calibration-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/formal-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/reserve-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/untouched-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/validation-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/test-manifest-v1"; do
  if [[ -e "${protected}" && ( ! -d "${protected}" || -n "$(find "${protected}" -mindepth 1 -print -quit)" ) ]]; then
    echo "ScreenQA protected role opened before Qwen-7B full diagnostic: ${protected}" >&2
    exit 2
  fi
done
if [[ -d "${root}" && -n "$(find "${root}" -mindepth 1 -print -quit)" \
  && "${BE_BB7_FULL_RESUME}" != 1 ]]; then
  echo "existing Qwen-7B full outputs require explicit audited resume" >&2
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
mkdir -p "${rollout_root}" "${merged_rollout_dir}" "${score_root}" \
  "${merged_score_dir}" "${analysis_dir}" "${execution_dir}"
cd "${repo_dir}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Slurm did not expose four H800 devices" >&2
  exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_gpus[@]}" -ne 4 ]]; then
  echo "Qwen-7B full expected four CUDA devices, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

job_start_epoch=$(date +%s)
queue_wait_seconds=$((job_start_epoch - BE_BB7_SUBMIT_EPOCH))
if [[ "${queue_wait_seconds}" -lt 0 ]]; then
  echo "Qwen-7B full submission epoch is in the future" >&2
  exit 2
fi

run_collect_shard() {
  local shard_index=$1
  local log_name=$2
  local shard_name
  shard_name=$(printf 'shard-%05d-of-00004' "${shard_index}")
  local shard_dir="${rollout_root}/${shard_name}"
  mkdir -p "${shard_dir}"
  CUDA_VISIBLE_DEVICES="${allocated_gpus[${shard_index}]}" \
    "${python_bin}" -m beyond_entropy collect-qwen \
      --manifest "${manifest}" \
      --expected-manifest-sha256 "${manifest_sha256}" \
      --output "${shard_dir}/rollouts.jsonl" \
      --resume \
      --checkpoint-interval 16 \
      --shard-count 4 \
      --shard-index "${shard_index}" \
      --model "${model}" \
      --model-revision "${model_revision}" \
      --scorer screenqa \
      --candidate-count 4 \
      --proposer ug-grid \
      --visual-crop-ratio 2.0 \
      --visual-cost 1.0 \
      --generation-seeds 0 \
      --bootstrap-resamples 0 \
      --bootstrap-seed 20260903 \
      --scientific-status "frozen Qwen-7B mechanism diagnostic on opened ScreenQA ranker development; no protected role and not candidate selection" \
      --max-new-tokens 32 \
      --min-pixels 200704 \
      --max-pixels 602112 \
      --attention-implementation sdpa \
      --system-prompt "You are a helpful assistant." \
      > "${shard_dir}/${log_name}" 2>&1
}

rollout_start=$(date +%s)
pids=()
for shard_index in 0 1 2 3; do
  run_collect_shard "${shard_index}" first-pass.log &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more Qwen-7B rollout shards failed; checkpoints retained" >&2
  exit 1
fi
rollout_first_seconds=$(( $(date +%s) - rollout_start ))
for shard_index in 0 1 2 3; do
  shard_name=$(printf 'shard-%05d-of-00004' "${shard_index}")
  shard_dir="${rollout_root}/${shard_name}"
  cp "${shard_dir}/rollouts.provenance.json" "${shard_dir}/rollouts.first-pass.provenance.json"
  sha256sum "${shard_dir}/rollouts.jsonl" | cut -d ' ' -f 1 > "${shard_dir}/rollouts.first-pass.sha256"
done

rollout_resume_start=$(date +%s)
pids=()
for shard_index in 0 1 2 3; do
  run_collect_shard "${shard_index}" resume.log &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more Qwen-7B rollout resume checks failed" >&2
  exit 1
fi
rollout_resume_seconds=$(( $(date +%s) - rollout_resume_start ))
for shard_index in 0 1 2 3; do
  shard_name=$(printf 'shard-%05d-of-00004' "${shard_index}")
  shard_dir="${rollout_root}/${shard_name}"
  expected_sha=$(<"${shard_dir}/rollouts.first-pass.sha256")
  actual_sha=$(sha256sum "${shard_dir}/rollouts.jsonl" | cut -d ' ' -f 1)
  records=$(wc -l < "${shard_dir}/rollouts.jsonl")
  resumed=$(jq -r '.resumed_from_records' "${shard_dir}/rollouts.provenance.json")
  if [[ "${actual_sha}" != "${expected_sha}" || "${resumed}" -ne "${records}" ]]; then
    echo "Qwen-7B rollout shard ${shard_index} resume audit failed" >&2
    exit 2
  fi
  jq -n \
    --arg before "${expected_sha}" --arg after "${actual_sha}" \
    --argjson records "${records}" --argjson resumed "${resumed}" \
    '{passed:true,records:$records,resumed_from_records:$resumed,rollouts_sha256_before_resume:$before,rollouts_sha256_after_resume:$after}' \
    > "${shard_dir}/resume.audit.json.tmp"
  mv "${shard_dir}/resume.audit.json.tmp" "${shard_dir}/resume.audit.json"
done

merged_rollouts="${merged_rollout_dir}/rollouts.jsonl"
"${python_bin}" "${rollout_merger}" \
  --manifest "${manifest}" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --run-root "${rollout_root}" \
  --shard-count 4 \
  --output "${merged_rollouts}" \
  --expected-code-revision "${actual_revision}" \
  --expected-scorer screenqa \
  --require-resume-audit \
  --bootstrap-resamples 100 \
  --bootstrap-seed 20260903
merged_rollouts_sha256=$(sha256sum "${merged_rollouts}" | cut -d ' ' -f 1)
if [[ "$(wc -l < "${merged_rollouts}")" -ne 2560 \
  || "$(jq -r '.selected_states' "${merged_rollout_dir}/rollouts.merge.json")" -ne 512 \
  || "$(jq -r '.merged_records' "${merged_rollout_dir}/rollouts.merge.json")" -ne 2560 ]]; then
  echo "Qwen-7B merged rollout dimensions failed" >&2
  exit 2
fi

run_score_shard() {
  local shard_index=$1
  local log_name=$2
  local output="${score_root}/answer-nll-shard-${shard_index}-of-4.jsonl"
  CUDA_VISIBLE_DEVICES="${allocated_gpus[${shard_index}]}" \
    "${python_bin}" "${scorer}" \
      --manifest "${manifest}" \
      --rollouts "${merged_rollouts}" \
      --output "${output}" \
      --expected-manifest-sha256 "${manifest_sha256}" \
      --expected-rollouts-sha256 "${merged_rollouts_sha256}" \
      --shard-count 4 \
      --shard-index "${shard_index}" \
      --checkpoint-interval 16 \
      --resume \
      --model "${model}" \
      --model-revision "${model_revision}" \
      --device-map cuda:0 \
      --dtype bfloat16 \
      --attention-implementation sdpa \
      --min-pixels 200704 \
      --max-pixels 602112 \
      --system-prompt "You are a helpful assistant." \
      --code-revision "${actual_revision}" \
      --scientific-status "frozen Qwen-7B mechanism diagnostic on opened ScreenQA ranker development; no protected role and not candidate selection" \
      > "${score_root}/shard-${shard_index}-${log_name}" 2>&1
}

score_start=$(date +%s)
pids=()
for shard_index in 0 1 2 3; do
  run_score_shard "${shard_index}" first-pass.log &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more Qwen-7B NLL shards failed; checkpoints retained" >&2
  exit 1
fi
score_first_seconds=$(( $(date +%s) - score_start ))
for shard_index in 0 1 2 3; do
  output="${score_root}/answer-nll-shard-${shard_index}-of-4.jsonl"
  provenance="${output%.jsonl}.provenance.json"
  cp "${provenance}" "${score_root}/answer-nll-shard-${shard_index}-of-4.first-pass.provenance.json"
  sha256sum "${output}" | cut -d ' ' -f 1 > "${score_root}/answer-nll-shard-${shard_index}-of-4.first-pass.sha256"
done

score_resume_start=$(date +%s)
pids=()
for shard_index in 0 1 2 3; do
  run_score_shard "${shard_index}" resume.log &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more Qwen-7B NLL resume checks failed" >&2
  exit 1
fi
score_resume_seconds=$(( $(date +%s) - score_resume_start ))
for shard_index in 0 1 2 3; do
  output="${score_root}/answer-nll-shard-${shard_index}-of-4.jsonl"
  provenance="${output%.jsonl}.provenance.json"
  expected_sha=$(<"${score_root}/answer-nll-shard-${shard_index}-of-4.first-pass.sha256")
  actual_sha=$(sha256sum "${output}" | cut -d ' ' -f 1)
  if [[ "${actual_sha}" != "${expected_sha}" \
    || "$(jq -r '.decisions' "${provenance}")" -ne 128 \
    || "$(jq -r '.records' "${provenance}")" -ne 640 \
    || "$(jq -r '.sources' "${provenance}")" -ne 128 \
    || "$(jq -r '.resumed_from_decisions' "${provenance}")" -ne 128 \
    || "$(jq -r '.raw_targets_written' "${provenance}")" != false ]]; then
    echo "Qwen-7B NLL shard ${shard_index} resume or dimension audit failed" >&2
    exit 2
  fi
done

validate_runtime() {
  local provenance=$1
  local label=$2
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
    echo "Qwen-7B ${label} runtime telemetry failed" >&2
    exit 2
  fi
}
for shard_index in 0 1 2 3; do
  shard_name=$(printf 'shard-%05d-of-00004' "${shard_index}")
  validate_runtime "${rollout_root}/${shard_name}/rollouts.provenance.json" "rollout shard ${shard_index}"
  validate_runtime "${score_root}/answer-nll-shard-${shard_index}-of-4.provenance.json" "NLL shard ${shard_index}"
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
merged_scores_sha256=$(sha256sum "${merged_scores}" | cut -d ' ' -f 1)

analysis_start=$(date +%s)
"${python_bin}" "${analyzer}" \
  --scores "${merged_scores}" \
  --protocol "${protocol}" \
  --implementation-contract "${analysis_implementation}" \
  --output-dir "${analysis_dir}" \
  --expected-scores-sha256 "${merged_scores_sha256}" \
  --expected-protocol-sha256 "${protocol_sha256}" \
  --expected-implementation-contract-sha256 "${analysis_implementation_sha256}" \
  --expected-decisions 512 \
  --expected-sources 512 \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 20260903 \
  --bootstrap-confidence 0.95 \
  --study-label "ScreenQA Qwen2.5-VL-7B opened development" \
  --scientific-status "frozen Qwen2.5-VL-7B mechanism diagnostic on opened ScreenQA ranker development; not candidate selection or independent validation" \
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
  || "$(jq -r '.study.label' "${report}")" != "ScreenQA Qwen2.5-VL-7B opened development" \
  || "$(jq -r '.outcome_use.protected_role_inputs_used' "${report}")" != false \
  || "$(jq -r '.outcome_use.calibration_or_formal_inputs_used' "${report}")" != false ]]; then
  echo "Qwen-7B full analysis contract failed" >&2
  exit 2
fi

job_end_epoch=$(date +%s)
execution_audit="${execution_dir}/job-${SLURM_JOB_ID}.json"
jq -n \
  --arg schema "qwen7b_full_execution_v1" \
  --arg job_id "${SLURM_JOB_ID}" \
  --arg code_revision "${actual_revision}" \
  --arg accelerator "NVIDIA H800" \
  --arg manifest_sha256 "${manifest_sha256}" \
  --arg rollouts_sha256 "${merged_rollouts_sha256}" \
  --arg scores_sha256 "${merged_scores_sha256}" \
  --arg report_sha256 "$(sha256sum "${report}" | cut -d ' ' -f 1)" \
  --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson rollout_first_seconds "${rollout_first_seconds}" \
  --argjson rollout_resume_seconds "${rollout_resume_seconds}" \
  --argjson score_first_seconds "${score_first_seconds}" \
  --argjson score_resume_seconds "${score_resume_seconds}" \
  --argjson analysis_seconds "${analysis_seconds}" \
  --argjson total_seconds "$((job_end_epoch - job_start_epoch))" \
  '{schema:$schema,job_id:$job_id,code_revision:$code_revision,accelerator:$accelerator,gpu_count:4,queue_wait_seconds:$queue_wait_seconds,timing_seconds:{rollout_first_pass:$rollout_first_seconds,rollout_resume_validation:$rollout_resume_seconds,answer_nll_first_pass:$score_first_seconds,answer_nll_resume_validation:$score_resume_seconds,analysis:$analysis_seconds,total:$total_seconds},artifacts:{manifest_sha256:$manifest_sha256,merged_rollouts_sha256:$rollouts_sha256,merged_scores_sha256:$scores_sha256,report_sha256:$report_sha256},protected_role_inputs_used:false}' \
  > "${execution_audit}.tmp"
mv "${execution_audit}.tmp" "${execution_audit}"

printf 'screenqa_qwen7b_full_complete=%s report_sha256=%s merged_rollouts_sha256=%s merged_scores_sha256=%s\n' \
  "${execution_audit}" "$(sha256sum "${report}" | cut -d ' ' -f 1)" \
  "${merged_rollouts_sha256}" "${merged_scores_sha256}"
