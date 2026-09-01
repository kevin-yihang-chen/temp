#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=04:00:00
#SBATCH --job-name=be-infovqa-decar-oof
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-infovqa-decar-oof-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 9 ]]; then
  echo "usage: worker REVISION WORKER_SHA FREEZE_SHA ROLLOUT_SHA NLL_SHA FEATURE_SHA INPUT_AUDIT_SHA GENERATION_EXECUTION_SHA SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_freeze_sha256=$3
rollouts_sha256=$4
nll_sha256=$5
features_sha256=$6
input_audit_sha256=$7
generation_execution_sha256=$8
submit_epoch=$9
for value in "${expected_worker_sha256}" "${expected_freeze_sha256}" \
  "${rollouts_sha256}" "${nll_sha256}" "${features_sha256}" \
  "${input_audit_sha256}" "${generation_execution_sha256}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "DECAR OOF received a malformed SHA-256" >&2
    exit 2
  fi
done
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "DECAR OOF submit epoch must be an integer" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
generation_root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
rollouts="${generation_root}/merged-rollouts/rollouts.jsonl"
answer_nll="${generation_root}/merged-nll/answer-nll.jsonl"
features="${generation_root}/merged-features/features-label-free.pt"
input_audit="${generation_root}/merged-features/decar-input-audit.json"
generation_execution_dir="${generation_root}/execution"
image_manifest="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-manifest-v1/image-manifest.jsonl"
outer_folds="${repo}/artifacts/infographicvqa-train-v1/decar-v1/allocation-v1/outer-folds.jsonl"
inner_folds="${repo}/artifacts/infographicvqa-train-v1/decar-v1/allocation-v1/inner-folds.jsonl"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-method-protocol-v1.md"
freeze="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-oof-evaluation-freeze-v1.md"
worker="${repo}/scripts/slurm_infographicvqa_decar_oof_h800.sh"
fit_runner="${repo}/scripts/fit_infographicvqa_decar_oof.py"
evaluation_runner="${repo}/scripts/evaluate_infographicvqa_decar_oof.py"
fit_dir="${generation_root}/nested-oof-v1"
evaluation_dir="${generation_root}/evaluation-v1"
execution_dir="${generation_root}/oof-execution"

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "DECAR OOF ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "DECAR OOF tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "DECAR OOF tracked worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${freeze}" "${expected_freeze_sha256}" freeze
require_hash "${rollouts}" "${rollouts_sha256}" rollouts
require_hash "${answer_nll}" "${nll_sha256}" answer-nll
require_hash "${features}" "${features_sha256}" features
require_hash "${input_audit}" "${input_audit_sha256}" input-audit
require_hash "${image_manifest}" 0916a6b5a32e15c4f5b3bf920e1ecd4f304aeb97ae186e0e3e846391e2304203 image-manifest
require_hash "${outer_folds}" 7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6b8cdb6d0af5a4da60 outer-folds
require_hash "${inner_folds}" 8b4977776841794185eedd8cbbfdb8e23ec2c44cc0be9931cc2e672b232f344c inner-folds
require_hash "${protocol}" d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342 protocol
mapfile -t generation_executions < <(find "${generation_execution_dir}" -maxdepth 1 -type f -name 'job-*.json' -print | sort)
if [[ "${#generation_executions[@]}" -ne 1 ]]; then
  echo "DECAR OOF requires exactly one generation execution record" >&2
  exit 2
fi
generation_execution=${generation_executions[0]}
require_hash "${generation_execution}" "${generation_execution_sha256}" generation-execution
if ! jq -e \
  --arg rollout "${rollouts_sha256}" --arg nll "${nll_sha256}" \
  --arg features "${features_sha256}" --arg audit "${input_audit_sha256}" '
    .schema == "infographicvqa_decar_full_generation_execution_v1" and
    .artifacts.merged_rollouts_sha256 == $rollout and
    .artifacts.merged_teacher_nll_sha256 == $nll and
    .artifacts.merged_label_free_features_sha256 == $features and
    .artifacts.decar_input_audit_sha256 == $audit and
    .population.questions == 23946 and .population.sources == 2204 and
    .population.images == 4406 and .population.actions_per_question == 5 and
    .generated_token_statistics_complete == true and
    .predictions_computed == false and
    .scientific_endpoints_used_for_selection == false and
    .validation_or_test_inputs_used == false
  ' "${generation_execution}" >/dev/null; then
  echo "DECAR OOF generation execution contract failed" >&2
  exit 2
fi
if ! jq -e '
  .passed == true and .decisions == 23946 and .sources == 2204 and
  .images == 4406 and .actions_per_decision == 5 and .scalar_dim == 16 and
  .generated_token_statistics_complete == true and
  .label_free_feature_storage == true and
  .inference_feature_outcomes_included == false and
  .scientific_endpoints_reported == false
' "${input_audit}" >/dev/null; then
  echo "DECAR OOF strict input audit failed" >&2
  exit 2
fi
if [[ -e "${fit_dir}" || -e "${evaluation_dir}" ]]; then
  echo "DECAR OOF refuses to overwrite fit or evaluation output" >&2
  exit 2
fi
mkdir -p "${execution_dir}"

export PYTHONPATH="${repo}/src"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Slurm exposed no GPU to DECAR OOF" >&2
  exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_gpus[@]}" -ne 1 ]]; then
  echo "DECAR OOF requires exactly one H800" >&2
  exit 2
fi
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ "${gpu_name}" != "NVIDIA H800" ]]; then
  echo "DECAR OOF accelerator changed: ${gpu_name}" >&2
  exit 2
fi

job_start_epoch=$(date +%s)
queue_wait_seconds=$((job_start_epoch - submit_epoch))
if [[ "${queue_wait_seconds}" -lt 0 ]]; then
  echo "DECAR OOF submit epoch is in the future" >&2
  exit 2
fi
echo "DECAR OOF start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

fit_start=$(date +%s)
"${python_bin}" "${fit_runner}" \
  --rollouts "${rollouts}" --expected-rollouts-sha256 "${rollouts_sha256}" \
  --answer-nll "${answer_nll}" --expected-answer-nll-sha256 "${nll_sha256}" \
  --features "${features}" --expected-features-sha256 "${features_sha256}" \
  --image-manifest "${image_manifest}" --expected-image-manifest-sha256 0916a6b5a32e15c4f5b3bf920e1ecd4f304aeb97ae186e0e3e846391e2304203 \
  --outer-folds "${outer_folds}" --expected-outer-folds-sha256 7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6b8cdb6d0af5a4da60 \
  --inner-folds "${inner_folds}" --expected-inner-folds-sha256 8b4977776841794185eedd8cbbfdb8e23ec2c44cc0be9931cc2e672b232f344c \
  --protocol "${protocol}" --expected-protocol-sha256 d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342 \
  --device cuda:0 --epochs 200 --output-dir "${fit_dir}"
fit_seconds=$(( $(date +%s) - fit_start ))

fit_predictions="${fit_dir}/predictions.jsonl"
fit_audit="${fit_dir}/audit.json"
fit_report="${fit_dir}/report.json"
fit_complete="${fit_dir}/complete.json"
if [[ "$(jq -r '.prediction_rows' "${fit_complete}")" -ne 23946 \
  || "$(jq -r '.prediction_outcomes_included' "${fit_complete}")" != false \
  || "$(jq -r '.scientific_endpoints_computed' "${fit_report}")" != false \
  || "$(jq -r '.scientific_endpoints_read' "${fit_report}")" != false ]]; then
  echo "DECAR OOF fit output contract failed" >&2
  exit 2
fi
predictions_sha256=$(sha "${fit_predictions}")
fit_audit_sha256=$(sha "${fit_audit}")
fit_report_sha256=$(sha "${fit_report}")
fit_complete_sha256=$(sha "${fit_complete}")

evaluation_start=$(date +%s)
"${python_bin}" "${evaluation_runner}" \
  --rollouts "${rollouts}" --expected-rollouts-sha256 "${rollouts_sha256}" \
  --predictions "${fit_predictions}" --expected-predictions-sha256 "${predictions_sha256}" \
  --oof-complete "${fit_complete}" --expected-oof-complete-sha256 "${fit_complete_sha256}" \
  --oof-audit "${fit_audit}" --expected-oof-audit-sha256 "${fit_audit_sha256}" \
  --oof-report "${fit_report}" --expected-oof-report-sha256 "${fit_report_sha256}" \
  --protocol "${protocol}" --expected-protocol-sha256 d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342 \
  --output-dir "${evaluation_dir}"
evaluation_seconds=$(( $(date +%s) - evaluation_start ))

evaluation_complete="${evaluation_dir}/complete.json"
evaluation_json="${evaluation_dir}/evaluation.json"
decision=$(jq -r '.decision' "${evaluation_complete}")
if [[ "${decision}" != decar_advanced_to_sealed_validation \
  && "${decision}" != decar_not_advanced ]]; then
  echo "DECAR OOF evaluation decision is invalid" >&2
  exit 2
fi
if [[ "$(jq -r '.validation_or_test_inputs_used' "${evaluation_complete}")" != false \
  || "$(jq -r '.population.decisions' "${evaluation_json}")" -ne 23946 \
  || "$(jq -r '.bootstrap.n_resamples' "${evaluation_json}")" -ne 20000 \
  || "$(jq -r '.bootstrap.seed' "${evaluation_json}")" -ne 20260917 ]]; then
  echo "DECAR OOF evaluation output contract failed" >&2
  exit 2
fi

job_end_epoch=$(date +%s)
execution="${execution_dir}/job-${SLURM_JOB_ID}.json"
jq -n \
  --arg schema infographicvqa_decar_oof_execution_v1 \
  --arg job_id "${SLURM_JOB_ID}" --arg code_revision "${expected_revision}" \
  --arg accelerator "${gpu_name}" --arg generation_execution_sha256 "${generation_execution_sha256}" \
  --arg predictions_sha256 "${predictions_sha256}" --arg fit_audit_sha256 "${fit_audit_sha256}" \
  --arg fit_report_sha256 "${fit_report_sha256}" --arg fit_complete_sha256 "${fit_complete_sha256}" \
  --arg evaluation_sha256 "$(sha "${evaluation_json}")" \
  --arg evaluation_complete_sha256 "$(sha "${evaluation_complete}")" \
  --arg decision "${decision}" --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson fit_seconds "${fit_seconds}" --argjson evaluation_seconds "${evaluation_seconds}" \
  --argjson total_seconds "$((job_end_epoch - job_start_epoch))" '
  {schema:$schema,job_id:$job_id,code_revision:$code_revision,accelerator:$accelerator,gpu_count:1,
   queue_wait_seconds:$queue_wait_seconds,timing_seconds:{fit:$fit_seconds,evaluation:$evaluation_seconds,total:$total_seconds},
   inputs:{generation_execution_sha256:$generation_execution_sha256},
   artifacts:{predictions_sha256:$predictions_sha256,fit_audit_sha256:$fit_audit_sha256,
   fit_report_sha256:$fit_report_sha256,fit_complete_sha256:$fit_complete_sha256,
   evaluation_sha256:$evaluation_sha256,evaluation_complete_sha256:$evaluation_complete_sha256},
   decision:$decision,validation_or_test_inputs_used:false}' > "${execution}.tmp"
mv "${execution}.tmp" "${execution}"
echo "DECAR OOF end: $(date --iso-8601=seconds)"
printf 'infographicvqa_decar_oof_complete=%s decision=%s execution_sha256=%s\n' \
  "${evaluation_complete}" "${decision}" "$(sha "${execution}")"
