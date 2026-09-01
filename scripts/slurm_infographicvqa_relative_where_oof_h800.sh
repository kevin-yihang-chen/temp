#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --job-name=be-infovqa-relative-where
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-infovqa-relative-where-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 10 ]]; then
  echo "usage: worker REVISION WORKER_SHA FIT_RUNNER_SHA EVAL_RUNNER_SHA TRAIN_MODULE_SHA EVAL_MODULE_SHA DECAR_EVAL_SHA PROTOCOL_SHA RESOURCE_SHA SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_fit_runner_sha256=$3
expected_eval_runner_sha256=$4
expected_train_module_sha256=$5
expected_eval_module_sha256=$6
expected_decar_eval_sha256=$7
expected_protocol_sha256=$8
resource_amendment_sha256=$9
submit_epoch=${10}
for value in "${expected_worker_sha256}" "${expected_fit_runner_sha256}" \
  "${expected_eval_runner_sha256}" "${expected_train_module_sha256}" \
  "${expected_eval_module_sha256}" "${expected_decar_eval_sha256}" \
  "${expected_protocol_sha256}" "${resource_amendment_sha256}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "relative-where received a malformed SHA-256" >&2
    exit 2
  fi
done
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "relative-where submit epoch must be an integer" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
rollouts="${root}/merged-rollouts/rollouts.jsonl"
answer_nll="${root}/merged-nll/answer-nll.jsonl"
features="${root}/merged-features/features-label-free.pt"
input_audit="${root}/merged-features/decar-input-audit.json"
generation_execution="${root}/execution/job-200130.json"
image_manifest="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-manifest-v1/image-manifest.jsonl"
outer_folds="${repo}/artifacts/infographicvqa-train-v1/decar-v1/allocation-v1/outer-folds.jsonl"
inner_folds="${repo}/artifacts/infographicvqa-train-v1/decar-v1/allocation-v1/inner-folds.jsonl"
decar_predictions="${root}/nested-oof-v1/predictions.jsonl"
decar_complete="${root}/nested-oof-v1/complete.json"
decar_audit="${root}/nested-oof-v1/audit.json"
decar_report="${root}/nested-oof-v1/report.json"
hybrid_evaluation="${root}/entropy-where-hybrid-v1/evaluation.json"
hybrid_complete="${root}/entropy-where-hybrid-v1/complete.json"
oracle_evaluation="${root}/entropy-oracle-where-factorization-v1/evaluation.json"
oracle_complete="${root}/entropy-oracle-where-factorization-v1/complete.json"
bootstrap_indices="${root}/evaluation-v1/bootstrap-indices.npy"
bootstrap_sources="${root}/evaluation-v1/bootstrap-sources.json"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-relative-where-oof-protocol-v1.md"
design_audit="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-relative-where-design-audit-20260901.md"
oracle_result="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-entropy-oracle-where-factorization-result-job-203078-v1.md"
resource_amendment="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-relative-where-resource-amendment-v1.md"
worker="${repo}/scripts/slurm_infographicvqa_relative_where_oof_h800.sh"
fit_runner="${repo}/scripts/fit_infographicvqa_relative_where_oof.py"
eval_runner="${repo}/scripts/evaluate_infographicvqa_relative_where_oof.py"
train_module="${repo}/src/beyond_entropy/infographicvqa_relative_where.py"
eval_module="${repo}/src/beyond_entropy/infographicvqa_relative_where_evaluation.py"
decar_eval_module="${repo}/src/beyond_entropy/infographicvqa_decar_evaluation.py"
fit_dir="${root}/relative-where-oof-v1"
evaluation_dir="${fit_dir}/evaluation-v1"
execution_dir="${root}/relative-where-execution"

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "relative-where ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "relative-where tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "relative-where tracked worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${fit_runner}" "${expected_fit_runner_sha256}" fit-runner
require_hash "${eval_runner}" "${expected_eval_runner_sha256}" eval-runner
require_hash "${train_module}" "${expected_train_module_sha256}" train-module
require_hash "${eval_module}" "${expected_eval_module_sha256}" eval-module
require_hash "${decar_eval_module}" "${expected_decar_eval_sha256}" decar-eval-module
require_hash "${protocol}" "${expected_protocol_sha256}" protocol
require_hash "${resource_amendment}" "${resource_amendment_sha256}" resource-amendment
require_hash "${rollouts}" 9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e rollouts
require_hash "${answer_nll}" 884de4ebc1ba83226393871cdfff819bc7553a61eee6ffa11697db03332ac646 answer-nll
require_hash "${features}" d0508726a50b4c1e54778392d08329b242a680fc13292cac1ebec8b42a175300 features
require_hash "${input_audit}" 00cdf507b39904d778b0f813bb86183d51239f0d8d44dc7596de08e48e7bbd8a input-audit
require_hash "${generation_execution}" 58392547b2aa288847dee56b894cf53ba5fa907647541ef7541f1eff38b3aab3 generation-execution
require_hash "${image_manifest}" 0916a6b5a32e15c4f5b3bf920e1ecd4f304aeb97ae186e0e3e846391e2304203 image-manifest
require_hash "${outer_folds}" 7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6a232b0837a outer-folds
require_hash "${inner_folds}" 8b4977776841794185eedd8cbbfdb8e23ec2c44cc0be9931cc2e672b232f344c inner-folds
require_hash "${decar_predictions}" c8338b1960ca223c892c3f992b0ccc3027d543113f67c82a734eccd7e3699c4b decar-predictions
require_hash "${decar_complete}" 8de073870fcade5ac111d59de81e9c70dc567c9900e0b223cdebca6a8318f31f decar-complete
require_hash "${decar_audit}" dc3193dfc626a3df50321f4d92a336ce784aee34f1f2e91c57cf87d1f8085537 decar-audit
require_hash "${decar_report}" ebc936739e970fbfda25ebe02ef71d6b7f46674f9d00092011b5665c2daa9bf0 decar-report
require_hash "${hybrid_evaluation}" ab4a179c2141de60c9a1d173e34a7166d075935e5283569439e3d93424344a62 hybrid-evaluation
require_hash "${hybrid_complete}" 4e3b3c2b2b2e1698fcf9bb3c9e71881b11e1357982bb1b6634a12af6a7aa03ac hybrid-complete
require_hash "${oracle_evaluation}" 6ef0869b453e1a70ad5f479e8a9604aa04ac0419a1dc92d1ce353c54f66f3025 oracle-evaluation
require_hash "${oracle_complete}" b940258389e558d0d0bae277bd8d5b081923ce505ca2e6ad8520a79bd6411de7 oracle-complete
require_hash "${bootstrap_indices}" 17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6 bootstrap-indices
require_hash "${bootstrap_sources}" 5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0 bootstrap-sources
require_hash "${design_audit}" c35f685b5b83af00cf2132cdcf8083e46566e3696bd82ec0a8ad4fe0adc0550b design-audit
require_hash "${oracle_result}" c520dc3fd9a5d25c0b7e626a88a55e629e56c8a6d1a473feb53d94bad4689cf0 oracle-result
if [[ -e "${fit_dir}" ]]; then
  echo "relative-where refuses to overwrite fit/evaluation output" >&2
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
  echo "relative-where strict input audit failed" >&2
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
  echo "Slurm exposed no GPU to relative-where" >&2
  exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_gpus[@]}" -ne 1 ]]; then
  echo "relative-where requires exactly one H800" >&2
  exit 2
fi
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ "${gpu_name}" != "NVIDIA H800" ]]; then
  echo "relative-where accelerator changed: ${gpu_name}" >&2
  exit 2
fi

job_start_epoch=$(date +%s)
queue_wait_seconds=$((job_start_epoch - submit_epoch))
if [[ "${queue_wait_seconds}" -lt 0 ]]; then
  echo "relative-where submit epoch is in the future" >&2
  exit 2
fi
echo "Relative-where start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

fit_start=$(date +%s)
"${python_bin}" "${fit_runner}" \
  --rollouts "${rollouts}" --expected-rollouts-sha256 9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e \
  --answer-nll "${answer_nll}" --expected-answer-nll-sha256 884de4ebc1ba83226393871cdfff819bc7553a61eee6ffa11697db03332ac646 \
  --features "${features}" --expected-features-sha256 d0508726a50b4c1e54778392d08329b242a680fc13292cac1ebec8b42a175300 \
  --image-manifest "${image_manifest}" --expected-image-manifest-sha256 0916a6b5a32e15c4f5b3bf920e1ecd4f304aeb97ae186e0e3e846391e2304203 \
  --outer-folds "${outer_folds}" --expected-outer-folds-sha256 7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6a232b0837a \
  --inner-folds "${inner_folds}" --expected-inner-folds-sha256 8b4977776841794185eedd8cbbfdb8e23ec2c44cc0be9931cc2e672b232f344c \
  --protocol "${protocol}" --expected-protocol-sha256 "${expected_protocol_sha256}" \
  --device cuda:0 --epochs 200 --output-dir "${fit_dir}"
fit_seconds=$(( $(date +%s) - fit_start ))

relative_predictions="${fit_dir}/predictions.jsonl"
relative_audit="${fit_dir}/audit.json"
relative_report="${fit_dir}/report.json"
relative_complete="${fit_dir}/complete.json"
if [[ "$(jq -r '.prediction_rows' "${relative_complete}")" -ne 23946 \
  || "$(jq -r '.prediction_outcomes_included' "${relative_complete}")" != false \
  || "$(jq -r '.scientific_endpoints_computed' "${relative_complete}")" != false \
  || "$(jq -r '.validation_or_test_inputs_used' "${relative_complete}")" != false ]]; then
  echo "relative-where fit output contract failed" >&2
  exit 2
fi
relative_predictions_sha256=$(sha "${relative_predictions}")
relative_audit_sha256=$(sha "${relative_audit}")
relative_report_sha256=$(sha "${relative_report}")
relative_complete_sha256=$(sha "${relative_complete}")

evaluation_start=$(date +%s)
"${python_bin}" "${eval_runner}" \
  --rollouts "${rollouts}" --expected-rollouts-sha256 9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e \
  --answer-nll "${answer_nll}" --expected-answer-nll-sha256 884de4ebc1ba83226393871cdfff819bc7553a61eee6ffa11697db03332ac646 \
  --relative-predictions "${relative_predictions}" --expected-relative-predictions-sha256 "${relative_predictions_sha256}" \
  --relative-complete "${relative_complete}" --expected-relative-complete-sha256 "${relative_complete_sha256}" \
  --relative-audit "${relative_audit}" --expected-relative-audit-sha256 "${relative_audit_sha256}" \
  --relative-report "${relative_report}" --expected-relative-report-sha256 "${relative_report_sha256}" \
  --decar-predictions "${decar_predictions}" --expected-decar-predictions-sha256 c8338b1960ca223c892c3f992b0ccc3027d543113f67c82a734eccd7e3699c4b \
  --decar-complete "${decar_complete}" --expected-decar-complete-sha256 8de073870fcade5ac111d59de81e9c70dc567c9900e0b223cdebca6a8318f31f \
  --decar-audit "${decar_audit}" --expected-decar-audit-sha256 dc3193dfc626a3df50321f4d92a336ce784aee34f1f2e91c57cf87d1f8085537 \
  --decar-report "${decar_report}" --expected-decar-report-sha256 ebc936739e970fbfda25ebe02ef71d6b7f46674f9d00092011b5665c2daa9bf0 \
  --hybrid-evaluation "${hybrid_evaluation}" --expected-hybrid-evaluation-sha256 ab4a179c2141de60c9a1d173e34a7166d075935e5283569439e3d93424344a62 \
  --hybrid-complete "${hybrid_complete}" --expected-hybrid-complete-sha256 4e3b3c2b2b2e1698fcf9bb3c9e71881b11e1357982bb1b6634a12af6a7aa03ac \
  --oracle-evaluation "${oracle_evaluation}" --expected-oracle-evaluation-sha256 6ef0869b453e1a70ad5f479e8a9604aa04ac0419a1dc92d1ce353c54f66f3025 \
  --oracle-complete "${oracle_complete}" --expected-oracle-complete-sha256 b940258389e558d0d0bae277bd8d5b081923ce505ca2e6ad8520a79bd6411de7 \
  --bootstrap-indices "${bootstrap_indices}" --expected-bootstrap-indices-sha256 17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6 \
  --bootstrap-sources "${bootstrap_sources}" --expected-bootstrap-sources-sha256 5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0 \
  --protocol "${protocol}" --expected-protocol-sha256 "${expected_protocol_sha256}" \
  --design-audit "${design_audit}" --expected-design-audit-sha256 c35f685b5b83af00cf2132cdcf8083e46566e3696bd82ec0a8ad4fe0adc0550b \
  --oracle-result "${oracle_result}" --expected-oracle-result-sha256 c520dc3fd9a5d25c0b7e626a88a55e629e56c8a6d1a473feb53d94bad4689cf0 \
  --output-dir "${evaluation_dir}"
evaluation_seconds=$(( $(date +%s) - evaluation_start ))

evaluation_complete="${evaluation_dir}/complete.json"
evaluation_json="${evaluation_dir}/evaluation.json"
decision=$(jq -r '.decision' "${evaluation_complete}")
if [[ "${decision}" != relative_where_train_supported \
  && "${decision}" != relative_where_train_not_supported ]]; then
  echo "relative-where evaluation decision is invalid" >&2
  exit 2
fi
if [[ "$(jq -r '.validation_or_test_inputs_used' "${evaluation_complete}")" != false \
  || "$(jq -r '.relative_prediction_outcomes_included' "${evaluation_complete}")" != false \
  || "$(jq -r '.population.decisions' "${evaluation_json}")" -ne 23946 \
  || "$(jq -r '.bootstrap.n_resamples' "${evaluation_json}")" -ne 20000 ]]; then
  echo "relative-where evaluation output contract failed" >&2
  exit 2
fi

job_end_epoch=$(date +%s)
execution="${execution_dir}/job-${SLURM_JOB_ID}.json"
jq -n \
  --arg schema infographicvqa_relative_where_oof_execution_v1 \
  --arg job_id "${SLURM_JOB_ID}" --arg code_revision "${expected_revision}" \
  --arg accelerator "${gpu_name}" --arg protocol_sha256 "${expected_protocol_sha256}" \
  --arg resource_amendment_sha256 "${resource_amendment_sha256}" \
  --arg predictions_sha256 "${relative_predictions_sha256}" \
  --arg fit_audit_sha256 "${relative_audit_sha256}" \
  --arg fit_report_sha256 "${relative_report_sha256}" \
  --arg fit_complete_sha256 "${relative_complete_sha256}" \
  --arg evaluation_sha256 "$(sha "${evaluation_json}")" \
  --arg evaluation_complete_sha256 "$(sha "${evaluation_complete}")" \
  --arg decision "${decision}" --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson fit_seconds "${fit_seconds}" --argjson evaluation_seconds "${evaluation_seconds}" \
  --argjson total_seconds "$((job_end_epoch - job_start_epoch))" '
  {schema:$schema,job_id:$job_id,code_revision:$code_revision,accelerator:$accelerator,gpu_count:1,cpu_count:12,
   queue_wait_seconds:$queue_wait_seconds,timing_seconds:{fit:$fit_seconds,evaluation:$evaluation_seconds,total:$total_seconds},
   inputs:{protocol_sha256:$protocol_sha256,resource_amendment_sha256:$resource_amendment_sha256},
   artifacts:{predictions_sha256:$predictions_sha256,fit_audit_sha256:$fit_audit_sha256,
   fit_report_sha256:$fit_report_sha256,fit_complete_sha256:$fit_complete_sha256,
   evaluation_sha256:$evaluation_sha256,evaluation_complete_sha256:$evaluation_complete_sha256},
   decision:$decision,credentials_present:false,validation_or_test_inputs_used:false}' > "${execution}.tmp"
mv "${execution}.tmp" "${execution}"
echo "Relative-where end: $(date --iso-8601=seconds)"
printf 'infographicvqa_relative_where_complete=%s decision=%s execution_sha256=%s\n' \
  "${evaluation_complete}" "${decision}" "$(sha "${execution}")"
