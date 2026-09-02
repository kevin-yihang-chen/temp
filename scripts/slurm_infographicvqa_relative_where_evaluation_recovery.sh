#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --job-name=be-infovqa-relwhere-recovery
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-infovqa-relative-where-eval-recovery-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "usage: worker REVISION WORKER_SHA RUNNER_SHA EVAL_MODULE_SHA DECAR_EVAL_SHA RECOVERY_SHA SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_runner_sha256=$3
expected_eval_module_sha256=$4
expected_decar_eval_sha256=$5
expected_recovery_sha256=$6
submit_epoch=$7
for value in "${expected_worker_sha256}" "${expected_runner_sha256}" \
  "${expected_eval_module_sha256}" "${expected_decar_eval_sha256}" \
  "${expected_recovery_sha256}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "relative-where recovery received a malformed SHA-256" >&2
    exit 2
  fi
done
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "relative-where recovery submit epoch must be an integer" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/bin/python
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
fit_dir="${root}/relative-where-oof-v1"
rollouts="${root}/merged-rollouts/rollouts.jsonl"
answer_nll="${root}/merged-nll/answer-nll.jsonl"
relative_predictions="${fit_dir}/predictions.jsonl"
relative_complete="${fit_dir}/complete.json"
relative_audit="${fit_dir}/audit.json"
relative_report="${fit_dir}/report.json"
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
recovery="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-relative-where-evaluation-recovery-v1.md"
failure_log="${repo}/slurm-infovqa-relative-where-203099.out"
worker="${repo}/scripts/slurm_infographicvqa_relative_where_evaluation_recovery.sh"
runner="${repo}/scripts/evaluate_infographicvqa_relative_where_oof.py"
eval_module="${repo}/src/beyond_entropy/infographicvqa_relative_where_evaluation.py"
decar_eval_module="${repo}/src/beyond_entropy/infographicvqa_decar_evaluation.py"
output_dir="${fit_dir}/evaluation-recovery-v1"
execution_dir="${root}/relative-where-execution"

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "relative-where recovery ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "relative-where recovery tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "relative-where recovery tracked worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${runner}" "${expected_runner_sha256}" runner
require_hash "${eval_module}" "${expected_eval_module_sha256}" eval-module
require_hash "${decar_eval_module}" "${expected_decar_eval_sha256}" decar-eval-module
require_hash "${recovery}" "${expected_recovery_sha256}" recovery-protocol
require_hash "${failure_log}" 1277c06f98a14ffbd8cfddb4a833c87a1a2a38de149cf786c87c6621e6f00def failure-log
require_hash "${rollouts}" 9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e rollouts
require_hash "${answer_nll}" 884de4ebc1ba83226393871cdfff819bc7553a61eee6ffa11697db03332ac646 answer-nll
require_hash "${relative_predictions}" 94e292d33fe9e55e9d50d53eef5f2590c4d5c751567feb2fe736867fc7dd6b6b relative-predictions
require_hash "${relative_complete}" 700170914af0e5721479fdd5594696cd872ac4f49ed5fcd5b6bd14649410b677 relative-complete
require_hash "${relative_audit}" 256c34ad9d370107950f9edf915c4a65337bf35df0b198fcb6bccf02d56319af relative-audit
require_hash "${relative_report}" f164e1481e09f3bc9be7450b7fc82fd682e0b1177b3c696db264c366c2d0202a relative-report
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
require_hash "${protocol}" 198fdecc382066e0ac020b3a791364c8e5ef100c488cb6969194db8e30f122a6 protocol
require_hash "${design_audit}" c35f685b5b83af00cf2132cdcf8083e46566e3696bd82ec0a8ad4fe0adc0550b design-audit
require_hash "${oracle_result}" c520dc3fd9a5d25c0b7e626a88a55e629e56c8a6d1a473feb53d94bad4689cf0 oracle-result
if [[ -e "${output_dir}" ]]; then
  echo "relative-where recovery refuses to overwrite output" >&2
  exit 2
fi
if [[ "$(jq -r '.prediction_rows' "${relative_complete}")" -ne 23946 \
  || "$(jq -r '.prediction_outcomes_included' "${relative_complete}")" != false \
  || "$(jq -r '.validation_or_test_inputs_used' "${relative_complete}")" != false ]]; then
  echo "relative-where recovery fit contract failed" >&2
  exit 2
fi

mkdir -p "${execution_dir}"
export PYTHONPATH="${repo}/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

job_start_epoch=$(date +%s)
queue_wait_seconds=$((job_start_epoch - submit_epoch))
if [[ "${queue_wait_seconds}" -lt 0 ]]; then
  echo "relative-where recovery submit epoch is in the future" >&2
  exit 2
fi
echo "Relative-where evaluation recovery start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "CPU allocation: ${SLURM_CPUS_PER_TASK:-unknown}; reserved GPU hidden from evaluator"

"${python_bin}" "${runner}" \
  --rollouts "${rollouts}" --expected-rollouts-sha256 9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e \
  --answer-nll "${answer_nll}" --expected-answer-nll-sha256 884de4ebc1ba83226393871cdfff819bc7553a61eee6ffa11697db03332ac646 \
  --relative-predictions "${relative_predictions}" --expected-relative-predictions-sha256 94e292d33fe9e55e9d50d53eef5f2590c4d5c751567feb2fe736867fc7dd6b6b \
  --relative-complete "${relative_complete}" --expected-relative-complete-sha256 700170914af0e5721479fdd5594696cd872ac4f49ed5fcd5b6bd14649410b677 \
  --relative-audit "${relative_audit}" --expected-relative-audit-sha256 256c34ad9d370107950f9edf915c4a65337bf35df0b198fcb6bccf02d56319af \
  --relative-report "${relative_report}" --expected-relative-report-sha256 f164e1481e09f3bc9be7450b7fc82fd682e0b1177b3c696db264c366c2d0202a \
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
  --protocol "${protocol}" --expected-protocol-sha256 198fdecc382066e0ac020b3a791364c8e5ef100c488cb6969194db8e30f122a6 \
  --design-audit "${design_audit}" --expected-design-audit-sha256 c35f685b5b83af00cf2132cdcf8083e46566e3696bd82ec0a8ad4fe0adc0550b \
  --oracle-result "${oracle_result}" --expected-oracle-result-sha256 c520dc3fd9a5d25c0b7e626a88a55e629e56c8a6d1a473feb53d94bad4689cf0 \
  --output-dir "${output_dir}"

complete="${output_dir}/complete.json"
evaluation="${output_dir}/evaluation.json"
decision=$(jq -r '.decision' "${complete}")
if [[ "${decision}" != relative_where_train_supported \
  && "${decision}" != relative_where_train_not_supported ]]; then
  echo "relative-where recovery decision is invalid" >&2
  exit 2
fi
if [[ "$(jq -r '.validation_or_test_inputs_used' "${complete}")" != false \
  || "$(jq -r '.relative_prediction_outcomes_included' "${complete}")" != false \
  || "$(jq -r '.privileged_teacher_used_only_in_evaluation' "${complete}")" != true \
  || "$(jq -r '.formal_bootstrap_reused' "${complete}")" != true \
  || "$(jq -r '.population.decisions' "${evaluation}")" -ne 23946 \
  || "$(jq -r '.bootstrap.n_resamples' "${evaluation}")" -ne 20000 \
  || "$(jq -r '[.operating_points[].frozen_comparators_exact_match] | all' "${evaluation}")" != true ]]; then
  echo "relative-where recovery output contract failed" >&2
  exit 2
fi

job_end_epoch=$(date +%s)
execution="${execution_dir}/job-${SLURM_JOB_ID}-evaluation-recovery.json"
jq -n \
  --arg schema infographicvqa_relative_where_evaluation_recovery_execution_v1 \
  --arg job_id "${SLURM_JOB_ID}" --arg code_revision "${expected_revision}" \
  --arg decision "${decision}" --arg evaluation_sha256 "$(sha "${evaluation}")" \
  --arg complete_sha256 "$(sha "${complete}")" \
  --arg recovery_sha256 "${expected_recovery_sha256}" \
  --arg failed_job_log_sha256 1277c06f98a14ffbd8cfddb4a833c87a1a2a38de149cf786c87c6621e6f00def \
  --arg predictions_sha256 94e292d33fe9e55e9d50d53eef5f2590c4d5c751567feb2fe736867fc7dd6b6b \
  --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson total_seconds "$((job_end_epoch - job_start_epoch))" '
  {schema:$schema,job_id:$job_id,code_revision:$code_revision,accelerator:"CPU",gpu_reserved:"NVIDIA RTX 4090",gpu_hidden:true,
   gpu_count:1,cpu_count:4,queue_wait_seconds:$queue_wait_seconds,total_seconds:$total_seconds,
   inputs:{recovery_sha256:$recovery_sha256,failed_job_log_sha256:$failed_job_log_sha256,
   predictions_sha256:$predictions_sha256},artifacts:{evaluation_sha256:$evaluation_sha256,complete_sha256:$complete_sha256},
   decision:$decision,credentials_present:false,validation_or_test_inputs_used:false}' > "${execution}.tmp"
mv "${execution}.tmp" "${execution}"
echo "Relative-where evaluation recovery end: $(date --iso-8601=seconds)"
printf 'infographicvqa_relative_where_recovery_complete=%s decision=%s execution_sha256=%s\n' \
  "${complete}" "${decision}" "$(sha "${execution}")"
