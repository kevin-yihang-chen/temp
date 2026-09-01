#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --job-name=be-infovqa-oracle-where
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-infovqa-oracle-where-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "usage: worker REVISION WORKER_SHA RUNNER_SHA MODULE_SHA FREEZE_SHA RESOURCE_SHA SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_runner_sha256=$3
expected_module_sha256=$4
expected_freeze_sha256=$5
resource_amendment_sha256=$6
submit_epoch=$7
for value in "${expected_worker_sha256}" "${expected_runner_sha256}" \
  "${expected_module_sha256}" "${expected_freeze_sha256}" \
  "${resource_amendment_sha256}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "oracle-where received a malformed SHA-256" >&2
    exit 2
  fi
done
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "oracle-where submit epoch must be an integer" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/bin/python
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
rollouts="${root}/merged-rollouts/rollouts.jsonl"
predictions="${root}/nested-oof-v1/predictions.jsonl"
oof_complete="${root}/nested-oof-v1/complete.json"
oof_audit="${root}/nested-oof-v1/audit.json"
oof_report="${root}/nested-oof-v1/report.json"
formal_evaluation="${root}/evaluation-v1/evaluation.json"
formal_complete="${root}/evaluation-v1/complete.json"
bootstrap_indices="${root}/evaluation-v1/bootstrap-indices.npy"
bootstrap_sources="${root}/evaluation-v1/bootstrap-sources.json"
hybrid_evaluation="${root}/entropy-where-hybrid-v1/evaluation.json"
hybrid_decision="${root}/entropy-where-hybrid-v1/decision.json"
hybrid_complete="${root}/entropy-where-hybrid-v1/complete.json"
hybrid_result="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-entropy-where-hybrid-result-job-203059-v1.md"
hybrid_freeze="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-entropy-where-hybrid-freeze-v1.md"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-method-protocol-v1.md"
freeze="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-entropy-when-oracle-where-factorization-freeze-v1.md"
resource_amendment="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-entropy-oracle-where-resource-amendment-v1.md"
worker="${repo}/scripts/slurm_infographicvqa_entropy_oracle_where_factorization.sh"
runner="${repo}/scripts/evaluate_infographicvqa_entropy_oracle_where_factorization.py"
module="${repo}/src/beyond_entropy/infographicvqa_decar_evaluation.py"
output_dir="${root}/entropy-oracle-where-factorization-v1"
execution_dir="${root}/oracle-where-execution"

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "oracle-where ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "oracle-where tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "oracle-where tracked worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${runner}" "${expected_runner_sha256}" runner
require_hash "${module}" "${expected_module_sha256}" module
require_hash "${freeze}" "${expected_freeze_sha256}" freeze
require_hash "${resource_amendment}" "${resource_amendment_sha256}" resource-amendment
require_hash "${rollouts}" 9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e rollouts
require_hash "${predictions}" c8338b1960ca223c892c3f992b0ccc3027d543113f67c82a734eccd7e3699c4b predictions
require_hash "${oof_complete}" 8de073870fcade5ac111d59de81e9c70dc567c9900e0b223cdebca6a8318f31f OOF-complete
require_hash "${oof_audit}" dc3193dfc626a3df50321f4d92a336ce784aee34f1f2e91c57cf87d1f8085537 OOF-audit
require_hash "${oof_report}" ebc936739e970fbfda25ebe02ef71d6b7f46674f9d00092011b5665c2daa9bf0 OOF-report
require_hash "${formal_evaluation}" ee5f9972e1d897c7fb833208a5722ee3a0313a05f0217f921966b3e0e1978df9 formal-evaluation
require_hash "${formal_complete}" d0443614c286349b7e360d646fef960816aba47a614bc20960c119d5e0ddeb79 formal-complete
require_hash "${bootstrap_indices}" 17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6 bootstrap-indices
require_hash "${bootstrap_sources}" 5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0 bootstrap-sources
require_hash "${hybrid_evaluation}" ab4a179c2141de60c9a1d173e34a7166d075935e5283569439e3d93424344a62 hybrid-evaluation
require_hash "${hybrid_decision}" 0597526725eac7efed05392fb652b04798de26b71d6de6d063303b49ec114d42 hybrid-decision
require_hash "${hybrid_complete}" 4e3b3c2b2b2e1698fcf9bb3c9e71881b11e1357982bb1b6634a12af6a7aa03ac hybrid-complete
require_hash "${hybrid_result}" 16ed848ee49702d1f1c41e9f59b2245585dc03a918e3ffacaf3520fc2fafefab hybrid-result
require_hash "${hybrid_freeze}" 86e61bb0c7a4ad5a259077314be3a83c6c95284d2b219c414016b2280292a8bb hybrid-freeze
require_hash "${protocol}" d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342 protocol
if [[ -e "${output_dir}" ]]; then
  echo "oracle-where refuses to overwrite output" >&2
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
  echo "oracle-where submit epoch is in the future" >&2
  exit 2
fi
echo "Oracle-where start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "CPU allocation: ${SLURM_CPUS_PER_TASK:-unknown}; GPU reserved for QOS admission and hidden from evaluator"

"${python_bin}" "${runner}" \
  --rollouts "${rollouts}" --expected-rollouts-sha256 9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e \
  --predictions "${predictions}" --expected-predictions-sha256 c8338b1960ca223c892c3f992b0ccc3027d543113f67c82a734eccd7e3699c4b \
  --oof-complete "${oof_complete}" --expected-oof-complete-sha256 8de073870fcade5ac111d59de81e9c70dc567c9900e0b223cdebca6a8318f31f \
  --oof-audit "${oof_audit}" --expected-oof-audit-sha256 dc3193dfc626a3df50321f4d92a336ce784aee34f1f2e91c57cf87d1f8085537 \
  --oof-report "${oof_report}" --expected-oof-report-sha256 ebc936739e970fbfda25ebe02ef71d6b7f46674f9d00092011b5665c2daa9bf0 \
  --formal-evaluation "${formal_evaluation}" --expected-formal-evaluation-sha256 ee5f9972e1d897c7fb833208a5722ee3a0313a05f0217f921966b3e0e1978df9 \
  --formal-complete "${formal_complete}" --expected-formal-complete-sha256 d0443614c286349b7e360d646fef960816aba47a614bc20960c119d5e0ddeb79 \
  --bootstrap-indices "${bootstrap_indices}" --expected-bootstrap-indices-sha256 17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6 \
  --bootstrap-sources "${bootstrap_sources}" --expected-bootstrap-sources-sha256 5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0 \
  --hybrid-evaluation "${hybrid_evaluation}" --expected-hybrid-evaluation-sha256 ab4a179c2141de60c9a1d173e34a7166d075935e5283569439e3d93424344a62 \
  --hybrid-decision "${hybrid_decision}" --expected-hybrid-decision-sha256 0597526725eac7efed05392fb652b04798de26b71d6de6d063303b49ec114d42 \
  --hybrid-complete "${hybrid_complete}" --expected-hybrid-complete-sha256 4e3b3c2b2b2e1698fcf9bb3c9e71881b11e1357982bb1b6634a12af6a7aa03ac \
  --hybrid-result "${hybrid_result}" --expected-hybrid-result-sha256 16ed848ee49702d1f1c41e9f59b2245585dc03a918e3ffacaf3520fc2fafefab \
  --hybrid-freeze "${hybrid_freeze}" --expected-hybrid-freeze-sha256 86e61bb0c7a4ad5a259077314be3a83c6c95284d2b219c414016b2280292a8bb \
  --protocol "${protocol}" --expected-protocol-sha256 d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342 \
  --factorization-freeze "${freeze}" --expected-factorization-freeze-sha256 "${expected_freeze_sha256}" \
  --output-dir "${output_dir}"

complete="${output_dir}/complete.json"
evaluation="${output_dir}/evaluation.json"
decision=$(jq -r '.decision' "${complete}")
if [[ "${decision}" != where_bottleneck_supported \
  && "${decision}" != where_bottleneck_not_supported ]]; then
  echo "oracle-where decision is invalid" >&2
  exit 2
fi
if [[ "$(jq -r '.validation_or_test_inputs_used' "${complete}")" != false \
  || "$(jq -r '.outcome_oracle_used' "${complete}")" != true \
  || "$(jq -r '.deployable_method_evidence' "${complete}")" != false \
  || "$(jq -r '.formal_bootstrap_reused' "${complete}")" != true \
  || "$(jq -r '.population.decisions' "${evaluation}")" -ne 23946 \
  || "$(jq -r '.bootstrap.n_resamples' "${evaluation}")" -ne 20000 ]]; then
  echo "oracle-where output contract failed" >&2
  exit 2
fi

job_end_epoch=$(date +%s)
execution="${execution_dir}/job-${SLURM_JOB_ID}.json"
jq -n \
  --arg schema infographicvqa_entropy_oracle_where_execution_v1 \
  --arg job_id "${SLURM_JOB_ID}" --arg code_revision "${expected_revision}" \
  --arg decision "${decision}" --arg evaluation_sha256 "$(sha "${evaluation}")" \
  --arg complete_sha256 "$(sha "${complete}")" \
  --arg freeze_sha256 "${expected_freeze_sha256}" \
  --arg resource_amendment_sha256 "${resource_amendment_sha256}" \
  --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson total_seconds "$((job_end_epoch - job_start_epoch))" '
  {schema:$schema,job_id:$job_id,code_revision:$code_revision,accelerator:"CPU",gpu_reserved:"NVIDIA RTX 4090",gpu_count:1,
   cpu_count:4,queue_wait_seconds:$queue_wait_seconds,total_seconds:$total_seconds,
   inputs:{freeze_sha256:$freeze_sha256,resource_amendment_sha256:$resource_amendment_sha256},
   artifacts:{evaluation_sha256:$evaluation_sha256,complete_sha256:$complete_sha256},
   decision:$decision,outcome_oracle_used:true,deployable_method_evidence:false,
   credentials_present:false,validation_or_test_inputs_used:false}' > "${execution}.tmp"
mv "${execution}.tmp" "${execution}"
echo "Oracle-where end: $(date --iso-8601=seconds)"
printf 'infographicvqa_oracle_where_complete=%s decision=%s execution_sha256=%s\n' \
  "${complete}" "${decision}" "$(sha "${execution}")"
