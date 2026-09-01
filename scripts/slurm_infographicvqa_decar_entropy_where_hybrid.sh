#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --job-name=be-infovqa-hybrid
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-infovqa-decar-hybrid-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: worker REVISION WORKER_SHA RUNNER_SHA MODULE_SHA FREEZE_SHA SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_runner_sha256=$3
expected_module_sha256=$4
expected_freeze_sha256=$5
submit_epoch=$6
for value in "${expected_worker_sha256}" "${expected_runner_sha256}" \
  "${expected_module_sha256}" "${expected_freeze_sha256}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "DECAR hybrid received a malformed SHA-256" >&2
    exit 2
  fi
done
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "DECAR hybrid submit epoch must be an integer" >&2
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
formal_result="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-oof-result-job-203049-v1.md"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-method-protocol-v1.md"
freeze="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-entropy-where-hybrid-freeze-v1.md"
worker="${repo}/scripts/slurm_infographicvqa_decar_entropy_where_hybrid.sh"
runner="${repo}/scripts/evaluate_infographicvqa_decar_entropy_where_hybrid.py"
module="${repo}/src/beyond_entropy/infographicvqa_decar_evaluation.py"
output_dir="${root}/entropy-where-hybrid-v1"
execution_dir="${root}/hybrid-execution"

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "DECAR hybrid ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "DECAR hybrid tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "DECAR hybrid tracked worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${runner}" "${expected_runner_sha256}" runner
require_hash "${module}" "${expected_module_sha256}" module
require_hash "${freeze}" "${expected_freeze_sha256}" freeze
require_hash "${rollouts}" 9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e rollouts
require_hash "${predictions}" c8338b1960ca223c892c3f992b0ccc3027d543113f67c82a734eccd7e3699c4b predictions
require_hash "${oof_complete}" 8de073870fcade5ac111d59de81e9c70dc567c9900e0b223cdebca6a8318f31f OOF-complete
require_hash "${oof_audit}" dc3193dfc626a3df50321f4d92a336ce784aee34f1f2e91c57cf87d1f8085537 OOF-audit
require_hash "${oof_report}" ebc936739e970fbfda25ebe02ef71d6b7f46674f9d00092011b5665c2daa9bf0 OOF-report
require_hash "${formal_evaluation}" ee5f9972e1d897c7fb833208a5722ee3a0313a05f0217f921966b3e0e1978df9 formal-evaluation
require_hash "${formal_complete}" d0443614c286349b7e360d646fef960816aba47a614bc20960c119d5e0ddeb79 formal-complete
require_hash "${bootstrap_indices}" 17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6 bootstrap-indices
require_hash "${bootstrap_sources}" 5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0 bootstrap-sources
require_hash "${formal_result}" bdf2ee531c76743fccdffde0873380640f0cf8cdd16ee31ef71d4d23e386143a formal-result
require_hash "${protocol}" d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342 protocol
if [[ -e "${output_dir}" ]]; then
  echo "DECAR hybrid refuses to overwrite output" >&2
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
  echo "DECAR hybrid submit epoch is in the future" >&2
  exit 2
fi
echo "DECAR hybrid start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "CPU allocation: ${SLURM_CPUS_PER_TASK:-unknown}; GPU allocation: none"

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
  --formal-result "${formal_result}" --expected-formal-result-sha256 bdf2ee531c76743fccdffde0873380640f0cf8cdd16ee31ef71d4d23e386143a \
  --protocol "${protocol}" --expected-protocol-sha256 d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342 \
  --freeze "${freeze}" --expected-freeze-sha256 "${expected_freeze_sha256}" \
  --output-dir "${output_dir}"

complete="${output_dir}/complete.json"
evaluation="${output_dir}/evaluation.json"
decision=$(jq -r '.decision' "${complete}")
if [[ "${decision}" != hybrid_train_supported \
  && "${decision}" != hybrid_train_not_supported ]]; then
  echo "DECAR hybrid decision is invalid" >&2
  exit 2
fi
if [[ "$(jq -r '.validation_or_test_inputs_used' "${complete}")" != false \
  || "$(jq -r '.formal_bootstrap_reused' "${complete}")" != true \
  || "$(jq -r '.population.decisions' "${evaluation}")" -ne 23946 \
  || "$(jq -r '.bootstrap.n_resamples' "${evaluation}")" -ne 20000 ]]; then
  echo "DECAR hybrid output contract failed" >&2
  exit 2
fi

job_end_epoch=$(date +%s)
execution="${execution_dir}/job-${SLURM_JOB_ID}.json"
jq -n \
  --arg schema infographicvqa_decar_entropy_where_hybrid_execution_v1 \
  --arg job_id "${SLURM_JOB_ID}" --arg code_revision "${expected_revision}" \
  --arg decision "${decision}" --arg evaluation_sha256 "$(sha "${evaluation}")" \
  --arg complete_sha256 "$(sha "${complete}")" \
  --arg freeze_sha256 "${expected_freeze_sha256}" \
  --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson total_seconds "$((job_end_epoch - job_start_epoch))" '
  {schema:$schema,job_id:$job_id,code_revision:$code_revision,accelerator:"CPU",gpu_count:0,
   cpu_count:4,queue_wait_seconds:$queue_wait_seconds,total_seconds:$total_seconds,
   inputs:{freeze_sha256:$freeze_sha256},
   artifacts:{evaluation_sha256:$evaluation_sha256,complete_sha256:$complete_sha256},
   decision:$decision,credentials_present:false,validation_or_test_inputs_used:false}' > "${execution}.tmp"
mv "${execution}.tmp" "${execution}"
echo "DECAR hybrid end: $(date --iso-8601=seconds)"
printf 'infographicvqa_decar_hybrid_complete=%s decision=%s execution_sha256=%s\n' \
  "${complete}" "${decision}" "$(sha "${execution}")"
