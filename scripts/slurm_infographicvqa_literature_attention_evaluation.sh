#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=be-infovqa-lit-eval
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-infovqa-lit-attn-eval-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "usage: worker REVISION WORKER_SHA RUNNER_SHA EVAL_SHA PROTOCOL_SHA FEATURE_JOB_ID FEATURE_CODE_REVISION SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_runner_sha256=$3
expected_eval_sha256=$4
expected_protocol_sha256=$5
feature_job_id=$6
feature_code_revision=$7
submit_epoch=$8
for value in "${expected_worker_sha256}" "${expected_runner_sha256}" \
  "${expected_eval_sha256}" "${expected_protocol_sha256}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "literature-attention evaluation received a malformed SHA-256" >&2
    exit 2
  fi
done
if [[ ! "${feature_job_id}" =~ ^[0-9]+$ || ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "literature-attention evaluation job ID or submit epoch is invalid" >&2
  exit 2
fi
if [[ ! "${feature_code_revision}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "literature-attention evaluation feature code revision is invalid" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/llava-med/bin/python
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
literature_root="${root}/literature-attention-where-v1"
raw_root="${root}/attention-where-v1"
literature_features="${literature_root}/merged-features/features-literature-attention-label-free.pt"
literature_complete="${literature_root}/complete.json"
literature_audit="${literature_root}/merged-features/literature-attention-feature-audit.json"
literature_merge="${literature_root}/merged-features/merge-report.json"
literature_execution="${literature_root}/execution/job-${feature_job_id}.json"
raw_features="${raw_root}/merged-features/features-question-region-attention-label-free.pt"
raw_complete="${raw_root}/complete.json"
raw_audit="${raw_root}/merged-features/attention-feature-audit.json"
raw_evaluation="${raw_root}/evaluation-v1/evaluation.json"
raw_evaluation_complete="${raw_root}/evaluation-v1/complete.json"
rollouts="${root}/merged-rollouts/rollouts.jsonl"
answer_nll="${root}/merged-nll/answer-nll.jsonl"
decar_predictions="${root}/nested-oof-v1/predictions.jsonl"
relative_predictions="${root}/relative-where-oof-v1/predictions.jsonl"
hybrid_evaluation="${root}/entropy-where-hybrid-v1/evaluation.json"
hybrid_complete="${root}/entropy-where-hybrid-v1/complete.json"
oracle_evaluation="${root}/entropy-oracle-where-factorization-v1/evaluation.json"
oracle_complete="${root}/entropy-oracle-where-factorization-v1/complete.json"
relative_evaluation="${root}/relative-where-oof-v1/evaluation-recovery-v1/evaluation.json"
relative_complete="${root}/relative-where-oof-v1/evaluation-recovery-v1/complete.json"
bootstrap_indices="${root}/evaluation-v1/bootstrap-indices.npy"
bootstrap_sources="${root}/evaluation-v1/bootstrap-sources.json"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-literature-attention-where-protocol-20260902-pending.md"
worker="${repo}/scripts/slurm_infographicvqa_literature_attention_evaluation.sh"
runner="${repo}/scripts/evaluate_infographicvqa_literature_attention_where.py"
eval_module="${repo}/src/beyond_entropy/infographicvqa_literature_attention_evaluation.py"
output_dir="${literature_root}/evaluation-v1"
execution_dir="${literature_root}/evaluation-execution"
raw_code_revision=2020b423f7daa6e8b9a942a02308137136bba548
model_revision=cc594898137f460bfe9f0759e9844b3ce807cfb5
source_features_sha256=d0508726a50b4c1e54778392d08329b242a680fc13292cac1ebec8b42a175300

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "literature-attention evaluation ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "literature-attention evaluation tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "literature-attention evaluation tracked worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${runner}" "${expected_runner_sha256}" runner
require_hash "${eval_module}" "${expected_eval_sha256}" eval-module
require_hash "${protocol}" "${expected_protocol_sha256}" protocol
require_hash "${rollouts}" 9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e rollouts
require_hash "${answer_nll}" 884de4ebc1ba83226393871cdfff819bc7553a61eee6ffa11697db03332ac646 answer-nll
require_hash "${decar_predictions}" c8338b1960ca223c892c3f992b0ccc3027d543113f67c82a734eccd7e3699c4b decar-predictions
require_hash "${relative_predictions}" 94e292d33fe9e55e9d50d53eef5f2590c4d5c751567feb2fe736867fc7dd6b6b relative-predictions
require_hash "${hybrid_evaluation}" ab4a179c2141de60c9a1d173e34a7166d075935e5283569439e3d93424344a62 hybrid-evaluation
require_hash "${hybrid_complete}" 4e3b3c2b2b2e1698fcf9bb3c9e71881b11e1357982bb1b6634a12af6a7aa03ac hybrid-complete
require_hash "${oracle_evaluation}" 6ef0869b453e1a70ad5f479e8a9604aa04ac0419a1dc92d1ce353c54f66f3025 oracle-evaluation
require_hash "${oracle_complete}" b940258389e558d0d0bae277bd8d5b081923ce505ca2e6ad8520a79bd6411de7 oracle-complete
require_hash "${relative_evaluation}" 1c51131d6b8599a3733c3018e0a53570552ff09fff19aa07bcb7bf61b984e61c relative-evaluation
require_hash "${relative_complete}" e7f1557d7a6b14ef6888b57c873b3574a15b01cdffc175b12893f7153a903afd relative-complete
require_hash "${bootstrap_indices}" 17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6 bootstrap-indices
require_hash "${bootstrap_sources}" 5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0 bootstrap-sources
if [[ -e "${output_dir}" ]]; then
  echo "literature-attention evaluation refuses to overwrite output" >&2
  exit 2
fi
for path in "${literature_features}" "${literature_complete}" "${literature_audit}" \
  "${literature_merge}" "${literature_execution}" "${raw_features}" \
  "${raw_complete}" "${raw_audit}" "${raw_evaluation}" \
  "${raw_evaluation_complete}"; do
  if [[ ! -f "${path}" ]]; then
    echo "literature-attention evaluation input is incomplete: ${path}" >&2
    exit 2
  fi
done
literature_features_sha=$(sha "${literature_features}")
literature_complete_sha=$(sha "${literature_complete}")
literature_audit_sha=$(sha "${literature_audit}")
literature_merge_sha=$(sha "${literature_merge}")
if ! jq -e \
  --arg features_sha "${literature_features_sha}" \
  --arg audit_sha "${literature_audit_sha}" \
  --arg execution_sha "$(sha "${literature_execution}")" '
    .schema == "infographicvqa_literature_attention_where_feature_complete_v1" and
    .passed == true and .decisions == 23946 and .sources == 2204 and .images == 4406 and
    .validation_or_test_inputs_used == false and .outcomes_included == false and
    .merged_features_sha256 == $features_sha and .audit_sha256 == $audit_sha and
    .execution_sha256 == $execution_sha
  ' "${literature_complete}" >/dev/null; then
  echo "literature-attention feature completion binding failed" >&2
  exit 2
fi
if ! jq -e --arg revision "${feature_code_revision}" --arg job_id "${feature_job_id}" '
    .schema == "infographicvqa_literature_attention_where_feature_execution_v1" and
    .job_id == $job_id and .slurm_state == "COMPLETED" and .exit_code == "0:0" and
    .code_revision == $revision and .validation_or_test_inputs_used == false and
    .outcomes_included == false
  ' "${literature_execution}" >/dev/null; then
  echo "literature-attention feature execution binding failed" >&2
  exit 2
fi

mkdir -p "${execution_dir}"
export PYTHONPATH="${repo}/src:${repo}/scripts"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HTTP_PROXY HTTPS_PROXY ALL_PROXY \
  http_proxy https_proxy all_proxy
job_start_epoch=$(date +%s)
queue_wait_seconds=$((job_start_epoch - submit_epoch))
if [[ "${queue_wait_seconds}" -lt 0 ]]; then
  echo "literature-attention evaluation submit epoch is in the future" >&2
  exit 2
fi
echo "literature-attention evaluation start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}; reserved GPU hidden from evaluator"

"${python_bin}" "${runner}" \
  --literature-features "${literature_features}" --expected-literature-features-sha256 "${literature_features_sha}" \
  --literature-complete "${literature_complete}" --expected-literature-complete-sha256 "${literature_complete_sha}" \
  --literature-audit "${literature_audit}" --expected-literature-audit-sha256 "${literature_audit_sha}" \
  --literature-merge-report "${literature_merge}" --expected-literature-merge-report-sha256 "${literature_merge_sha}" \
  --raw-attention-features "${raw_features}" --expected-raw-attention-features-sha256 "$(sha "${raw_features}")" \
  --raw-attention-complete "${raw_complete}" --expected-raw-attention-complete-sha256 "$(sha "${raw_complete}")" \
  --raw-attention-audit "${raw_audit}" --expected-raw-attention-audit-sha256 "$(sha "${raw_audit}")" \
  --raw-attention-evaluation "${raw_evaluation}" --expected-raw-attention-evaluation-sha256 "$(sha "${raw_evaluation}")" \
  --raw-attention-evaluation-complete "${raw_evaluation_complete}" --expected-raw-attention-evaluation-complete-sha256 "$(sha "${raw_evaluation_complete}")" \
  --rollouts "${rollouts}" --expected-rollouts-sha256 9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e \
  --answer-nll "${answer_nll}" --expected-answer-nll-sha256 884de4ebc1ba83226393871cdfff819bc7553a61eee6ffa11697db03332ac646 \
  --decar-predictions "${decar_predictions}" --expected-decar-predictions-sha256 c8338b1960ca223c892c3f992b0ccc3027d543113f67c82a734eccd7e3699c4b \
  --relative-predictions "${relative_predictions}" --expected-relative-predictions-sha256 94e292d33fe9e55e9d50d53eef5f2590c4d5c751567feb2fe736867fc7dd6b6b \
  --hybrid-evaluation "${hybrid_evaluation}" --expected-hybrid-evaluation-sha256 ab4a179c2141de60c9a1d173e34a7166d075935e5283569439e3d93424344a62 \
  --hybrid-complete "${hybrid_complete}" --expected-hybrid-complete-sha256 4e3b3c2b2b2e1698fcf9bb3c9e71881b11e1357982bb1b6634a12af6a7aa03ac \
  --oracle-evaluation "${oracle_evaluation}" --expected-oracle-evaluation-sha256 6ef0869b453e1a70ad5f479e8a9604aa04ac0419a1dc92d1ce353c54f66f3025 \
  --oracle-complete "${oracle_complete}" --expected-oracle-complete-sha256 b940258389e558d0d0bae277bd8d5b081923ce505ca2e6ad8520a79bd6411de7 \
  --relative-evaluation "${relative_evaluation}" --expected-relative-evaluation-sha256 1c51131d6b8599a3733c3018e0a53570552ff09fff19aa07bcb7bf61b984e61c \
  --relative-complete "${relative_complete}" --expected-relative-complete-sha256 e7f1557d7a6b14ef6888b57c873b3574a15b01cdffc175b12893f7153a903afd \
  --bootstrap-indices "${bootstrap_indices}" --expected-bootstrap-indices-sha256 17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6 \
  --bootstrap-sources "${bootstrap_sources}" --expected-bootstrap-sources-sha256 5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0 \
  --protocol "${protocol}" --expected-protocol-sha256 "${expected_protocol_sha256}" \
  --expected-literature-code-revision "${feature_code_revision}" \
  --expected-raw-attention-code-revision "${raw_code_revision}" \
  --expected-model-revision "${model_revision}" \
  --expected-source-features-sha256 "${source_features_sha256}" \
  --output-dir "${output_dir}"

complete="${output_dir}/complete.json"
evaluation="${output_dir}/evaluation.json"
decision=$(jq -r '.decision' "${complete}")
if [[ "${decision}" != literature_attention_where_train_supported \
  && "${decision}" != literature_attention_where_train_not_supported ]]; then
  echo "literature-attention evaluation decision is invalid" >&2
  exit 2
fi
if [[ "$(jq -r '.validation_or_test_inputs_used' "${complete}")" != false \
  || "$(jq -r '.multiplicity_corrected' "${complete}")" != true \
  || "$(jq -r '.population.decisions' "${evaluation}")" -ne 23946 \
  || "$(jq -r '.bootstrap.n_resamples' "${evaluation}")" -ne 20000 \
  || "$(jq -r '[.operating_points[].frozen_comparators_exact_match] | all' "${evaluation}")" != true ]]; then
  echo "literature-attention evaluation output contract failed" >&2
  exit 2
fi
job_end_epoch=$(date +%s)
execution="${execution_dir}/job-${SLURM_JOB_ID}.json"
jq -n \
  --arg schema infographicvqa_literature_attention_where_evaluation_execution_v1 \
  --arg job_id "${SLURM_JOB_ID}" --arg code_revision "${expected_revision}" \
  --arg feature_job_id "${feature_job_id}" --arg feature_code_revision "${feature_code_revision}" \
  --arg feature_complete_sha256 "${literature_complete_sha}" \
  --arg decision "${decision}" --arg evaluation_sha256 "$(sha "${evaluation}")" \
  --arg complete_sha256 "$(sha "${complete}")" \
  --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson total_seconds "$((job_end_epoch - job_start_epoch))" '
  {schema:$schema,job_id:$job_id,code_revision:$code_revision,accelerator:"CPU",
   gpu_reserved:"NVIDIA RTX 4090",gpu_hidden:true,gpu_count:1,cpu_count:4,
   queue_wait_seconds:$queue_wait_seconds,total_seconds:$total_seconds,
   feature_job_id:$feature_job_id,feature_code_revision:$feature_code_revision,
   feature_complete_sha256:$feature_complete_sha256,
   artifacts:{evaluation_sha256:$evaluation_sha256,complete_sha256:$complete_sha256},
   decision:$decision,credentials_present:false,validation_or_test_inputs_used:false}' \
  > "${execution}.tmp"
mv "${execution}.tmp" "${execution}"
echo "literature-attention evaluation end: $(date --iso-8601=seconds)"
printf 'literature_attention_evaluation_complete=%s decision=%s execution_sha256=%s\n' \
  "${complete}" "${decision}" "$(sha "${execution}")"
