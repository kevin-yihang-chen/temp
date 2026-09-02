#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=be-infovqa-relwhere-audit
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-infovqa-relative-where-action-audit-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "usage: worker REVISION WORKER_SHA RUNNER_SHA MODULE_SHA PROTOCOL_SHA RESULT_SHA SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_runner_sha256=$3
expected_module_sha256=$4
expected_protocol_sha256=$5
expected_result_sha256=$6
submit_epoch=$7
for value in "${expected_worker_sha256}" "${expected_runner_sha256}" \
  "${expected_module_sha256}" "${expected_protocol_sha256}" \
  "${expected_result_sha256}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "relative-where action audit received a malformed SHA-256" >&2
    exit 2
  fi
done
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "relative-where action audit submit epoch must be an integer" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/bin/python
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
fit_dir="${root}/relative-where-oof-v1"
predictions="${fit_dir}/predictions.jsonl"
fit_complete="${fit_dir}/complete.json"
answer_nll="${root}/merged-nll/answer-nll.jsonl"
rollouts="${root}/merged-rollouts/rollouts.jsonl"
parent_evaluation="${fit_dir}/evaluation-recovery-v1/evaluation.json"
parent_complete="${fit_dir}/evaluation-recovery-v1/complete.json"
parent_result="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-relative-where-oof-result-job-203237-v1.md"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-relative-where-action-generalization-audit-protocol-v1.md"
worker="${repo}/scripts/slurm_infographicvqa_relative_where_action_generalization_audit.sh"
runner="${repo}/scripts/audit_infographicvqa_relative_where_action_generalization.py"
module="${repo}/src/beyond_entropy/infographicvqa_relative_where_diagnostics.py"
output_dir="${root}/relative-where-action-generalization-audit-v1"
execution_dir="${root}/relative-where-execution"

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "relative-where action audit ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "relative-where action audit tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "relative-where action audit tracked worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${runner}" "${expected_runner_sha256}" runner
require_hash "${module}" "${expected_module_sha256}" module
require_hash "${protocol}" "${expected_protocol_sha256}" protocol
require_hash "${parent_result}" "${expected_result_sha256}" parent-result
require_hash "${predictions}" 94e292d33fe9e55e9d50d53eef5f2590c4d5c751567feb2fe736867fc7dd6b6b predictions
require_hash "${fit_complete}" 700170914af0e5721479fdd5594696cd872ac4f49ed5fcd5b6bd14649410b677 fit-complete
require_hash "${answer_nll}" 884de4ebc1ba83226393871cdfff819bc7553a61eee6ffa11697db03332ac646 answer-nll
require_hash "${rollouts}" 9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e rollouts
require_hash "${parent_evaluation}" 1c51131d6b8599a3733c3018e0a53570552ff09fff19aa07bcb7bf61b984e61c parent-evaluation
require_hash "${parent_complete}" e7f1557d7a6b14ef6888b57c873b3574a15b01cdffc175b12893f7153a903afd parent-complete
if [[ -e "${output_dir}" ]]; then
  echo "relative-where action audit refuses to overwrite output" >&2
  exit 2
fi
if [[ "$(jq -r '.decision' "${parent_complete}")" != relative_where_train_not_supported \
  || "$(jq -r '.validation_or_test_inputs_used' "${parent_complete}")" != false \
  || "$(jq -r '.prediction_outcomes_included' "${fit_complete}")" != false ]]; then
  echo "relative-where action audit parent contract failed" >&2
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
  echo "relative-where action audit submit epoch is in the future" >&2
  exit 2
fi
echo "Relative-where action audit start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "CPU allocation: ${SLURM_CPUS_PER_TASK:-unknown}; reserved GPU hidden from diagnostic"

"${python_bin}" "${runner}" \
  --predictions "${predictions}" --expected-predictions-sha256 94e292d33fe9e55e9d50d53eef5f2590c4d5c751567feb2fe736867fc7dd6b6b \
  --answer-nll "${answer_nll}" --expected-answer-nll-sha256 884de4ebc1ba83226393871cdfff819bc7553a61eee6ffa11697db03332ac646 \
  --rollouts "${rollouts}" --expected-rollouts-sha256 9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e \
  --parent-result "${parent_result}" --expected-parent-result-sha256 "${expected_result_sha256}" \
  --protocol "${protocol}" --expected-protocol-sha256 "${expected_protocol_sha256}" \
  --output-dir "${output_dir}"

complete="${output_dir}/complete.json"
audit="${output_dir}/audit.json"
if [[ "$(jq -r '.decisions' "${complete}")" -ne 23946 \
  || "$(jq -r '.sources' "${complete}")" -ne 2204 \
  || "$(jq -r '.prediction_outcomes_included' "${complete}")" != false \
  || "$(jq -r '.validation_or_test_inputs_used' "${complete}")" != false \
  || "$(jq -r '.changes_parent_train_gate' "${complete}")" != false \
  || "$(jq -r '.variants | length' "${audit}")" -ne 4 \
  || "$(jq -r '.variants.relative_teacher_entropy.by_outer_fold | length' "${audit}")" -ne 5 \
  || "$(jq -r '.variants.relative_teacher_entropy.by_confidence_decile | length' "${audit}")" -ne 10 ]]; then
  echo "relative-where action audit output contract failed" >&2
  exit 2
fi

job_end_epoch=$(date +%s)
execution="${execution_dir}/job-${SLURM_JOB_ID}-action-generalization-audit.json"
jq -n \
  --arg schema infographicvqa_relative_where_action_generalization_execution_v1 \
  --arg job_id "${SLURM_JOB_ID}" --arg code_revision "${expected_revision}" \
  --arg protocol_sha256 "${expected_protocol_sha256}" \
  --arg parent_result_sha256 "${expected_result_sha256}" \
  --arg audit_sha256 "$(sha "${audit}")" --arg complete_sha256 "$(sha "${complete}")" \
  --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson total_seconds "$((job_end_epoch - job_start_epoch))" '
  {schema:$schema,job_id:$job_id,code_revision:$code_revision,accelerator:"CPU",gpu_reserved:"NVIDIA RTX 4090",gpu_hidden:true,
   gpu_count:1,cpu_count:4,queue_wait_seconds:$queue_wait_seconds,total_seconds:$total_seconds,
   inputs:{protocol_sha256:$protocol_sha256,parent_result_sha256:$parent_result_sha256},
   artifacts:{audit_sha256:$audit_sha256,complete_sha256:$complete_sha256},
   credentials_present:false,validation_or_test_inputs_used:false,changes_parent_train_gate:false}' > "${execution}.tmp"
mv "${execution}.tmp" "${execution}"
echo "Relative-where action audit end: $(date --iso-8601=seconds)"
printf 'infographicvqa_relative_where_action_audit_complete=%s execution_sha256=%s\n' \
  "${complete}" "$(sha "${execution}")"
