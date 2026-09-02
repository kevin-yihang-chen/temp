#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --job-name=be-infovqa-stop-diag
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-infovqa-stop-diag-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: worker REVISION WORKER_SHA RUNNER_SHA MODULE_SHA PROTOCOL_SHA SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_runner_sha256=$3
expected_module_sha256=$4
expected_protocol_sha256=$5
submit_epoch=$6
for value in "${expected_worker_sha256}" "${expected_runner_sha256}" \
  "${expected_module_sha256}" "${expected_protocol_sha256}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "attention-stop diagnostic received a malformed SHA-256" >&2
    exit 2
  fi
done
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "attention-stop diagnostic submit epoch must be an integer" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/llava-med/bin/python
root="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"
attention_root="${root}/attention-where-v1"
rollouts="${root}/merged-rollouts/rollouts.jsonl"
features="${attention_root}/merged-features/features-question-region-attention-label-free.pt"
feature_complete="${attention_root}/complete.json"
feature_audit="${attention_root}/merged-features/attention-feature-audit.json"
evaluation="${attention_root}/evaluation-v1/evaluation.json"
evaluation_complete="${attention_root}/evaluation-v1/complete.json"
bootstrap_indices="${root}/evaluation-v1/bootstrap-indices.npy"
bootstrap_sources="${root}/evaluation-v1/bootstrap-sources.json"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-attention-stop-factorization-diagnostic-protocol-v1.md"
worker="${repo}/scripts/slurm_infographicvqa_attention_stop_factorization.sh"
runner="${repo}/scripts/diagnose_infographicvqa_attention_stop_factorization.py"
module="${repo}/src/beyond_entropy/infographicvqa_attention_stop_diagnostic.py"
output_dir="${attention_root}/attention-stop-factorization-v1"
execution_dir="${attention_root}/attention-stop-factorization-execution"
feature_code_revision=2020b423f7daa6e8b9a942a02308137136bba548
model_revision=cc594898137f460bfe9f0759e9844b3ce807cfb5
source_features_sha256=d0508726a50b4c1e54778392d08329b242a680fc13292cac1ebec8b42a175300

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "attention-stop diagnostic ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "attention-stop diagnostic tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "attention-stop diagnostic tracked worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${runner}" "${expected_runner_sha256}" runner
require_hash "${module}" "${expected_module_sha256}" module
require_hash "${protocol}" "${expected_protocol_sha256}" protocol
require_hash "${rollouts}" 9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e rollouts
require_hash "${features}" 009cdbfa1761f9b53b791a50f70b7e98bdf275eec8743d8bcaf078a52ded8ce8 features
require_hash "${feature_complete}" 6eb313cf1bf4e5f61a8decc0c6ef70605009826c1bf4f815deb7f8111ec7bf40 feature-complete
require_hash "${feature_audit}" 27ba5df9d45f9837f685d64589e32740238de6ff0ce46ce54ce6a1ac21a1d471 feature-audit
require_hash "${evaluation}" 5c8bced0fdad0a4f7c3ad0dca8bf8cf31d40be4c9d2318c6b42ea72d065366ee evaluation
require_hash "${evaluation_complete}" ea38fb7adb024a1c96a6ec160d921687affb3ac0222aecba3f5d422728a4cbf5 evaluation-complete
require_hash "${bootstrap_indices}" 17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6 bootstrap-indices
require_hash "${bootstrap_sources}" 5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0 bootstrap-sources
if [[ -e "${output_dir}" ]]; then
  echo "attention-stop diagnostic refuses to overwrite output" >&2
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
  echo "attention-stop diagnostic submit epoch is in the future" >&2
  exit 2
fi
echo "attention-stop diagnostic start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}; reserved GPU hidden from diagnostic"

"${python_bin}" "${runner}" \
  --rollouts "${rollouts}" --expected-rollouts-sha256 9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e \
  --attention-features "${features}" --expected-attention-features-sha256 009cdbfa1761f9b53b791a50f70b7e98bdf275eec8743d8bcaf078a52ded8ce8 \
  --attention-complete "${feature_complete}" --expected-attention-complete-sha256 6eb313cf1bf4e5f61a8decc0c6ef70605009826c1bf4f815deb7f8111ec7bf40 \
  --attention-audit "${feature_audit}" --expected-attention-audit-sha256 27ba5df9d45f9837f685d64589e32740238de6ff0ce46ce54ce6a1ac21a1d471 \
  --attention-evaluation "${evaluation}" --expected-attention-evaluation-sha256 5c8bced0fdad0a4f7c3ad0dca8bf8cf31d40be4c9d2318c6b42ea72d065366ee \
  --attention-evaluation-complete "${evaluation_complete}" --expected-attention-evaluation-complete-sha256 ea38fb7adb024a1c96a6ec160d921687affb3ac0222aecba3f5d422728a4cbf5 \
  --bootstrap-indices "${bootstrap_indices}" --expected-bootstrap-indices-sha256 17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6 \
  --bootstrap-sources "${bootstrap_sources}" --expected-bootstrap-sources-sha256 5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0 \
  --protocol "${protocol}" --expected-protocol-sha256 "${expected_protocol_sha256}" \
  --expected-attention-code-revision "${feature_code_revision}" \
  --expected-model-revision "${model_revision}" \
  --expected-source-features-sha256 "${source_features_sha256}" \
  --output-dir "${output_dir}"

complete="${output_dir}/complete.json"
diagnostic="${output_dir}/diagnostic.json"
if [[ "$(jq -r '.passed' "${complete}")" != true \
  || "$(jq -r '.valid_for_formal_selection' "${complete}")" != false \
  || "$(jq -r '.validation_or_test_inputs_used' "${complete}")" != false \
  || "$(jq -r '.population.decisions' "${diagnostic}")" -ne 23946 \
  || "$(jq -r '.bootstrap.n_resamples' "${diagnostic}")" -ne 20000 ]]; then
  echo "attention-stop diagnostic output contract failed" >&2
  exit 2
fi
job_end_epoch=$(date +%s)
execution="${execution_dir}/job-${SLURM_JOB_ID}.json"
jq -n \
  --arg schema infographicvqa_attention_stop_factorization_execution_v1 \
  --arg job_id "${SLURM_JOB_ID}" --arg code_revision "${expected_revision}" \
  --arg feature_code_revision "${feature_code_revision}" \
  --arg diagnostic_sha256 "$(sha "${diagnostic}")" \
  --arg complete_sha256 "$(sha "${complete}")" \
  --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson total_seconds "$((job_end_epoch - job_start_epoch))" '
  {schema:$schema,job_id:$job_id,code_revision:$code_revision,
   feature_code_revision:$feature_code_revision,accelerator:"CPU",
   gpu_reserved:"NVIDIA RTX 4090",gpu_hidden:true,gpu_count:1,cpu_count:4,
   queue_wait_seconds:$queue_wait_seconds,total_seconds:$total_seconds,
   artifacts:{diagnostic_sha256:$diagnostic_sha256,complete_sha256:$complete_sha256},
   raw_negative_decision_preserved:true,valid_for_formal_selection:false,
   credentials_present:false,validation_or_test_inputs_used:false}' \
  > "${execution}.tmp"
mv "${execution}.tmp" "${execution}"
echo "attention-stop diagnostic end: $(date --iso-8601=seconds)"
printf 'attention_stop_diagnostic_complete=%s execution_sha256=%s\n' \
  "${complete}" "$(sha "${execution}")"
