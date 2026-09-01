#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --job-name=be-infovqa-decar-full-mat
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-infovqa-decar-full-mat-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: worker EXPECTED_REVISION WORKER_SHA256 FREEZE_SHA256 SUBMIT_EPOCH" >&2
  exit 2
fi
expected_revision=$1
expected_worker_sha256=$2
expected_freeze_sha256=$3
submit_epoch=$4
if [[ ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "DECAR full materialization submit epoch must be an integer" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/bin/python
dataset_dir="${repo}/data/infographicvqa/lmms-539088e-train"
download_manifest="${dataset_dir}/download-manifest.json"
source_manifest="${repo}/artifacts/infographicvqa-train-v1/source-audit-v1/source-manifest.jsonl"
outer_folds="${repo}/artifacts/infographicvqa-train-v1/decar-v1/allocation-v1/outer-folds.jsonl"
inner_folds="${repo}/artifacts/infographicvqa-train-v1/decar-v1/allocation-v1/inner-folds.jsonl"
protocol="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-method-protocol-v1.md"
allocation_result="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-allocation-result-v1.md"
pilot_result="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-pilot-result-v1.md"
freeze="${repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-full-materialization-freeze-v1.md"
worker="${repo}/scripts/slurm_infographicvqa_decar_full_materialize.sh"
materializer="${repo}/scripts/materialize_infographicvqa_decar_full.py"
module="${repo}/src/beyond_entropy/infographicvqa_decar_manifest.py"
output_dir="${repo}/artifacts/infographicvqa-train-v1/decar-v1/full-manifest-v1"

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "DECAR full materialization ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

cd "${repo}"
if [[ "$(git rev-parse HEAD)" != "${expected_revision}" ]]; then
  echo "DECAR full materialization tracked revision changed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "DECAR full materialization tracked worktree must be clean" >&2
  exit 2
fi
require_hash "${worker}" "${expected_worker_sha256}" worker
require_hash "${freeze}" "${expected_freeze_sha256}" freeze
require_hash "${materializer}" 7051b2c68a18caab112e519cee708d26f74cac6d3f05b73c53ded20985f4be7f materializer
require_hash "${module}" 6974d3a2a157e04935c80a9bd9344bb9c2c88fe4291fb13b321364f9e513b639 module
require_hash "${download_manifest}" ecc46c6a073ebd89fc114cba6fee5c711c8600e596b5a785bec981d98b168f13 download-manifest
require_hash "${source_manifest}" fc577513dd8f9993f40d14454c7ec4ecf48897ff0d1660479fb5c49d3ae9512a source-manifest
require_hash "${outer_folds}" 7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6a232b0837a outer-folds
require_hash "${inner_folds}" 8b4977776841794185eedd8cbbfdb8e23ec2c44cc0be9931cc2e672b232f344c inner-folds
require_hash "${protocol}" d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342 protocol
require_hash "${allocation_result}" 3d0948cc6840b008cd4b19408ff002ed0756bb0d9f7f5e6b8cdb6d0af5a4da60 allocation-result
require_hash "${pilot_result}" d91f756e82ee2ce58edf4a66b3fde3433d0f2466cc72f04994d758dc1c23f697 pilot-result
if [[ -e "${output_dir}" ]]; then
  echo "DECAR full materialization output already exists" >&2
  exit 2
fi

export PYTHONPATH="${repo}/src"
export PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

start_epoch=$(date +%s)
queue_wait_seconds=$((start_epoch - submit_epoch))
echo "DECAR full materialization start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "tracked revision: ${expected_revision}"

"${python_bin}" "${materializer}" \
  --dataset-dir "${dataset_dir}" \
  --download-manifest "${download_manifest}" \
  --expected-download-manifest-sha256 ecc46c6a073ebd89fc114cba6fee5c711c8600e596b5a785bec981d98b168f13 \
  --source-manifest "${source_manifest}" \
  --expected-source-manifest-sha256 fc577513dd8f9993f40d14454c7ec4ecf48897ff0d1660479fb5c49d3ae9512a \
  --outer-folds "${outer_folds}" \
  --expected-outer-folds-sha256 7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6a232b0837a \
  --inner-folds "${inner_folds}" \
  --expected-inner-folds-sha256 8b4977776841794185eedd8cbbfdb8e23ec2c44cc0be9931cc2e672b232f344c \
  --protocol "${protocol}" \
  --expected-protocol-sha256 d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342 \
  --allocation-result "${allocation_result}" \
  --expected-allocation-result-sha256 3d0948cc6840b008cd4b19408ff002ed0756bb0d9f7f5e6b8cdb6d0af5a4da60 \
  --pilot-result "${pilot_result}" \
  --expected-pilot-result-sha256 d91f756e82ee2ce58edf4a66b3fde3433d0f2466cc72f04994d758dc1c23f697 \
  --output-dir "${output_dir}"

complete="${output_dir}/complete.json"
report="${output_dir}/report.json"
task_manifest="${output_dir}/task-manifest.jsonl"
image_manifest="${output_dir}/image-manifest.jsonl"
if [[ "$(jq -r '.population.questions' "${report}")" -ne 23946 \
  || "$(jq -r '.population.sources' "${report}")" -ne 2204 \
  || "$(jq -r '.population.images' "${report}")" -ne 4406 \
  || "$(jq -r '.audits.validation_or_test_rows_read' "${report}")" != false \
  || "$(jq -r '.audits.fold_ids_not_serialized_in_task_manifest' "${report}")" != true \
  || "$(wc -l < "${task_manifest}")" -ne 23946 \
  || "$(wc -l < "${image_manifest}")" -ne 4406 \
  || "$(jq -r '.task_manifest.sha256' "${complete}")" != "$(sha "${task_manifest}")" ]]; then
  echo "DECAR full materialization terminal audit failed" >&2
  exit 2
fi

end_epoch=$(date +%s)
execution="${output_dir}/execution.json"
jq -n \
  --arg schema infographicvqa_decar_full_materialization_execution_v1 \
  --arg job_id "${SLURM_JOB_ID}" \
  --arg code_revision "${expected_revision}" \
  --arg complete_sha256 "$(sha "${complete}")" \
  --arg report_sha256 "$(sha "${report}")" \
  --arg task_manifest_sha256 "$(sha "${task_manifest}")" \
  --arg image_manifest_sha256 "$(sha "${image_manifest}")" \
  --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson runtime_seconds "$((end_epoch - start_epoch))" \
  '{schema:$schema,job_id:$job_id,code_revision:$code_revision,queue_wait_seconds:$queue_wait_seconds,runtime_seconds:$runtime_seconds,population:{questions:23946,sources:2204,images:4406},artifacts:{complete_sha256:$complete_sha256,report_sha256:$report_sha256,task_manifest_sha256:$task_manifest_sha256,image_manifest_sha256:$image_manifest_sha256},validation_or_test_inputs_used:false,task_outcomes_computed:false}' \
  > "${execution}.tmp"
mv "${execution}.tmp" "${execution}"

echo "DECAR full materialization end: $(date --iso-8601=seconds)"
printf 'infographicvqa_decar_full_materialization_complete=%s execution_sha256=%s\n' \
  "${execution}" "$(sha "${execution}")"
