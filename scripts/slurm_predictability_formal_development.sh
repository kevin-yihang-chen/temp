#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=be-pred-formal-dev
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-pred-formal-dev-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_PRED_EXPECTED_CODE_REVISION
  BE_PRED_EXPECTED_GPU_TOKEN
  BE_PRED_BENCHMARK
  BE_PRED_ROLE
  BE_PRED_RUN_ROOT
  BE_PRED_EXPECTED_STATES
  BE_PRED_CHECKPOINT_INTERVAL
  BE_PRED_WORKER_SHA256
  BE_PRED_FINALIZER_SHA256
  BE_PRED_CLI_SHA256
  BE_PRED_BACKEND_SHA256
  BE_PRED_SEMANTIC_SHA256
  BE_PRED_FEATURES_SHA256
  BE_PRED_POST_ACTION_SHA256
  BE_PRED_IMAGE_OPS_SHA256
  BE_PRED_AUDIT_SHA256
  BE_PRED_ARTIFACTS_SHA256
  BE_PRED_SOURCE_MANIFEST_SHA256
  BE_PRED_PROTOCOL_SHA256
  BE_PRED_EXECUTION_CONFIG_SHA256
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
protocol="${repo_dir}/configs/predictability_audit_v1.json"
execution_config="${repo_dir}/configs/predictability_formal_execution_v1.json"
worker="${repo_dir}/scripts/slurm_predictability_formal_development.sh"
finalizer="${repo_dir}/scripts/finalize_predictability_development_role.py"
collector_cli="${repo_dir}/src/beyond_entropy/cli.py"
backend_module="${repo_dir}/src/beyond_entropy/qwen_backend.py"
semantic_module="${repo_dir}/src/beyond_entropy/qwen_semantic.py"
features_module="${repo_dir}/src/beyond_entropy/predictability_features.py"
post_action_module="${repo_dir}/src/beyond_entropy/predictability_post_action.py"
image_ops_module="${repo_dir}/src/beyond_entropy/image_ops.py"
audit_module="${repo_dir}/src/beyond_entropy/predictability_audit.py"
artifacts_module="${repo_dir}/src/beyond_entropy/predictability_matrix_artifacts.py"
source_manifest="${repo_dir}/data/predictability-audit-v1/${BE_PRED_BENCHMARK}/${BE_PRED_ROLE}/manifest.jsonl"
run_dir="${BE_PRED_RUN_ROOT}/${BE_PRED_BENCHMARK}/${BE_PRED_ROLE}"
attempt_dir="${run_dir}/attempts"
rollouts="${run_dir}/rollouts.jsonl"
rollout_provenance="${run_dir}/rollouts.provenance.json"
features="${run_dir}/features.pt"
completion="${run_dir}/complete.json"
status_file="${attempt_dir}/job-${SLURM_JOB_ID}.execution.json"
lock_file="${run_dir}/active.lock"

check_hash() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "formal development ${label} hash mismatch" >&2
    exit 2
  fi
}

case "${BE_PRED_BENCHMARK}" in
  chartqa|docvqa|hrbench) scorer="${BE_PRED_BENCHMARK}" ;;
  *) echo "unsupported formal benchmark" >&2; exit 2 ;;
esac
case "${BE_PRED_ROLE}" in
  train|validation) ;;
  *) echo "formal development role must be train or validation" >&2; exit 2 ;;
esac
if [[ ! "${BE_PRED_EXPECTED_STATES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid expected state count" >&2
  exit 2
fi
if [[ ! "${BE_PRED_CHECKPOINT_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid checkpoint interval" >&2
  exit 2
fi

actual_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_revision}" != "${BE_PRED_EXPECTED_CODE_REVISION}" ]]; then
  echo "formal development code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before formal development" >&2
  exit 2
fi
check_hash "${source_manifest}" "${BE_PRED_SOURCE_MANIFEST_SHA256}" manifest
check_hash "${protocol}" "${BE_PRED_PROTOCOL_SHA256}" protocol
check_hash "${execution_config}" "${BE_PRED_EXECUTION_CONFIG_SHA256}" "execution config"
check_hash "${worker}" "${BE_PRED_WORKER_SHA256}" worker
check_hash "${finalizer}" "${BE_PRED_FINALIZER_SHA256}" finalizer
check_hash "${collector_cli}" "${BE_PRED_CLI_SHA256}" "collector CLI"
check_hash "${backend_module}" "${BE_PRED_BACKEND_SHA256}" backend
check_hash "${semantic_module}" "${BE_PRED_SEMANTIC_SHA256}" semantic
check_hash "${features_module}" "${BE_PRED_FEATURES_SHA256}" features
check_hash "${post_action_module}" "${BE_PRED_POST_ACTION_SHA256}" "post-action probe"
check_hash "${image_ops_module}" "${BE_PRED_IMAGE_OPS_SHA256}" "image operations"
check_hash "${audit_module}" "${BE_PRED_AUDIT_SHA256}" audit
check_hash "${artifacts_module}" "${BE_PRED_ARTIFACTS_SHA256}" "artifact loader"
if [[ "$(jq -er '.protocol.sha256' "${execution_config}")" != "${BE_PRED_PROTOCOL_SHA256}" ]]; then
  echo "execution config protocol binding mismatch" >&2
  exit 2
fi
if [[ "$(jq -er '.checkpoint_interval_states' "${execution_config}")" != "${BE_PRED_CHECKPOINT_INTERVAL}" ]]; then
  echo "execution config checkpoint interval mismatch" >&2
  exit 2
fi
configured_states=$(jq -er --arg b "${BE_PRED_BENCHMARK}" --arg r "${BE_PRED_ROLE}" '.roles_in_submission_order[] | select(.benchmark == $b and .role == $r) | .states' "${execution_config}")
if [[ "${configured_states}" != "${BE_PRED_EXPECTED_STATES}" ]]; then
  echo "execution config state count mismatch" >&2
  exit 2
fi
actual_states=$(awk 'NF { count += 1 } END { print count + 0 }' "${source_manifest}")
if [[ "${actual_states}" != "${BE_PRED_EXPECTED_STATES}" ]]; then
  echo "development manifest state count mismatch" >&2
  exit 2
fi
if [[ -e "${completion}" ]]; then
  echo "formal development role is already sealed" >&2
  exit 2
fi
mkdir -p "${attempt_dir}"
if ! ( set -o noclobber; printf '%s\n' "${SLURM_JOB_ID}" > "${lock_file}" ) 2>/dev/null; then
  echo "formal development role already has an active lock" >&2
  exit 2
fi

finish() {
  local exit_code=$?
  trap - EXIT
  set +e
  rm -f -- "${lock_file}"
  "${python_bin}" - "${status_file}" "${exit_code}" "${actual_revision}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "predictability_formal_development_execution_v1",
    "status": "completed" if int(sys.argv[2]) == 0 else "failed",
    "exit_code": int(sys.argv[2]),
    "code_revision": sys.argv[3],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "benchmark": os.environ.get("BE_PRED_BENCHMARK"),
    "role": os.environ.get("BE_PRED_ROLE"),
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
  exit "${exit_code}"
}
trap finish EXIT

export PYTHONPATH="${repo_dir}:${repo_dir}/src"
export BE_CODE_REVISION="${actual_revision}"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

gpu_info=$("${python_bin}" - <<'PY'
import json
import torch
print(json.dumps({
    "count": torch.cuda.device_count(),
    "name": torch.cuda.get_device_name(0) if torch.cuda.device_count() else None,
}))
PY
)
if [[ "$(jq -r '.count' <<< "${gpu_info}")" -ne 1 ]]; then
  echo "formal development requires exactly one visible GPU" >&2
  exit 2
fi
gpu_name=$(jq -r '.name' <<< "${gpu_info}")
if [[ "${gpu_name,,}" != *"${BE_PRED_EXPECTED_GPU_TOKEN,,}"* ]]; then
  echo "expected ${BE_PRED_EXPECTED_GPU_TOKEN}, got ${gpu_name}" >&2
  exit 2
fi

model=$(jq -er '.feature_extraction.model' "${protocol}")
model_revision=$(jq -er '.feature_extraction.model_revision' "${protocol}")
max_new_tokens=$(jq -er '.feature_extraction.max_new_tokens' "${protocol}")
min_pixels=$(jq -er '.feature_extraction.min_pixels' "${protocol}")
max_pixels=$(jq -er '.feature_extraction.max_pixels' "${protocol}")
"${python_bin}" - "${model}" "${model_revision}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_files_only=True)
PY

cd "${repo_dir}"
collect_resume=()
if [[ -e "${rollouts}" ]]; then collect_resume=(--resume); fi
"${python_bin}" -m beyond_entropy collect-qwen \
  --manifest "${source_manifest}" \
  --expected-manifest-sha256 "${BE_PRED_SOURCE_MANIFEST_SHA256}" \
  --output "${rollouts}" \
  --checkpoint-interval "${BE_PRED_CHECKPOINT_INTERVAL}" \
  --model "${model}" \
  --model-revision "${model_revision}" \
  --scorer "${scorer}" \
  --candidate-count 4 \
  --proposer ug-grid \
  --visual-crop-ratio 2.0 \
  --visual-cost 1.0 \
  --generation-seeds 0 \
  --bootstrap-resamples 100 \
  --bootstrap-seed 20260903 \
  --scientific-status "formal development ${BE_PRED_BENCHMARK} ${BE_PRED_ROLE}; not test" \
  --max-new-tokens "${max_new_tokens}" \
  --min-pixels "${min_pixels}" \
  --max-pixels "${max_pixels}" \
  --device-map cuda:0 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --system-prompt "You are a helpful assistant." \
  "${collect_resume[@]}"

feature_resume=()
if [[ -e "${features}" ]]; then feature_resume=(--resume); fi
"${python_bin}" scripts/extract_predictability_features.py \
  --rollouts "${rollouts}" \
  --manifest "${source_manifest}" \
  --output "${features}" \
  --dataset-role "${BE_PRED_ROLE}" \
  --model "${model}" \
  --revision "${model_revision}" \
  --device-map cuda:0 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --min-pixels "${min_pixels}" \
  --max-pixels "${max_pixels}" \
  --checkpoint-interval "${BE_PRED_CHECKPOINT_INTERVAL}" \
  "${feature_resume[@]}"

"${python_bin}" scripts/finalize_predictability_development_role.py \
  --benchmark "${BE_PRED_BENCHMARK}" \
  --role "${BE_PRED_ROLE}" \
  --manifest "${source_manifest}" \
  --rollouts "${rollouts}" \
  --rollout-provenance "${rollout_provenance}" \
  --features "${features}" \
  --protocol "${protocol}" \
  --expected-protocol-sha256 "${BE_PRED_PROTOCOL_SHA256}" \
  --code-revision "${actual_revision}" \
  --expected-states "${BE_PRED_EXPECTED_STATES}" \
  --output "${completion}"
