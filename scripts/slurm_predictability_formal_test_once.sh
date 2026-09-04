#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=be-pred-test-once
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-pred-test-once-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_PRED_TEST_PLAN
  BE_PRED_TEST_PLAN_SHA256
  BE_PRED_TEST_EXPECTED_CODE_REVISION
  BE_PRED_TEST_EXPECTED_GPU_TOKEN
  BE_PRED_TEST_WORKER_SHA256
  BE_PRED_TEST_STARTER_SHA256
  BE_PRED_TEST_FINALIZER_SHA256
  BE_PRED_TEST_SPEC_BUILDER_SHA256
  BE_PRED_TEST_EVALUATOR_SHA256
  BE_PRED_TEST_RENDERER_SHA256
  BE_PRED_TEST_TRANSACTION_MODULE_SHA256
  BE_PRED_TEST_ARTIFACTS_MODULE_SHA256
  BE_PRED_TEST_MATRIX_MODULE_SHA256
  BE_PRED_TEST_VERDICT_MODULE_SHA256
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
worker="${repo_dir}/scripts/slurm_predictability_formal_test_once.sh"
starter="${repo_dir}/scripts/start_predictability_test_transaction.py"
finalizer="${repo_dir}/scripts/finalize_predictability_test_role.py"
spec_builder="${repo_dir}/scripts/build_predictability_test_input_spec.py"
evaluator="${repo_dir}/scripts/evaluate_predictability_matrix_test_once.py"
renderer="${repo_dir}/scripts/render_predictability_audit.py"
transaction_module="${repo_dir}/src/beyond_entropy/predictability_test_transaction.py"
artifacts_module="${repo_dir}/src/beyond_entropy/predictability_matrix_artifacts.py"
matrix_module="${repo_dir}/src/beyond_entropy/predictability_matrix.py"
verdict_module="${repo_dir}/src/beyond_entropy/predictability_verdict.py"
plan=$(realpath "${BE_PRED_TEST_PLAN}")

check_hash() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "formal test ${label} hash mismatch" >&2
    exit 2
  fi
}

actual_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_revision}" != "${BE_PRED_TEST_EXPECTED_CODE_REVISION}" ]]; then
  echo "formal test code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain)" ]]; then
  echo "worktree must be fully clean before formal test" >&2
  exit 2
fi
check_hash "${plan}" "${BE_PRED_TEST_PLAN_SHA256}" plan
check_hash "${worker}" "${BE_PRED_TEST_WORKER_SHA256}" worker
check_hash "${starter}" "${BE_PRED_TEST_STARTER_SHA256}" starter
check_hash "${finalizer}" "${BE_PRED_TEST_FINALIZER_SHA256}" finalizer
check_hash "${spec_builder}" "${BE_PRED_TEST_SPEC_BUILDER_SHA256}" "spec builder"
check_hash "${evaluator}" "${BE_PRED_TEST_EVALUATOR_SHA256}" evaluator
check_hash "${renderer}" "${BE_PRED_TEST_RENDERER_SHA256}" renderer
check_hash "${transaction_module}" "${BE_PRED_TEST_TRANSACTION_MODULE_SHA256}" "transaction module"
check_hash "${artifacts_module}" "${BE_PRED_TEST_ARTIFACTS_MODULE_SHA256}" "artifacts module"
check_hash "${matrix_module}" "${BE_PRED_TEST_MATRIX_MODULE_SHA256}" "matrix module"
check_hash "${verdict_module}" "${BE_PRED_TEST_VERDICT_MODULE_SHA256}" "verdict module"

run_root=$(jq -er '.run_root' "${plan}")
attempt_dir="${run_root}/attempts"
status_file="${attempt_dir}/job-${SLURM_JOB_ID}.execution.json"
mkdir -p "${attempt_dir}"
finish() {
  local exit_code=$?
  trap - EXIT
  set +e
  "${python_bin}" - "${status_file}" "${exit_code}" "${actual_revision}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "predictability_formal_test_execution_v1",
    "status": "completed" if int(sys.argv[2]) == 0 else "failed",
    "exit_code": int(sys.argv[2]),
    "code_revision": sys.argv[3],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
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
  echo "formal test requires exactly one visible GPU" >&2
  exit 2
fi
gpu_name=$(jq -r '.name' <<< "${gpu_info}")
if [[ "${gpu_name,,}" != *"${BE_PRED_TEST_EXPECTED_GPU_TOKEN,,}"* ]]; then
  echo "expected ${BE_PRED_TEST_EXPECTED_GPU_TOKEN}, got ${gpu_name}" >&2
  exit 2
fi

protocol=$(jq -er '.protocol.path' "${plan}")
protocol_sha256=$(jq -er '.protocol.sha256' "${plan}")
model=$(jq -er '.feature_extraction.model' "${protocol}")
model_revision=$(jq -er '.feature_extraction.model_revision' "${protocol}")
max_new_tokens=$(jq -er '.feature_extraction.max_new_tokens' "${protocol}")
min_pixels=$(jq -er '.feature_extraction.min_pixels' "${protocol}")
max_pixels=$(jq -er '.feature_extraction.max_pixels' "${protocol}")
checkpoint_interval=$(jq -er '.checkpoint_interval_states' "$(jq -er '.execution_config.path' "${plan}")")
"${python_bin}" - "${model}" "${model_revision}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_files_only=True)
PY

# This is the irreversible boundary. No held-out manifest is stat'ed, hashed,
# counted, or loaded above this call.
"${python_bin}" "${starter}" \
  --plan "${plan}" \
  --expected-plan-sha256 "${BE_PRED_TEST_PLAN_SHA256}" \
  --repo-root "${repo_dir}"
access_ledger=$(jq -er '.access_ledger' "${plan}")
access_ledger_sha256=$(sha256sum "${access_ledger}")
access_ledger_sha256=${access_ledger_sha256%% *}
allocation_report=$(jq -er '.allocation_report.path' "${plan}")
allocation_report_sha256=$(jq -er '.allocation_report.sha256' "${plan}")
check_hash "${allocation_report}" "${allocation_report_sha256}" "allocation report"

cd "${repo_dir}"
for benchmark in chartqa docvqa hrbench; do
  manifest=$(jq -er --arg b "${benchmark}" '.benchmarks[$b].manifest_path' "${plan}")
  expected_manifest_sha256=$(jq -er --arg b "${benchmark}" '.benchmarks[$b].expected_manifest_sha256' "${plan}")
  expected_states=$(jq -er --arg b "${benchmark}" '.benchmarks[$b].expected_states' "${plan}")
  if [[ "$(jq -er --arg b "${benchmark}" '.benchmarks[$b].test.historically_opened' "${allocation_report}")" != "false" ]]; then
    echo "${benchmark} test allocation was historically opened" >&2
    exit 2
  fi
  if [[ "$(jq -er --arg b "${benchmark}" '.benchmarks[$b].test.manifest_sha256' "${allocation_report}")" != "${expected_manifest_sha256}" ]]; then
    echo "${benchmark} test allocation manifest hash mismatch" >&2
    exit 2
  fi
  if [[ "$(jq -er --arg b "${benchmark}" '.benchmarks[$b].test.states' "${allocation_report}")" != "${expected_states}" ]]; then
    echo "${benchmark} test allocation state count mismatch" >&2
    exit 2
  fi
  check_hash "${manifest}" "${expected_manifest_sha256}" "${benchmark} test manifest"
  actual_states=$(awk 'NF { count += 1 } END { print count + 0 }' "${manifest}")
  if [[ "${actual_states}" != "${expected_states}" ]]; then
    echo "${benchmark} test manifest state count mismatch" >&2
    exit 2
  fi
  role_dir="${run_root}/${benchmark}/test"
  if [[ -e "${role_dir}" ]]; then
    echo "formal test role directory already exists; automatic retry forbidden" >&2
    exit 2
  fi
  mkdir -p "${role_dir}"
  rollouts="${role_dir}/rollouts.jsonl"
  rollout_provenance="${role_dir}/rollouts.provenance.json"
  features="${role_dir}/features.pt"
  completion="${role_dir}/complete.json"

  "${python_bin}" -m beyond_entropy collect-qwen \
    --manifest "${manifest}" \
    --expected-manifest-sha256 "${expected_manifest_sha256}" \
    --output "${rollouts}" \
    --checkpoint-interval "${checkpoint_interval}" \
    --model "${model}" \
    --model-revision "${model_revision}" \
    --scorer "${benchmark}" \
    --candidate-count 4 \
    --proposer ug-grid \
    --visual-crop-ratio 2.0 \
    --visual-cost 1.0 \
    --generation-seeds 0 \
    --bootstrap-resamples 100 \
    --bootstrap-seed 20260903 \
    --scientific-status "formal held-out one-shot ${benchmark} test" \
    --max-new-tokens "${max_new_tokens}" \
    --min-pixels "${min_pixels}" \
    --max-pixels "${max_pixels}" \
    --device-map cuda:0 \
    --dtype bfloat16 \
    --attention-implementation sdpa \
    --system-prompt "You are a helpful assistant."

  "${python_bin}" scripts/extract_predictability_features.py \
    --rollouts "${rollouts}" \
    --manifest "${manifest}" \
    --output "${features}" \
    --dataset-role test \
    --model "${model}" \
    --revision "${model_revision}" \
    --device-map cuda:0 \
    --dtype bfloat16 \
    --attention-implementation sdpa \
    --min-pixels "${min_pixels}" \
    --max-pixels "${max_pixels}" \
    --checkpoint-interval "${checkpoint_interval}"

  "${python_bin}" "${finalizer}" \
    --benchmark "${benchmark}" \
    --manifest "${manifest}" \
    --rollouts "${rollouts}" \
    --rollout-provenance "${rollout_provenance}" \
    --features "${features}" \
    --protocol "${protocol}" \
    --expected-protocol-sha256 "${protocol_sha256}" \
    --access-ledger "${access_ledger}" \
    --expected-access-ledger-sha256 "${access_ledger_sha256}" \
    --test-transaction-plan-sha256 "${BE_PRED_TEST_PLAN_SHA256}" \
    --code-revision "${actual_revision}" \
    --expected-states "${expected_states}" \
    --output "${completion}"
done

"${python_bin}" "${spec_builder}" \
  --plan "${plan}" \
  --expected-plan-sha256 "${BE_PRED_TEST_PLAN_SHA256}" \
  --repo-root "${repo_dir}"
test_input_spec=$(jq -er '.test_input_spec' "${plan}")
test_input_spec_sha256=$(sha256sum "${test_input_spec}")
test_input_spec_sha256=${test_input_spec_sha256%% *}
"${python_bin}" "${evaluator}" \
  --input-spec "${test_input_spec}" \
  --input-spec-sha256 "${test_input_spec_sha256}" \
  --repo-root "${repo_dir}"
report_output=$(jq -er '.report_output' "${plan}")
report_sha256=$(sha256sum "${report_output}")
report_sha256=${report_sha256%% *}
"${python_bin}" "${renderer}" \
  --report "${report_output}" \
  --expected-report-sha256 "${report_sha256}" \
  --output "$(jq -er '.final_audit_output' "${plan}")"
