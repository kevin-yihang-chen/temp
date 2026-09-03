#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=be-pred-feature-smoke
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-pred-feature-smoke-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_PRED_EXPECTED_CODE_REVISION
  BE_PRED_EXPECTED_GPU_TOKEN
  BE_PRED_BENCHMARK
  BE_PRED_RUN_ROOT
  BE_PRED_SMOKE_COUNT
  BE_PRED_WORKER_SHA256
  BE_PRED_CLI_SHA256
  BE_PRED_BACKEND_SHA256
  BE_PRED_SEMANTIC_SHA256
  BE_PRED_FEATURES_SHA256
  BE_PRED_POST_ACTION_SHA256
  BE_PRED_IMAGE_OPS_SHA256
  BE_PRED_AUDIT_SHA256
  BE_PRED_SOURCE_MANIFEST_SHA256
  BE_PRED_PROTOCOL_SHA256
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
worker="${repo_dir}/scripts/slurm_predictability_feature_smoke.sh"
collector_cli="${repo_dir}/src/beyond_entropy/cli.py"
backend_module="${repo_dir}/src/beyond_entropy/qwen_backend.py"
semantic_module="${repo_dir}/src/beyond_entropy/qwen_semantic.py"
features_module="${repo_dir}/src/beyond_entropy/predictability_features.py"
post_action_module="${repo_dir}/src/beyond_entropy/predictability_post_action.py"
image_ops_module="${repo_dir}/src/beyond_entropy/image_ops.py"
audit_module="${repo_dir}/src/beyond_entropy/predictability_audit.py"
model=Qwen/Qwen2.5-VL-3B-Instruct
model_revision=66285546d2b821cf421d4f5eb2576359d3770cd3
case "${BE_PRED_BENCHMARK}" in
  chartqa)
    source_manifest="${repo_dir}/data/predictability-audit-v1/chartqa/train/manifest.jsonl"
    scorer=chartqa
    ;;
  docvqa)
    source_manifest="${repo_dir}/data/predictability-audit-v1/docvqa/train/manifest.jsonl"
    scorer=docvqa
    ;;
  hrbench)
    source_manifest="${repo_dir}/data/predictability-audit-v1/hrbench/train/manifest.jsonl"
    scorer=hrbench
    ;;
  *)
    echo "unsupported predictability smoke benchmark: ${BE_PRED_BENCHMARK}" >&2
    exit 2
    ;;
esac
run_dir="${BE_PRED_RUN_ROOT}/job-${SLURM_JOB_ID}"
manifest="${run_dir}/manifest.jsonl"
rollouts="${run_dir}/rollouts.jsonl"
features="${run_dir}/features.pt"
status_file="${run_dir}/execution.json"

check_hash() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "predictability smoke ${label} hash mismatch" >&2
    exit 2
  fi
}

actual_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_revision}" != "${BE_PRED_EXPECTED_CODE_REVISION}" ]]; then
  echo "predictability smoke code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before predictability smoke" >&2
  exit 2
fi
check_hash "${source_manifest}" "${BE_PRED_SOURCE_MANIFEST_SHA256}" manifest
check_hash "${protocol}" "${BE_PRED_PROTOCOL_SHA256}" protocol
check_hash "${worker}" "${BE_PRED_WORKER_SHA256}" worker
check_hash "${collector_cli}" "${BE_PRED_CLI_SHA256}" "collector CLI"
check_hash "${backend_module}" "${BE_PRED_BACKEND_SHA256}" backend
check_hash "${semantic_module}" "${BE_PRED_SEMANTIC_SHA256}" semantic
check_hash "${features_module}" "${BE_PRED_FEATURES_SHA256}" features
check_hash "${post_action_module}" "${BE_PRED_POST_ACTION_SHA256}" "post-action probe"
check_hash "${image_ops_module}" "${BE_PRED_IMAGE_OPS_SHA256}" "image operations"
check_hash "${audit_module}" "${BE_PRED_AUDIT_SHA256}" audit

if [[ -e "${run_dir}" ]]; then
  echo "refusing to reuse predictability smoke run directory: ${run_dir}" >&2
  exit 2
fi
mkdir -p "${run_dir}"

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
    "schema": "predictability_feature_smoke_execution_v1",
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

gpu_info=$(
  "${python_bin}" - <<'PY'
import json
import torch

print(json.dumps({
    "count": torch.cuda.device_count(),
    "name": torch.cuda.get_device_name(0) if torch.cuda.device_count() else None,
}))
PY
)
if [[ "$(jq -r '.count' <<< "${gpu_info}")" -ne 1 ]]; then
  echo "predictability smoke requires exactly one visible GPU" >&2
  exit 2
fi
gpu_name=$(jq -r '.name' <<< "${gpu_info}")
if [[ "${gpu_name,,}" != *"${BE_PRED_EXPECTED_GPU_TOKEN,,}"* ]]; then
  echo "expected ${BE_PRED_EXPECTED_GPU_TOKEN}, got ${gpu_name}" >&2
  exit 2
fi

"${python_bin}" - "${model}" "${model_revision}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_files_only=True)
PY

"${python_bin}" - "${source_manifest}" "${manifest}" "${BE_PRED_SMOKE_COUNT}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
count = int(sys.argv[3])
if count <= 0:
    raise SystemExit("smoke count must be positive")
rows = []
with source.open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        image = Path(str(row["image_path"]))
        row["image_path"] = str(
            (image if image.is_absolute() else source.parent / image).resolve()
        )
        rows.append(row)
        if len(rows) == count:
            break
if len(rows) != count:
    raise SystemExit(f"requested {count} smoke rows, found {len(rows)}")
payload = "".join(
    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
).encode()
temporary = destination.with_name(destination.name + ".tmp")
with temporary.open("xb") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
temporary.replace(destination)
print(hashlib.sha256(payload).hexdigest())
PY
manifest_sha256=$(sha256sum "${manifest}" | cut -d ' ' -f 1)

cd "${repo_dir}"
"${python_bin}" -m beyond_entropy collect-qwen \
  --manifest "${manifest}" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --output "${rollouts}" \
  --checkpoint-interval 1 \
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
  --scientific-status "train-role ${BE_PRED_BENCHMARK} engineering smoke; not a benchmark claim" \
  --max-new-tokens 16 \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --attention-implementation sdpa \
  --system-prompt "You are a helpful assistant."

"${python_bin}" scripts/extract_predictability_features.py \
  --rollouts "${rollouts}" \
  --manifest "${manifest}" \
  --output "${features}" \
  --dataset-role retrospective_smoke \
  --model "${model}" \
  --revision "${model_revision}" \
  --device-map cuda:0 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --min-pixels 200704 \
  --max-pixels 602112 \
  --checkpoint-interval 1

"${python_bin}" - "${manifest}" "${rollouts}" "${features}" "${run_dir}/smoke-report.json" "${BE_PRED_SMOKE_COUNT}" <<'PY'
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.predictability_audit import collapse_fixed_entropy_tool
from beyond_entropy.predictability_features import (
    load_predictability_feature_dataset,
    post_action_probe_examples_from_feature_dataset,
)

manifest, rollouts, features, output = map(
    lambda value: Path(value).resolve(), sys.argv[1:5]
)
expected_states = int(sys.argv[5])
records = read_jsonl(rollouts)
payload, examples = load_predictability_feature_dataset(features)
post_action_examples = post_action_probe_examples_from_feature_dataset(payload)
if len(records) != 5 * expected_states or len(examples) != expected_states:
    raise SystemExit("smoke coverage differs from states with four sibling crops")
outcomes = collapse_fixed_entropy_tool(records)
if len(outcomes) != expected_states or any(
    outcome.tool_calls != 4 or outcome.tool_cost != 4.0 for outcome in outcomes
):
    raise SystemExit("fixed visual tool did not charge exactly four calls")
dimensions = {
    level: sorted({len(example.inputs.feature_vector(level)) for example in examples})
    for level in ("l0_uncertainty", "l1_shallow", "l2_semantic", "l3_frozen_qwen")
}
if dimensions["l0_uncertainty"] != [3] or dimensions["l1_shallow"] != [22]:
    raise SystemExit(f"L0/L1 dimension contract failed: {dimensions}")
if not all(
    math.isfinite(value)
    for example in examples
    for value in example.inputs.feature_vector("l0_uncertainty")
):
    raise SystemExit("L0 confidence statistics are non-finite")
post_action_dimensions = sorted(
    {len(example.inputs.feature_vector()) for example in post_action_examples}
)
if post_action_dimensions != [6167]:
    raise SystemExit(
        f"post-action probe dimension contract failed: {post_action_dimensions}"
    )
if not all(
    math.isfinite(value)
    for example in post_action_examples
    for value in example.inputs.feature_vector()
):
    raise SystemExit("post-action probe features are non-finite")
metadata = payload["metadata"]
if metadata.get("dataset_role") != "retrospective_smoke":
    raise SystemExit("feature smoke role changed")
report = {
    "schema": "predictability_feature_real_smoke_v2",
    "passed": True,
    "scientific_status": (
        f"engineering smoke on {os.environ.get('BE_PRED_BENCHMARK')} train role; "
        "not a benchmark claim"
    ),
    "benchmark": os.environ.get("BE_PRED_BENCHMARK"),
    "code_revision": os.environ.get("BE_CODE_REVISION"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "records": len(records),
    "states": len(examples),
    "fixed_tool_calls_per_state": sorted({outcome.tool_calls for outcome in outcomes}),
    "feature_dimensions": {
        level: values[0] if len(values) == 1 else values
        for level, values in dimensions.items()
    },
    "post_action_probe_dimension": post_action_dimensions[0],
    "post_action_probe_deployable": False,
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "rollouts_sha256": hashlib.sha256(rollouts.read_bytes()).hexdigest(),
    "features_sha256": hashlib.sha256(features.read_bytes()).hexdigest(),
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(json.dumps(report, indent=2, sort_keys=True))
PY
