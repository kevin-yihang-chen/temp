#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=be-screenqa-7b-smoke
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-7b-smoke-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_BB7_EXPECTED_CODE_REVISION
  BE_BB7_EXPECTED_GPU_TOKEN
  BE_BB7_RUN_ROOT
  BE_BB7_WORKER_SHA256
  BE_BB7_CLI_SHA256
  BE_BB7_BACKEND_SHA256
  BE_BB7_ROLLOUT_SHA256
  BE_BB7_CROPS_SHA256
  BE_BB7_BENCHMARKS_SHA256
  BE_BB7_SCORE_MODULE_SHA256
  BE_BB7_SCORER_SHA256
  BE_BB7_VERIFIER_MODULE_SHA256
  BE_BB7_VERIFIER_CLI_SHA256
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
manifest="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-manifest-v1/backbone-7b-source512-manifest-v1.jsonl"
protocol="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-diagnostic-protocol-v1.md"
activation="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/backbone-7b-population-activation-v1.md"
collector_cli="${repo_dir}/src/beyond_entropy/cli.py"
backend_module="${repo_dir}/src/beyond_entropy/qwen_backend.py"
rollout_module="${repo_dir}/src/beyond_entropy/rollout.py"
crops_module="${repo_dir}/src/beyond_entropy/crops.py"
benchmarks_module="${repo_dir}/src/beyond_entropy/benchmarks.py"
score_module="${repo_dir}/src/beyond_entropy/answer_likelihood.py"
scorer="${repo_dir}/scripts/score_visual_action_answer_nll.py"
verifier_module="${repo_dir}/src/beyond_entropy/backbone_smoke.py"
verifier="${repo_dir}/scripts/verify_backbone_diagnostic_smoke.py"
worker="${repo_dir}/scripts/slurm_screenqa_backbone_7b_smoke.sh"
manifest_sha256=4af43ac80a1666c174774d1c33383adca625e1ef4fc535ffb74e627f149290d0
protocol_sha256=1cd70d11168e12a2855ec01e8a869d89b82c4e87c3d864c566ed7db02bb61474
activation_sha256=a26b8bc6e8a7c81df3cad59f05ac3c9b35b6c340e71f5596175556cd0af6ee6e
model=Qwen/Qwen2.5-VL-7B-Instruct
model_revision=cc594898137f460bfe9f0759e9844b3ce807cfb5
run_dir="${BE_BB7_RUN_ROOT}/job-${SLURM_JOB_ID}"
rollouts="${run_dir}/rollouts.jsonl"
answer_nll="${run_dir}/answer-nll.jsonl"

check_hash() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Qwen-7B smoke ${label} hash mismatch" >&2
    exit 2
  fi
}

actual_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_revision}" != "${BE_BB7_EXPECTED_CODE_REVISION}" ]]; then
  echo "Qwen-7B smoke code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before Qwen-7B smoke" >&2
  exit 2
fi
check_hash "${manifest}" "${manifest_sha256}" manifest
check_hash "${protocol}" "${protocol_sha256}" protocol
check_hash "${activation}" "${activation_sha256}" activation
check_hash "${worker}" "${BE_BB7_WORKER_SHA256}" worker
check_hash "${collector_cli}" "${BE_BB7_CLI_SHA256}" "collector CLI"
check_hash "${backend_module}" "${BE_BB7_BACKEND_SHA256}" backend
check_hash "${rollout_module}" "${BE_BB7_ROLLOUT_SHA256}" rollout
check_hash "${crops_module}" "${BE_BB7_CROPS_SHA256}" crops
check_hash "${benchmarks_module}" "${BE_BB7_BENCHMARKS_SHA256}" benchmarks
check_hash "${score_module}" "${BE_BB7_SCORE_MODULE_SHA256}" "score module"
check_hash "${scorer}" "${BE_BB7_SCORER_SHA256}" scorer
check_hash "${verifier_module}" "${BE_BB7_VERIFIER_MODULE_SHA256}" "verifier module"
check_hash "${verifier}" "${BE_BB7_VERIFIER_CLI_SHA256}" verifier

if [[ -e "${run_dir}" ]]; then
  echo "refusing to reuse Qwen-7B smoke run directory: ${run_dir}" >&2
  exit 2
fi
mkdir -p "${run_dir}"

export PYTHONPATH="${repo_dir}/src"
export BE_CODE_REVISION="${actual_revision}"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache
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
  echo "Qwen-7B smoke requires exactly one visible GPU" >&2
  exit 2
fi
gpu_name=$(jq -r '.name' <<< "${gpu_info}")
if [[ "${gpu_name,,}" != *"${BE_BB7_EXPECTED_GPU_TOKEN,,}"* ]]; then
  echo "Qwen-7B smoke expected ${BE_BB7_EXPECTED_GPU_TOKEN}, got ${gpu_name}" >&2
  exit 2
fi

"${python_bin}" - "${model}" "${model_revision}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=sys.argv[1],
    revision=sys.argv[2],
    local_files_only=True,
)
print("pinned Qwen-7B snapshot is present")
PY

collect_args=(
  -m beyond_entropy collect-qwen
  --manifest "${manifest}"
  --expected-manifest-sha256 "${manifest_sha256}"
  --limit 32
  --output "${rollouts}"
  --resume
  --checkpoint-interval 8
  --shard-count 1
  --shard-index 0
  --model "${model}"
  --model-revision "${model_revision}"
  --scorer screenqa
  --candidate-count 4
  --proposer ug-grid
  --visual-crop-ratio 2.0
  --visual-cost 1.0
  --generation-seeds 0
  --bootstrap-resamples 100
  --bootstrap-seed 20260903
  --scientific-status "endpoint-blind Qwen-7B engineering smoke on opened ScreenQA ranker development; no task endpoint may select hardware"
  --max-new-tokens 32
  --min-pixels 200704
  --max-pixels 602112
  --attention-implementation sdpa
  --system-prompt "You are a helpful assistant."
)

cd "${repo_dir}"
rollout_start=$(date +%s)
"${python_bin}" "${collect_args[@]}"
rollout_first_seconds=$(( $(date +%s) - rollout_start ))
rollouts_sha256=$(sha256sum "${rollouts}" | cut -d ' ' -f 1)
cp "${rollouts%.jsonl}.provenance.json" "${run_dir}/rollouts.first-pass.provenance.json"
rollout_resume_start=$(date +%s)
"${python_bin}" "${collect_args[@]}"
rollout_resume_seconds=$(( $(date +%s) - rollout_resume_start ))
if [[ "$(sha256sum "${rollouts}" | cut -d ' ' -f 1)" != "${rollouts_sha256}" ]]; then
  echo "Qwen-7B smoke rollout resume changed completed bytes" >&2
  exit 2
fi

"${python_bin}" - "${rollouts}" "${rollouts_sha256}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

rollouts = Path(sys.argv[1])
expected = sys.argv[2]
provenance = json.loads(rollouts.with_suffix(".provenance.json").read_text())
records = sum(1 for line in rollouts.open(encoding="utf-8") if line.strip())
actual = hashlib.sha256(rollouts.read_bytes()).hexdigest()
payload = {
    "passed": actual == expected == provenance.get("output_sha256"),
    "records": records,
    "resumed_from_records": provenance.get("resumed_from_records"),
    "rollouts_sha256_before_resume": expected,
    "rollouts_sha256_after_resume": actual,
}
path = rollouts.parent / "resume.audit.json"
temporary = path.with_name(path.name + ".tmp")
with temporary.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
temporary.replace(path)
if not payload["passed"]:
    raise SystemExit("Qwen-7B smoke rollout resume audit failed")
PY

score_args=(
  "${scorer}"
  --manifest "${manifest}"
  --rollouts "${rollouts}"
  --output "${answer_nll}"
  --expected-manifest-sha256 "${manifest_sha256}"
  --expected-rollouts-sha256 "${rollouts_sha256}"
  --shard-count 1
  --shard-index 0
  --checkpoint-interval 8
  --resume
  --model "${model}"
  --model-revision "${model_revision}"
  --device-map cuda:0
  --dtype bfloat16
  --attention-implementation sdpa
  --min-pixels 200704
  --max-pixels 602112
  --system-prompt "You are a helpful assistant."
  --code-revision "${actual_revision}"
  --scientific-status "endpoint-blind Qwen-7B answer-likelihood smoke on opened ScreenQA ranker development; no task endpoint may select hardware"
)
nll_start=$(date +%s)
"${python_bin}" "${score_args[@]}"
nll_first_seconds=$(( $(date +%s) - nll_start ))
nll_sha256=$(sha256sum "${answer_nll}" | cut -d ' ' -f 1)
cp "${answer_nll%.jsonl}.provenance.json" "${run_dir}/answer-nll.first-pass.provenance.json"
nll_resume_start=$(date +%s)
"${python_bin}" "${score_args[@]}"
nll_resume_seconds=$(( $(date +%s) - nll_resume_start ))
if [[ "$(sha256sum "${answer_nll}" | cut -d ' ' -f 1)" != "${nll_sha256}" ]]; then
  echo "Qwen-7B smoke NLL resume changed completed bytes" >&2
  exit 2
fi

"${python_bin}" "${verifier}" \
  --manifest "${manifest}" \
  --rollouts "${rollouts}" \
  --rollout-provenance "${rollouts%.jsonl}.provenance.json" \
  --rollout-resume-audit "${run_dir}/resume.audit.json" \
  --answer-nll "${answer_nll}" \
  --answer-nll-provenance "${answer_nll%.jsonl}.provenance.json" \
  --output "${run_dir}/smoke.complete.json" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --expected-decisions 32 \
  --expected-model "${model}" \
  --expected-model-revision "${model_revision}" \
  --expected-gpu-name "${BE_BB7_EXPECTED_GPU_TOKEN}" \
  --expected-code-revision "${actual_revision}" \
  --rollout-seconds "${rollout_first_seconds}" \
  --rollout-resume-seconds "${rollout_resume_seconds}" \
  --answer-nll-seconds "${nll_first_seconds}" \
  --answer-nll-resume-seconds "${nll_resume_seconds}"

printf 'screenqa_backbone_7b_smoke_complete=%s gpu=%s rollout_seconds=%s nll_seconds=%s\n' \
  "${run_dir}/smoke.complete.json" "${gpu_name}" \
  "${rollout_first_seconds}" "${nll_first_seconds}"
