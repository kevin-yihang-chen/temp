#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=be-sequential-smoke
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-sequential-smoke-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_SEQ_BENCHMARK BE_SEQ_ROLE BE_SEQ_LIMIT BE_SEQ_RUN_ROOT
  BE_SEQ_CODE_REVISION BE_SEQ_WORKER_SHA256 BE_SEQ_GENERATOR_SHA256
  BE_SEQ_SCHEMA_SHA256 BE_SEQ_ROLLOUT_SHA256 BE_SEQ_CRITIC_SHA256
  BE_SEQ_BACKEND_SHA256 BE_SEQ_SEMANTIC_SHA256 BE_SEQ_MANIFEST_SHA256
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done

repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
worker="${repo}/scripts/slurm_sequential_smoke.sh"
generator="${repo}/scripts/generate_counterfactual_prefixes.py"
schema="${repo}/src/beyond_entropy/sequential_schema.py"
rollout_module="${repo}/src/beyond_entropy/sequential_rollout.py"
critic="${repo}/src/beyond_entropy/acquisition_critic.py"
backend="${repo}/src/beyond_entropy/qwen_backend.py"
semantic="${repo}/src/beyond_entropy/qwen_semantic.py"
manifest="${repo}/data/predictability-audit-v1/${BE_SEQ_BENCHMARK}/${BE_SEQ_ROLE}/manifest.jsonl"
run_dir="${BE_SEQ_RUN_ROOT}/job-${SLURM_JOB_ID}"
status="${run_dir}/execution.json"

check_hash() {
  local actual
  actual=$(sha256sum "$1")
  actual=${actual%% *}
  if [[ "${actual}" != "$2" ]]; then
    echo "sequential smoke $3 hash mismatch" >&2
    exit 2
  fi
}

if [[ "$(git -C "${repo}" rev-parse HEAD)" != "${BE_SEQ_CODE_REVISION}" ]]; then
  echo "sequential smoke code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before sequential smoke" >&2
  exit 2
fi
check_hash "${worker}" "${BE_SEQ_WORKER_SHA256}" worker
check_hash "${generator}" "${BE_SEQ_GENERATOR_SHA256}" generator
check_hash "${schema}" "${BE_SEQ_SCHEMA_SHA256}" schema
check_hash "${rollout_module}" "${BE_SEQ_ROLLOUT_SHA256}" rollout
check_hash "${critic}" "${BE_SEQ_CRITIC_SHA256}" critic
check_hash "${backend}" "${BE_SEQ_BACKEND_SHA256}" backend
check_hash "${semantic}" "${BE_SEQ_SEMANTIC_SHA256}" semantic
check_hash "${manifest}" "${BE_SEQ_MANIFEST_SHA256}" manifest

if [[ -e "${run_dir}" ]]; then
  echo "refusing to reuse sequential smoke run directory" >&2
  exit 2
fi
mkdir -p "${run_dir}"

finish() {
  local exit_code=$?
  trap - EXIT
  set +e
  "${python_bin}" - "${status}" "${exit_code}" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
path=Path(sys.argv[1])
value={"schema":"sequential_smoke_execution_v1",
       "status":"completed" if int(sys.argv[2]) == 0 else "failed",
       "exit_code":int(sys.argv[2]),"slurm_job_id":os.environ.get("SLURM_JOB_ID"),
       "finished_at_utc":datetime.now(timezone.utc).isoformat()}
temporary=path.with_name(path.name+".tmp")
temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
temporary.replace(path)
PY
  exit "${exit_code}"
}
trap finish EXIT

export PYTHONPATH="${repo}/src"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export BE_CODE_REVISION="${BE_SEQ_CODE_REVISION}"

"${python_bin}" "${generator}" \
  --manifest "${manifest}" \
  --output "${run_dir}/rollouts.jsonl" \
  --features-output "${run_dir}/features.pt" \
  --benchmark "${BE_SEQ_BENCHMARK}" \
  --dataset-role "${BE_SEQ_ROLE}" \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --generation-seed 0 \
  --limit "${BE_SEQ_LIMIT}" \
  --checkpoint-interval 4

