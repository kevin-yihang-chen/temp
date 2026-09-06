#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=be-sequential-critic
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-sequential-critic-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL
#SBATCH --no-requeue

set -euo pipefail

required=(
  BE_SEQ_BENCHMARK BE_SEQ_CONFIG BE_SEQ_TRAIN_FEATURES BE_SEQ_VALIDATION_FEATURES
  BE_SEQ_VALIDATION_ROLLOUTS BE_SEQ_RUN_ROOT BE_SEQ_CODE_REVISION
  BE_SEQ_WORKER_SHA256 BE_SEQ_TRAINER_SHA256 BE_SEQ_EVALUATOR_SHA256
  BE_SEQ_CRITIC_MODULE_SHA256 BE_SEQ_METRICS_SHA256 BE_SEQ_POLICY_SHA256
  BE_SEQ_SCHEMA_SHA256 BE_SEQ_CONFIG_SHA256 BE_SEQ_TRAIN_FEATURES_SHA256
  BE_SEQ_VALIDATION_FEATURES_SHA256 BE_SEQ_VALIDATION_ROLLOUTS_SHA256
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done

repo=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
worker="${repo}/scripts/slurm_sequential_critic_pilot.sh"
trainer="${repo}/scripts/train_acquisition_critic.py"
evaluator="${repo}/scripts/eval_sequential_policy.py"
critic_module="${repo}/src/beyond_entropy/acquisition_critic.py"
metrics_module="${repo}/src/beyond_entropy/sequential_metrics.py"
policy_module="${repo}/src/beyond_entropy/stopping_policy.py"
schema_module="${repo}/src/beyond_entropy/sequential_schema.py"
run_dir="${BE_SEQ_RUN_ROOT}/job-${SLURM_JOB_ID}"
status="${run_dir}/execution.json"

check_hash() {
  local actual
  actual=$(sha256sum "$1")
  actual=${actual%% *}
  if [[ "${actual}" != "$2" ]]; then
    echo "sequential critic $3 hash mismatch" >&2
    exit 2
  fi
}

if [[ "$(git -C "${repo}" rev-parse HEAD)" != "${BE_SEQ_CODE_REVISION}" ]]; then
  echo "sequential critic code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before sequential critic fit" >&2
  exit 2
fi
check_hash "${worker}" "${BE_SEQ_WORKER_SHA256}" worker
check_hash "${trainer}" "${BE_SEQ_TRAINER_SHA256}" trainer
check_hash "${evaluator}" "${BE_SEQ_EVALUATOR_SHA256}" evaluator
check_hash "${critic_module}" "${BE_SEQ_CRITIC_MODULE_SHA256}" critic-module
check_hash "${metrics_module}" "${BE_SEQ_METRICS_SHA256}" metrics-module
check_hash "${policy_module}" "${BE_SEQ_POLICY_SHA256}" policy-module
check_hash "${schema_module}" "${BE_SEQ_SCHEMA_SHA256}" schema-module
check_hash "${BE_SEQ_CONFIG}" "${BE_SEQ_CONFIG_SHA256}" config
check_hash "${BE_SEQ_TRAIN_FEATURES}" "${BE_SEQ_TRAIN_FEATURES_SHA256}" train-features
check_hash "${BE_SEQ_VALIDATION_FEATURES}" "${BE_SEQ_VALIDATION_FEATURES_SHA256}" validation-features
check_hash "${BE_SEQ_VALIDATION_ROLLOUTS}" "${BE_SEQ_VALIDATION_ROLLOUTS_SHA256}" validation-rollouts

if [[ -e "${run_dir}" ]]; then
  echo "refusing to reuse sequential critic run directory" >&2
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
value={"schema":"sequential_critic_execution_v1",
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
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"${python_bin}" "${trainer}" \
  --config "${BE_SEQ_CONFIG}" \
  --train-features "${BE_SEQ_TRAIN_FEATURES}" \
  --validation-features "${BE_SEQ_VALIDATION_FEATURES}" \
  --output "${run_dir}/critic"

"${python_bin}" "${evaluator}" \
  --config "${BE_SEQ_CONFIG}" \
  --features "${BE_SEQ_VALIDATION_FEATURES}" \
  --rollouts "${BE_SEQ_VALIDATION_ROLLOUTS}" \
  --critics "${run_dir}/critic/critics.pt" \
  --dataset-role validation \
  --output "${run_dir}/evaluation"

