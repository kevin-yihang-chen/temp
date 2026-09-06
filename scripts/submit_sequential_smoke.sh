#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 {rtx_4090|h800|h100} {chartqa|docvqa|hrbench} {train|validation} STATE_COUNT" >&2
  exit 2
fi
gpu=$1
benchmark=$2
role=$3
limit=$4
case "${gpu}" in
  rtx_4090) partition=debug; gres=gpu:rtx_4090:1 ;;
  h800) partition=q-hgpu-small; gres=gpu:h800:1 ;;
  h100) partition=q-hgpu-small; gres=gpu:h100:1 ;;
  *) echo "unsupported GPU" >&2; exit 2 ;;
esac
case "${benchmark}" in chartqa|docvqa|hrbench) ;; *) exit 2 ;; esac
case "${role}" in train|validation) ;; *) exit 2 ;; esac
if [[ ! "${limit}" =~ ^[1-9][0-9]*$ || "${limit}" -gt 512 ]]; then
  echo "bounded development state count must be in [1,512]" >&2
  exit 2
fi

repo=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo}/.slurm-notify-email"
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != "yihangc@connect.hku.hk" ]]; then
  echo "invalid notification email" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before submission" >&2
  exit 2
fi
revision=$(git -C "${repo}" rev-parse HEAD)
worker="${repo}/scripts/slurm_sequential_smoke.sh"
generator="${repo}/scripts/generate_counterfactual_prefixes.py"
schema="${repo}/src/beyond_entropy/sequential_schema.py"
rollout="${repo}/src/beyond_entropy/sequential_rollout.py"
critic="${repo}/src/beyond_entropy/acquisition_critic.py"
backend="${repo}/src/beyond_entropy/qwen_backend.py"
semantic="${repo}/src/beyond_entropy/qwen_semantic.py"
manifest="${repo}/data/predictability-audit-v1/${benchmark}/${role}/manifest.jsonl"
run_root="${repo}/artifacts/sequential-acquisition-v1/smoke-${benchmark}-${role}-${limit}-v1"
if [[ -e "${run_root}" ]]; then
  echo "refusing to reuse smoke root: ${run_root}" >&2
  exit 2
fi
digest() { sha256sum "$1" | cut -d ' ' -f 1; }
exports="ALL,BE_SEQ_BENCHMARK=${benchmark},BE_SEQ_ROLE=${role},BE_SEQ_LIMIT=${limit},BE_SEQ_RUN_ROOT=${run_root},BE_SEQ_CODE_REVISION=${revision},BE_SEQ_WORKER_SHA256=$(digest "${worker}"),BE_SEQ_GENERATOR_SHA256=$(digest "${generator}"),BE_SEQ_SCHEMA_SHA256=$(digest "${schema}"),BE_SEQ_ROLLOUT_SHA256=$(digest "${rollout}"),BE_SEQ_CRITIC_SHA256=$(digest "${critic}"),BE_SEQ_BACKEND_SHA256=$(digest "${backend}"),BE_SEQ_SEMANTIC_SHA256=$(digest "${semantic}"),BE_SEQ_MANIFEST_SHA256=$(digest "${manifest}")"
/usr/local/slurm/bin/sbatch \
  --partition="${partition}" \
  --gres="${gres}" \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export="${exports}" \
  "${worker}"
