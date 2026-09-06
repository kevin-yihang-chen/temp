#!/usr/bin/env bash
set -euo pipefail

repo=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo}/.slurm-notify-email"
IFS= read -r notify_email < "${mail_file}"
[[ "${notify_email}" == "yihangc@connect.hku.hk" ]] || {
  echo "invalid notification email" >&2; exit 2;
}
[[ -z "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]] || {
  echo "tracked worktree must be clean before submission" >&2; exit 2;
}
[[ ! -e "${repo}/data/factorized-phase-c-v1" ]] || {
  echo "Phase-C allocation already exists" >&2; exit 2;
}
[[ ! -e "${repo}/data/factorized-phase-c-v1.staging" ]] || {
  echo "Phase-C allocation staging already exists" >&2; exit 2;
}

digest() { sha256sum "$1" | cut -d ' ' -f 1; }
revision=$(git -C "${repo}" rev-parse HEAD)
worker="${repo}/scripts/slurm_factorized_phase_c_allocation.sh"
allocator="${repo}/scripts/freeze_factorized_phase_c_data.py"
helpers="${repo}/src/beyond_entropy/phase_c_allocation.py"
reused="${repo}/scripts/freeze_predictability_data.py"
manifest_export="${repo}/src/beyond_entropy/manifest_export.py"
config="${repo}/configs/factorized_phase_c_allocation_v1.json"
exports="ALL,BE_PHASE_C_CODE_REVISION=${revision},BE_PHASE_C_WORKER_SHA256=$(digest "${worker}"),BE_PHASE_C_ALLOCATOR_SHA256=$(digest "${allocator}"),BE_PHASE_C_HELPERS_SHA256=$(digest "${helpers}"),BE_PHASE_C_REUSED_ALLOCATOR_SHA256=$(digest "${reused}"),BE_PHASE_C_MANIFEST_EXPORT_SHA256=$(digest "${manifest_export}"),BE_PHASE_C_CONFIG_SHA256=$(digest "${config}")"

/usr/local/slurm/bin/sbatch \
  --mail-user="${notify_email}" --mail-type=ALL --export="${exports}" "${worker}"
