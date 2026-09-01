#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=08:15:00
#SBATCH --job-name=be-infovqa-decar-recover
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-infovqa-decar-recover-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "usage: recovery-worker SCIENTIFIC_REVISION LAUNCHER_REVISION WRAPPER_SHA256 RECOVERY_FREEZE_SHA256 ORIGINAL_WORKER_SHA256 GENERATION_FREEZE_SHA256 PRIOR_JOB_ID SUBMIT_EPOCH" >&2
  exit 2
fi
scientific_revision=$1
launcher_revision=$2
expected_wrapper_sha256=$3
expected_recovery_freeze_sha256=$4
expected_original_worker_sha256=$5
expected_generation_freeze_sha256=$6
prior_job_id=$7
submit_epoch=$8

frozen_scientific_revision=5b1b0211372ccb96ec21fc55fa954d427a5504b5
frozen_original_worker_sha256=4bb26a8977de2f1838b9cd2838cedbe6d11d6b3c9157df3c70deb17dc94acc86
frozen_generation_freeze_sha256=f9ee68799c17c0f5864fac61ae6ea52268017623bb833e2e0e0faf1f4c3f9a0b
frozen_relocated_worker_sha256=8e2bb53dc067e0f81ee3372c3f468e3b56ef76ccbea0a6e4ffc6eda3a642f388
export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
live_repo=/userhome/cs3/yihangc/Documents/beyond-entropy
wrapper="${live_repo}/scripts/slurm_infographicvqa_decar_full_recovery_h800.sh"
recovery_freeze="${live_repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-full-recovery-freeze-v1.md"
original_worker="${live_repo}/scripts/slurm_infographicvqa_decar_full_h800.sh"
generation_freeze="${live_repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-full-generation-freeze-v1.md"
live_manifest_dir="${live_repo}/artifacts/infographicvqa-train-v1/decar-v1/full-manifest-v1"
live_root="${live_repo}/artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1"

sha() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  if [[ ! -f "${path}" || "$(sha "${path}")" != "${expected}" ]]; then
    echo "DECAR recovery ${label} SHA-256 mismatch" >&2
    exit 2
  fi
}

if [[ "${scientific_revision}" != "${frozen_scientific_revision}" ]]; then
  echo "DECAR recovery scientific revision is not the frozen generation revision" >&2
  exit 2
fi
if [[ "${expected_original_worker_sha256}" != "${frozen_original_worker_sha256}" \
  || "${expected_generation_freeze_sha256}" != "${frozen_generation_freeze_sha256}" ]]; then
  echo "DECAR recovery original generation binding changed" >&2
  exit 2
fi
if [[ ! "${launcher_revision}" =~ ^[0-9a-f]{40}$ \
  || ! "${prior_job_id}" =~ ^[0-9]+$ \
  || ! "${submit_epoch}" =~ ^[0-9]+$ ]]; then
  echo "DECAR recovery received malformed launcher metadata" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" || ! "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "DECAR recovery must run under Slurm" >&2
  exit 2
fi

cd "${live_repo}"
if [[ "$(git rev-parse HEAD)" != "${launcher_revision}" ]]; then
  echo "DECAR recovery launcher revision changed after submission" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "DECAR recovery tracked launcher worktree must be clean" >&2
  exit 2
fi
require_hash "${wrapper}" "${expected_wrapper_sha256}" recovery-worker
require_hash "${recovery_freeze}" "${expected_recovery_freeze_sha256}" recovery-freeze
require_hash "${original_worker}" "${expected_original_worker_sha256}" original-worker
require_hash "${generation_freeze}" "${expected_generation_freeze_sha256}" generation-freeze
git cat-file -e "${scientific_revision}^{commit}"
if [[ ! -f "${live_manifest_dir}/task-manifest.jsonl" \
  || ! -f "${live_manifest_dir}/image-manifest.jsonl" \
  || ! -f "${live_manifest_dir}/complete.json" ]]; then
  echo "DECAR recovery live manifest is incomplete" >&2
  exit 2
fi
if [[ ! -d "${live_root}" || -z "$(find "${live_root}" -mindepth 1 -print -quit)" ]]; then
  echo "DECAR recovery found no checkpointed full output" >&2
  exit 2
fi
if [[ -e "${live_root}/execution/job-${prior_job_id}.json" ]]; then
  echo "DECAR recovery refuses a prior job with a completed execution record" >&2
  exit 2
fi

prior_record=$(/usr/local/slurm/bin/scontrol show job -o "${prior_job_id}")
prior_state=$(sed -n 's/.*JobState=\([^ ]*\).*/\1/p' <<< "${prior_record}" | head -n 1)
prior_exit_code=$(sed -n 's/.*ExitCode=\([^ ]*\).*/\1/p' <<< "${prior_record}" | head -n 1)
case "${prior_state}" in
  TIMEOUT|NODE_FAIL|PREEMPTED|FAILED|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|REVOKED|CANCELLED*) ;;
  *)
    echo "DECAR recovery requires a terminal unsuccessful prior job, got ${prior_state:-unknown}" >&2
    exit 2
    ;;
esac

scratch_base=$(mktemp -d /tmp/be-infovqa-decar-recovery.XXXXXX)
scratch_repo="${scratch_base}/repo"
cleanup() {
  if [[ -d "${scratch_repo}" ]]; then
    git -C "${live_repo}" worktree remove --force "${scratch_repo}" >/dev/null 2>&1 || true
  fi
  rm -rf -- "${scratch_base}"
}
trap cleanup EXIT INT TERM

recovery_start_epoch=$(date +%s)
queue_wait_seconds=$((recovery_start_epoch - submit_epoch))
if [[ "${queue_wait_seconds}" -lt 0 ]]; then
  echo "DECAR recovery submit epoch is in the future" >&2
  exit 2
fi

git -C "${live_repo}" worktree add --detach "${scratch_repo}" "${scientific_revision}"
if [[ "$(git -C "${scratch_repo}" rev-parse HEAD)" != "${scientific_revision}" \
  || -n "$(git -C "${scratch_repo}" status --porcelain --untracked-files=no)" ]]; then
  echo "DECAR recovery isolated scientific worktree audit failed" >&2
  exit 2
fi
scratch_parent="${scratch_repo}/artifacts/infographicvqa-train-v1/decar-v1"
mkdir -p "${scratch_parent}"
ln -s "${live_manifest_dir}" "${scratch_parent}/full-manifest-v1"
ln -s "${live_root}" "${scratch_parent}/full-qwen7b-v1"

scratch_worker="${scratch_repo}/scripts/slurm_infographicvqa_decar_full_h800.sh"
scratch_freeze="${scratch_repo}/artifacts/docvqa-train-factorized-v2/ops/infographicvqa-decar-full-generation-freeze-v1.md"
require_hash "${scratch_worker}" "${expected_original_worker_sha256}" isolated-original-worker
require_hash "${scratch_freeze}" "${expected_generation_freeze_sha256}" isolated-generation-freeze
runtime_worker="${scratch_base}/scientific-worker-relocated.sh"
awk '
  $0 == "repo=/userhome/cs3/yihangc/Documents/beyond-entropy" {
    print "repo=${BE_RECOVERY_SCIENTIFIC_REPO}"
    replacements += 1
    next
  }
  { print }
  END { if (replacements != 1) exit 42 }
' "${scratch_worker}" > "${runtime_worker}"
chmod 700 "${runtime_worker}"
require_hash "${runtime_worker}" "${frozen_relocated_worker_sha256}" relocated-original-worker
export BE_RECOVERY_SCIENTIFIC_REPO="${scratch_repo}"

echo "DECAR full recovery start: $(date --iso-8601=seconds)"
echo "Slurm recovery job: ${SLURM_JOB_ID}"
echo "prior terminal job: ${prior_job_id} state=${prior_state} exit=${prior_exit_code}"
echo "scientific revision: ${scientific_revision}"
echo "launcher revision: ${launcher_revision}"

bash "${runtime_worker}" \
  "${scientific_revision}" "${expected_original_worker_sha256}" \
  "${expected_generation_freeze_sha256}" 1 "${submit_epoch}"

execution="${live_root}/execution/job-${SLURM_JOB_ID}.json"
if [[ ! -f "${execution}" ]] || ! jq -e \
  --arg revision "${scientific_revision}" \
  '.code_revision == $revision and .population.questions == 23946 and
   .predictions_computed == false and .validation_or_test_inputs_used == false' \
  "${execution}" >/dev/null; then
  echo "DECAR recovery scientific execution record failed" >&2
  exit 2
fi

recovery_end_epoch=$(date +%s)
recovery_record="${live_root}/execution/job-${SLURM_JOB_ID}.recovery.json"
jq -n \
  --arg schema infographicvqa_decar_full_generation_recovery_v1 \
  --arg recovery_job_id "${SLURM_JOB_ID}" --arg prior_job_id "${prior_job_id}" \
  --arg prior_state "${prior_state}" --arg prior_exit_code "${prior_exit_code}" \
  --arg scientific_revision "${scientific_revision}" \
  --arg launcher_revision "${launcher_revision}" \
  --arg wrapper_sha256 "${expected_wrapper_sha256}" \
  --arg recovery_freeze_sha256 "${expected_recovery_freeze_sha256}" \
  --arg original_worker_sha256 "${expected_original_worker_sha256}" \
  --arg relocated_worker_sha256 "${frozen_relocated_worker_sha256}" \
  --arg generation_freeze_sha256 "${expected_generation_freeze_sha256}" \
  --arg execution_sha256 "$(sha "${execution}")" \
  --argjson queue_wait_seconds "${queue_wait_seconds}" \
  --argjson recovery_seconds "$((recovery_end_epoch - recovery_start_epoch))" \
  '{schema:$schema,recovery_job_id:$recovery_job_id,prior_job_id:$prior_job_id,
    prior_terminal_state:$prior_state,prior_exit_code:$prior_exit_code,
    scientific_revision:$scientific_revision,launcher_revision:$launcher_revision,
    resume_mode:true,isolated_scientific_worktree:true,
    wrapper_sha256:$wrapper_sha256,recovery_freeze_sha256:$recovery_freeze_sha256,
    original_worker_sha256:$original_worker_sha256,
    relocated_worker_sha256:$relocated_worker_sha256,
    generation_freeze_sha256:$generation_freeze_sha256,
    scientific_execution_sha256:$execution_sha256,
    queue_wait_seconds:$queue_wait_seconds,recovery_seconds:$recovery_seconds,
    validation_or_test_inputs_used:false}' > "${recovery_record}.tmp"
mv "${recovery_record}.tmp" "${recovery_record}"
echo "DECAR full recovery end: $(date --iso-8601=seconds)"
printf 'infographicvqa_decar_full_recovery_complete=%s recovery_sha256=%s\n' \
  "${recovery_record}" "$(sha "${recovery_record}")"
