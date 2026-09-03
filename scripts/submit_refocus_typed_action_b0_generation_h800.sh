#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
jq_bin=/userhome/cs3/yihangc/anaconda3/bin/jq
worker="${repo}/scripts/slurm_refocus_typed_action_b0_generation_h800.sh"
runner="${repo}/scripts/run_refocus_typed_action_b0_generation.py"
config="${repo}/configs/refocus_typed_action_b0_generation_v1.json"
cd "${repo}"

revision=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "tracked and visible untracked worktree must be clean before B0 submission" >&2
  exit 2
fi
for path in "${worker}" "${runner}" "${config}"; do
  if [[ ! -f "${path}" ]]; then
    echo "required B0 generation input is absent: ${path}" >&2
    exit 2
  fi
done
if [[ ! -x "${jq_bin}" ]]; then
  echo "frozen jq is absent" >&2
  exit 2
fi

dataset_rel=$("${jq_bin}" -er '.data.dataset' "${config}")
converter_report_rel=$("${jq_bin}" -er '.data.converter_report' "${config}")
processor_report_rel=$("${jq_bin}" -er '.data.processor_executor_report' "${config}")
dataset="${repo}/${dataset_rel}"
converter_report="${repo}/${converter_report_rel}"
processor_report="${repo}/${processor_report_rel}"
for path in "${dataset}" "${converter_report}" "${processor_report}"; do
  if [[ ! -f "${path}" ]]; then
    echo "frozen B0 artifact is absent: ${path}" >&2
    exit 2
  fi
done

if ! "${jq_bin}" -e '
  .schema == "refocus_typed_action_b0_generation_protocol_v1" and
  .study_role == "baseline_correctness_only" and
  .uses_reward_target == false and
  .data.row_count == 1 and
  .data.development_split == "b0_smoke" and
  .data.protected_split_contents_accessed == false and
  .sampling.generation_count == 16 and
  (.sampling.seeds | length) == 16 and
  .sampling.n == 1 and
  .resources.partition == "q-h800" and
  .resources.gpu_type == "H800" and
  .resources.gpu_count == 1 and
  .resources.optimizer_steps == 0 and
  .resources.checkpoints_written == 0 and
  .resources.notification_email == "yihangc@connect.hku.hk" and
  .resources.slurm_mail_type == "ALL" and
  .analysis.raw_model_text_execution_allowed == false
' "${config}" >/dev/null; then
  echo "B0 generation protocol contract failed" >&2
  exit 2
fi
if [[ "$(sha256sum "${dataset}" | awk '{print $1}')" != \
  "$("${jq_bin}" -er '.data.dataset_sha256' "${config}")" ]]; then
  echo "B0 dataset SHA-256 mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "${converter_report}" | awk '{print $1}')" != \
  "$("${jq_bin}" -er '.data.converter_report_sha256' "${config}")" ]]; then
  echo "B0 converter report SHA-256 mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "${processor_report}" | awk '{print $1}')" != \
  "$("${jq_bin}" -er '.data.processor_executor_report_sha256' "${config}")" ]]; then
  echo "B0 processor/executor report SHA-256 mismatch" >&2
  exit 2
fi
if ! "${jq_bin}" -e '
  .decision == "refocus_typed_action_b0_real_runtime_smoke_passed" and
  (.checks | all(.[]; . == true)) and
  .model_weights_loaded == false and
  .optimizer_steps == 0 and
  .checkpoints_written == 0 and
  .protected_split_contents_accessed == false
' "${processor_report}" >/dev/null; then
  echo "B0 processor/executor prerequisite did not pass" >&2
  exit 2
fi

quota=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
gpu_used=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
reserve=$("${jq_bin}" -er '.resources.minimum_gpu_quota_reserve_minutes' "${config}")
if [[ -z "${gpu_limit}" || -z "${gpu_used}" || $((gpu_limit - gpu_used)) -lt "${reserve}" ]]; then
  echo "B0 generation lacks the frozen GPU-minute reserve" >&2
  exit 2
fi
minimum_disk_gib=$("${jq_bin}" -er '.resources.minimum_free_persistent_disk_gib' "${config}")
available_kb=$(df -Pk "${repo}" | awk 'NR==2 {print $4}')
if [[ -z "${available_kb}" || "${available_kb}" -lt $((minimum_disk_gib * 1024 * 1024)) ]]; then
  echo "B0 generation lacks the frozen persistent-disk reserve" >&2
  exit 2
fi

worker_sha256=$(sha256sum "${worker}" | awk '{print $1}')
runner_sha256=$(sha256sum "${runner}" | awk '{print $1}')
config_sha256=$(sha256sum "${config}" | awk '{print $1}')
dataset_sha256=$(sha256sum "${dataset}" | awk '{print $1}')
converter_report_sha256=$(sha256sum "${converter_report}" | awk '{print $1}')
processor_report_sha256=$(sha256sum "${processor_report}" | awk '{print $1}')
submit_epoch=$(date +%s)
args=(
  "${revision}"
  "${worker_sha256}"
  "${runner_sha256}"
  "${config_sha256}"
  "${dataset_sha256}"
  "${converter_report_sha256}"
  "${processor_report_sha256}"
  "${submit_epoch}"
)
/usr/local/slurm/bin/sbatch --test-only --export=NONE "${worker}" "${args[@]}" >/dev/null
submission=$(/usr/local/slurm/bin/sbatch --parsable --export=NONE "${worker}" "${args[@]}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse B0 generation Slurm job ID: ${submission}" >&2
  exit 2
fi
printf 'typed_action_b0_job_id=%s code_revision=%s gpu_type=h800 gpu_count=1 checkpoints=0\n' \
  "${job_id}" "${revision}"
