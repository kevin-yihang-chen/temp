#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

export PATH=/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/bin
repo=/userhome/cs3/yihangc/Documents/beyond-entropy
worker="${repo}/scripts/slurm_vtool_action_credit_g1_h800.sh"
launcher="${repo}/scripts/run_vtool_action_credit_g1.py"
config="${repo}/configs/vtool_action_credit_g1_v1.json"
runtime=/userhome/cs3/yihangc/Documents/runtime/vtool-action-credit-g1
python_bin=/userhome/cs3/yihangc/anaconda3/envs/beyond-entropy-vtool-g1/bin/python
dataproto_smoke="${repo}/scripts/smoke_vtool_action_credit_dataproto.py"
jq_bin=/userhome/cs3/yihangc/anaconda3/bin/jq
cd "${repo}"
if [[ ! -x "${jq_bin}" ]]; then
  echo "required jq executable is absent: ${jq_bin}" >&2
  exit 2
fi
runtime_audit_relative=$("${jq_bin}" -er '.preflight.full_train_runtime_audit_report' "${config}")
runtime_audit="${repo}/${runtime_audit_relative}"
minimum_free_gib=$("${jq_bin}" -er '.resources.minimum_free_persistent_disk_gib' "${config}")
if [[ ! "${minimum_free_gib}" =~ ^[0-9]+$ || "${minimum_free_gib}" -ne 64 ]]; then
  echo "G1 frozen persistent-disk requirement must be 64 GiB" >&2
  exit 2
fi
minimum_free_kb=$((minimum_free_gib * 1024 * 1024))
available_kb=$(df -Pk "${repo}" | awk 'NR==2 {print $4}')
if [[ -z "${available_kb}" || "${available_kb}" -lt "${minimum_free_kb}" ]]; then
  echo "G1 requires at least ${minimum_free_gib} GiB free persistent disk before submission" >&2
  exit 2
fi

revision=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "repository worktree must be clean before G1 submission" >&2
  exit 2
fi
for path in "${worker}" "${launcher}" "${config}" "${runtime_audit}" "${dataproto_smoke}"; do
  if [[ ! -f "${path}" ]]; then
    echo "required G1 input is absent: ${path}" >&2
    exit 2
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  echo "frozen G1 Python executable is absent: ${python_bin}" >&2
  exit 2
fi
dataproto_report=$(PYTHONPATH="${runtime}:${repo}:${repo}/src" \
  "${python_bin}" "${dataproto_smoke}")
if ! "${jq_bin}" -e '
  .decision == "vtool_action_credit_dataproto_chunk_passed" and
  .chunks == 4 and
  .protected_split_contents_accessed == false and
  .model_weights_loaded == false and
  (.checks | all(.[]; . == true))
' <<< "${dataproto_report}" >/dev/null; then
  echo "G1 DataProto chunk smoke failed" >&2
  exit 2
fi
if /usr/local/slurm/bin/squeue -h -u yihangc -n be-vtool-g1-signed | grep -q .; then
  echo "a paired-signed G1 job is already queued or running" >&2
  exit 2
fi

quota=$(/usr/local/bin/show-cpu-gpu-quota)
gpu_limit=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\1/p' <<< "${quota}")
gpu_used=$(sed -n 's/^GPU Quota in Minutes: \([0-9][0-9]*\), Used: \([0-9][0-9]*\)$/\2/p' <<< "${quota}")
if [[ -z "${gpu_limit}" || -z "${gpu_used}" || $((gpu_limit - gpu_used)) -lt 480 ]]; then
  echo "G1 needs a 480 GPU-minute reserve" >&2
  exit 2
fi

worker_sha256=$(sha256sum "${worker}" | awk '{print $1}')
launcher_sha256=$(sha256sum "${launcher}" | awk '{print $1}')
config_sha256=$(sha256sum "${config}" | awk '{print $1}')
audit_sha256=$(sha256sum "${runtime_audit}" | awk '{print $1}')
submit_epoch=$(date +%s)
args=(
  "${revision}"
  "${worker_sha256}"
  "${launcher_sha256}"
  "${config_sha256}"
  "${audit_sha256}"
  "${runtime_audit}"
  "${submit_epoch}"
)
/usr/local/slurm/bin/sbatch --test-only --export=NONE "${worker}" "${args[@]}" >/dev/null
submission=$(/usr/local/slurm/bin/sbatch --parsable --export=NONE "${worker}" "${args[@]}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "could not parse paired-signed G1 Slurm job ID: ${submission}" >&2
  exit 2
fi
printf 'vtool_action_credit_g1_job_id=%s code_revision=%s gpu_type=h800 gpu_count=4 max_steps=2\n' \
  "${job_id}" "${revision}"
