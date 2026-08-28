#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 {docvqa|textvqa}" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo_dir}/.slurm-notify-email"
task=$1

case "${task}" in
  docvqa)
    manifest="${repo_dir}/data/cross-benchmark-v1/docvqa-dev/manifest.jsonl"
    expected_count=824
    expected_sha256=873df25b9df1bcff1aa12ad99a352bc7d7cc89ade4a0db02caf1510a3163f862
    run_dir="${repo_dir}/artifacts/cross-benchmark-dev-v1/docvqa/qwen3b-c4-seed0"
    ;;
  textvqa)
    manifest="${repo_dir}/data/cross-benchmark-v1/textvqa-dev/manifest.jsonl"
    expected_count=318
    expected_sha256=bfe1105df2b9f37ed352207a46d519c0a3468a677759ec8039dbbbdec1fd54fa
    run_dir="${repo_dir}/artifacts/cross-benchmark-dev-v1/textvqa/qwen3b-c4-seed0"
    ;;
  *)
    echo "Unsupported development task: ${task}" >&2
    exit 2
    ;;
esac

if [[ ! -r "${mail_file}" ]]; then
  echo "Missing private Slurm email file: ${mail_file}" >&2
  exit 2
fi
IFS= read -r notify_email < "${mail_file}"
if [[ "${notify_email}" != *@* ]]; then
  echo "Invalid notification email in ${mail_file}" >&2
  exit 2
fi
if [[ "$(wc -l < "${manifest}")" -ne "${expected_count}" ]]; then
  echo "Frozen ${task} development manifest count mismatch" >&2
  exit 2
fi
actual_sha256=$(sha256sum "${manifest}")
actual_sha256=${actual_sha256%% *}
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
  echo "Frozen ${task} development manifest SHA-256 mismatch" >&2
  exit 2
fi
code_revision=$(git -C "${repo_dir}" rev-parse HEAD)

exec sbatch \
  --job-name="be-${task}-dev" \
  --mail-user="${notify_email}" \
  --mail-type=ALL \
  --export=ALL,BE_DEV_TASK="${task}",BE_DEV_MANIFEST="${manifest}",BE_DEV_MANIFEST_SHA256="${expected_sha256}",BE_DEV_RUN_DIR="${run_dir}",BE_CODE_REVISION="${code_revision}" \
  "${repo_dir}/scripts/slurm_cross_benchmark_dev.sh"
