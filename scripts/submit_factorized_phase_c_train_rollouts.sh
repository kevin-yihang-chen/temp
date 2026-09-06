#!/usr/bin/env bash
set -euo pipefail

repo=/userhome/cs3/yihangc/Documents/beyond-entropy
mail_file="${repo}/.slurm-notify-email"
allocation_report="${repo}/data/factorized-phase-c-v1/allocation.report.json"
IFS= read -r notify_email < "${mail_file}"
[[ "${notify_email}" == "yihangc@connect.hku.hk" ]] || {
  echo "invalid notification email" >&2; exit 2;
}
[[ -z "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]] || {
  echo "tracked worktree must be clean before submission" >&2; exit 2;
}
[[ -f "${allocation_report}" ]] || { echo "missing Phase-C allocation report" >&2; exit 2; }

requested=("$@")
if [[ "${#requested[@]}" -eq 0 ]]; then
  requested=(chartqa docvqa hrbench)
fi

digest() { sha256sum "$1" | cut -d ' ' -f 1; }
revision=$(git -C "${repo}" rev-parse HEAD)
worker="${repo}/scripts/slurm_factorized_phase_c_train_rollouts.sh"
generator="${repo}/scripts/generate_counterfactual_prefixes.py"
merger_cli="${repo}/scripts/merge_sequential_rollout_shards.py"
merger_module="${repo}/src/beyond_entropy/sequential_rollout_shards.py"
backend="${repo}/src/beyond_entropy/qwen_backend.py"
rollout_module="${repo}/src/beyond_entropy/sequential_rollout.py"
schema="${repo}/src/beyond_entropy/sequential_schema.py"
sharding="${repo}/src/beyond_entropy/sharding.py"
benchmarks="${repo}/src/beyond_entropy/benchmarks.py"
report_sha256=$(digest "${allocation_report}")

for benchmark in "${requested[@]}"; do
  case "${benchmark}" in
    chartqa|docvqa|hrbench) ;;
    *) echo "invalid Phase-C benchmark: ${benchmark}" >&2; exit 2 ;;
  esac
  manifest="${repo}/data/factorized-phase-c-v1/${benchmark}/train/manifest.jsonl"
  expected_manifest_sha256=$(
    jq -er --arg benchmark "${benchmark}" \
      '.benchmarks[$benchmark].train.manifest_sha256' "${allocation_report}"
  )
  [[ "$(digest "${manifest}")" == "${expected_manifest_sha256}" ]] || {
    echo "allocation report manifest hash mismatch for ${benchmark}" >&2; exit 2;
  }
  exports="ALL,BE_PHASE_C_BENCHMARK=${benchmark},BE_PHASE_C_CODE_REVISION=${revision},BE_PHASE_C_MANIFEST_SHA256=${expected_manifest_sha256},BE_PHASE_C_ALLOCATION_REPORT_SHA256=${report_sha256},BE_PHASE_C_WORKER_SHA256=$(digest "${worker}"),BE_PHASE_C_GENERATOR_SHA256=$(digest "${generator}"),BE_PHASE_C_MERGER_CLI_SHA256=$(digest "${merger_cli}"),BE_PHASE_C_MERGER_MODULE_SHA256=$(digest "${merger_module}"),BE_PHASE_C_BACKEND_SHA256=$(digest "${backend}"),BE_PHASE_C_ROLLOUT_MODULE_SHA256=$(digest "${rollout_module}"),BE_PHASE_C_SCHEMA_SHA256=$(digest "${schema}"),BE_PHASE_C_SHARDING_SHA256=$(digest "${sharding}"),BE_PHASE_C_BENCHMARKS_SHA256=$(digest "${benchmarks}")"
  /usr/local/slurm/bin/sbatch \
    --job-name="be-pc-${benchmark}" \
    --mail-user="${notify_email}" --mail-type=ALL \
    --export="${exports}" "${worker}"
done

