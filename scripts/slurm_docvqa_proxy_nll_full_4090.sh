#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=be-docvqa-proxy-nll
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-docvqa-proxy-nll-full-4090-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_DOCVQA_PROXY_EXPECTED_CODE_REVISION
  BE_DOCVQA_PROXY_SCORER_SHA256
  BE_DOCVQA_PROXY_SCORE_MODULE_SHA256
  BE_DOCVQA_PROXY_MERGER_SHA256
  BE_DOCVQA_PROXY_ANALYZER_SHA256
  BE_DOCVQA_PROXY_AUDIT_MODULE_SHA256
  BE_DOCVQA_PROXY_WORKER_SHA256
  BE_DOCVQA_PROXY_PROTOCOL_SHA256
  BE_DOCVQA_PROXY_IMPLEMENTATION_SHA256
  BE_DOCVQA_PROXY_ACTIVATION_SHA256
  BE_DOCVQA_PROXY_FULL_RESUME
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done
if [[ "${BE_DOCVQA_PROXY_FULL_RESUME}" != 0 \
  && "${BE_DOCVQA_PROXY_FULL_RESUME}" != 1 ]]; then
  echo "BE_DOCVQA_PROXY_FULL_RESUME must be 0 or 1" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
root="${repo_dir}/artifacts/docvqa-train-factorized-v2/proxy-to-outcome-cross-domain-v1/full-v1"
shard_dir="${root}/score-shards"
merged_dir="${root}/merged"
analysis_dir="${root}/analysis"
manifest="${repo_dir}/data/docvqa-train-factorized-v2/ranker-training/manifest.jsonl"
rollouts="${repo_dir}/artifacts/docvqa-train-factorized-v2/ranker-training/qwen3b-c4-seed0/rollouts.jsonl"
protocol="${repo_dir}/artifacts/docvqa-train-factorized-v2/ops/proxy-to-outcome-cross-domain-protocol-v1.md"
implementation="${repo_dir}/artifacts/docvqa-train-factorized-v2/ops/proxy-to-outcome-cross-domain-implementation-v1.md"
activation="${repo_dir}/artifacts/docvqa-train-factorized-v2/ops/proxy-to-outcome-cross-domain-activation-v1.md"
smoke="${repo_dir}/artifacts/docvqa-train-factorized-v2/proxy-to-outcome-cross-domain-v1/smoke-v1/answer-nll.jsonl"
smoke_provenance="${smoke%.jsonl}.provenance.json"
scorer="${repo_dir}/scripts/score_visual_action_answer_nll.py"
score_module="${repo_dir}/src/beyond_entropy/answer_likelihood.py"
merger="${repo_dir}/scripts/merge_visual_action_answer_nll.py"
analyzer="${repo_dir}/scripts/analyze_visual_action_proxy_outcomes.py"
audit_module="${repo_dir}/src/beyond_entropy/proxy_outcome_audit.py"
worker="${repo_dir}/scripts/slurm_docvqa_proxy_nll_full_4090.sh"
manifest_sha256=871ea5b924badba8e0f23477fbd40d2e77085a5f69726d0c48e674e72d64a25d
rollouts_sha256=9109d5c896e59959937c2171e897dd81341237ff51fc5745fb4c56325ddd8fc3
protocol_sha256=106879da7d15db351a4145e5a06c43fc3f33803182d1ca4e6f08362b076f8cbe
implementation_sha256=43d884c2749539d9d25b05038e7d4897783c1ae38041a00ff70440b42d4e5a73
activation_sha256=251c663720ff00376644bbd4b0423ddd00efc7b126b45e69dda9e7d4c94403a5
smoke_sha256=9aa6e35023a5246b4433f12eb2ee1a3149ce62b066e2e2ec61171d6b2471bc9d
smoke_provenance_sha256=6e036a454f261a348c40dbcdb6342a1d3567559a600cf3d4036b4359254574e3
scorer_sha256=d278b8cd50a58133d6f512467dce8b53a38a690ade3e874b9721c61adabe523d
score_module_sha256=10c2b647b6ebbc036d6ce06b046521476b4f3d26e73e66b63b7d3f32382b51e4
merger_sha256=4e5c8f2a97e9bdfed835f592e6cc9e52138134e4b6d1cfcd855c013b05f5974d
analyzer_sha256=0147a7215ac4956eb908322cce880512e6961ee8ba1cf6ce4321c5084c22e266
audit_module_sha256=7ad2fe4a710e60ca3d1d7f69584c9344c2eee6533e176ddb5e28063b16dae5a4
model_revision=66285546d2b821cf421d4f5eb2576359d3770cd3

check_hash() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "DocVQA full proxy-NLL ${label} hash mismatch" >&2
    exit 2
  fi
}

actual_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_revision}" != "${BE_DOCVQA_PROXY_EXPECTED_CODE_REVISION}" ]]; then
  echo "DocVQA full proxy-NLL code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before DocVQA full proxy-NLL" >&2
  exit 2
fi
check_hash "${manifest}" "${manifest_sha256}" manifest
check_hash "${rollouts}" "${rollouts_sha256}" rollouts
check_hash "${protocol}" "${protocol_sha256}" protocol
check_hash "${implementation}" "${implementation_sha256}" implementation
check_hash "${activation}" "${activation_sha256}" activation
check_hash "${smoke}" "${smoke_sha256}" smoke
check_hash "${smoke_provenance}" "${smoke_provenance_sha256}" "smoke provenance"
check_hash "${scorer}" "${scorer_sha256}" scorer
check_hash "${score_module}" "${score_module_sha256}" "score module"
check_hash "${merger}" "${merger_sha256}" merger
check_hash "${analyzer}" "${analyzer_sha256}" analyzer
check_hash "${audit_module}" "${audit_module_sha256}" "audit module"
check_hash "${worker}" "${BE_DOCVQA_PROXY_WORKER_SHA256}" worker
if [[ "${BE_DOCVQA_PROXY_SCORER_SHA256}" != "${scorer_sha256}" \
  || "${BE_DOCVQA_PROXY_SCORE_MODULE_SHA256}" != "${score_module_sha256}" \
  || "${BE_DOCVQA_PROXY_MERGER_SHA256}" != "${merger_sha256}" \
  || "${BE_DOCVQA_PROXY_ANALYZER_SHA256}" != "${analyzer_sha256}" \
  || "${BE_DOCVQA_PROXY_AUDIT_MODULE_SHA256}" != "${audit_module_sha256}" \
  || "${BE_DOCVQA_PROXY_PROTOCOL_SHA256}" != "${protocol_sha256}" \
  || "${BE_DOCVQA_PROXY_IMPLEMENTATION_SHA256}" != "${implementation_sha256}" \
  || "${BE_DOCVQA_PROXY_ACTIVATION_SHA256}" != "${activation_sha256}" ]]; then
  echo "DocVQA full proxy-NLL submitted contract mismatch" >&2
  exit 2
fi
if [[ -d "${root}" && -n "$(find "${root}" -mindepth 1 -print -quit)" \
  && "${BE_DOCVQA_PROXY_FULL_RESUME}" != 1 ]]; then
  echo "existing DocVQA full outputs require audited resume" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src"
export HF_HOME=/userhome/cs3/yihangc/Data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "${shard_dir}" "${merged_dir}" "${analysis_dir}"
cd "${repo_dir}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Slurm did not expose four CUDA devices" >&2
  exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_gpus[@]}" -ne 4 ]]; then
  echo "expected four CUDA devices, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

pids=()
for shard_index in 0 1 2 3; do
  shard_output="${shard_dir}/answer-nll-shard-${shard_index}-of-4.jsonl"
  shard_log="${shard_dir}/shard-${shard_index}.log"
  CUDA_VISIBLE_DEVICES="${allocated_gpus[${shard_index}]}" \
    "${python_bin}" "${scorer}" \
    --manifest "${manifest}" \
    --rollouts "${rollouts}" \
    --output "${shard_output}" \
    --expected-manifest-sha256 "${manifest_sha256}" \
    --expected-rollouts-sha256 "${rollouts_sha256}" \
    --shard-count 4 \
    --shard-index "${shard_index}" \
    --checkpoint-interval 32 \
    --resume \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --model-revision "${model_revision}" \
    --device-map cuda:0 \
    --dtype bfloat16 \
    --attention-implementation sdpa \
    --min-pixels 200704 \
    --max-pixels 602112 \
    --system-prompt "You are a helpful assistant." \
    --code-revision "${actual_revision}" \
    --scientific-status "frozen retrospective cross-domain replication on opened DocVQA ranker-development data using RTX 4090; no protected-role input and not independent validation" \
    > "${shard_log}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more DocVQA score shards failed; checkpoints retained" >&2
  exit 1
fi
for shard_index in 0 1 2 3; do
  provenance="${shard_dir}/answer-nll-shard-${shard_index}-of-4.provenance.json"
  if [[ "$(jq -r '.decisions' "${provenance}")" -ne 3395 \
    || "$(jq -r '.records' "${provenance}")" -ne 16975 \
    || "$(jq -r '.raw_targets_written' "${provenance}")" != false \
    || "$(jq -r '.measurement_config.accelerator_name' "${provenance}")" != *4090* ]]; then
    echo "DocVQA shard ${shard_index} output contract failed" >&2
    exit 2
  fi
done

merged="${merged_dir}/answer-nll.jsonl"
merge_args=(
  --output "${merged}"
  --expected-shard-count 4
  --expected-decisions 13580
  --expected-records 67900
  --expected-sources 3500
)
for shard_index in 0 1 2 3; do
  merge_args+=(--shard "${shard_dir}/answer-nll-shard-${shard_index}-of-4.jsonl")
done
"${python_bin}" "${merger}" "${merge_args[@]}"
merged_sha256=$(sha256sum "${merged}" | cut -d ' ' -f 1)

"${python_bin}" "${analyzer}" \
  --scores "${merged}" \
  --protocol "${protocol}" \
  --implementation-contract "${implementation}" \
  --output-dir "${analysis_dir}" \
  --expected-scores-sha256 "${merged_sha256}" \
  --expected-protocol-sha256 "${protocol_sha256}" \
  --expected-implementation-contract-sha256 "${implementation_sha256}" \
  --expected-decisions 13580 \
  --expected-sources 3500 \
  --bootstrap-resamples 2000 \
  --bootstrap-seed 20260901 \
  --bootstrap-confidence 0.95 \
  --study-label "DocVQA ranker development" \
  --scientific-status "retrospective cross-domain replication on opened DocVQA ranker-development data; not candidate selection or independent validation" \
  --interpretation-boundary "Only the opened DocVQA ranker-development manifest and sibling outcomes are inputs. Existing DocVQA calibration/formal results and every ScreenQA protected role are excluded. Thresholds are descriptive only and cannot revise either prior candidate branch." \
  --code-revision "${actual_revision}"

report="${analysis_dir}/report.json"
if [[ "$(jq -r '.population.decisions' "${report}")" -ne 13580 \
  || "$(jq -r '.population.zoom_actions' "${report}")" -ne 54320 \
  || "$(jq -r '.study.label' "${report}")" != "DocVQA ranker development" \
  || "$(jq -r '.outcome_use.protected_role_inputs_used' "${report}")" != false \
  || "$(jq -r '.outcome_use.calibration_or_formal_inputs_used' "${report}")" != false ]]; then
  echo "DocVQA proxy-to-outcome report contract failed" >&2
  exit 2
fi
printf 'docvqa_proxy_outcome_report_sha256=%s merged_sha256=%s decisions=13580 zoom_actions=54320\n' \
  "$(sha256sum "${report}" | cut -d ' ' -f 1)" "${merged_sha256}"
