#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=be-screenqa-proxy-nll
#SBATCH --output=/userhome/cs3/yihangc/Documents/beyond-entropy/slurm-screenqa-proxy-nll-full-4090-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL

set -euo pipefail

required=(
  BE_PROXY_EXPECTED_CODE_REVISION BE_PROXY_SCORER_SHA256
  BE_PROXY_SCORE_MODULE_SHA256 BE_PROXY_MERGER_SHA256
  BE_PROXY_ANALYZER_SHA256 BE_PROXY_AUDIT_MODULE_SHA256
  BE_PROXY_FULL_WORKER_SHA256 BE_PROXY_PROTOCOL_SHA256
  BE_PROXY_IMPLEMENTATION_CONTRACT_SHA256
  BE_PROXY_HARDWARE_PROTOCOL_SHA256 BE_PROXY_HARDWARE_REPORT_SHA256
  BE_PROXY_HARDWARE_COMPLETION_SHA256 BE_PROXY_HARDWARE_ACTIVATION_SHA256
  BE_PROXY_FULL_RESUME
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing ${name}" >&2
    exit 2
  fi
done
if [[ "${BE_PROXY_FULL_RESUME}" != 0 && "${BE_PROXY_FULL_RESUME}" != 1 ]]; then
  echo "BE_PROXY_FULL_RESUME must be 0 or 1" >&2
  exit 2
fi

repo_dir=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
root="${repo_dir}/artifacts/screenqa-train-factorized-v1/proxy-to-outcome-audit-v1/full-v1"
shard_dir="${root}/score-shards"
merged_dir="${root}/merged"
analysis_dir="${root}/analysis"
manifest="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-manifest-v1/manifest.jsonl"
rollouts="${repo_dir}/artifacts/screenqa-train-factorized-v1/ranker-rollouts-v1/merged/rollouts.jsonl"
protocol="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/proxy-to-outcome-audit-protocol-v1.md"
implementation_contract="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/proxy-to-outcome-analysis-implementation-v1.md"
hardware_protocol="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/proxy-nll-hardware-consistency-protocol-v1.md"
hardware_report="${repo_dir}/artifacts/screenqa-train-factorized-v1/proxy-to-outcome-audit-v1/hardware-consistency-v1/report.json"
hardware_completion="${repo_dir}/artifacts/screenqa-train-factorized-v1/proxy-to-outcome-audit-v1/hardware-consistency-v1/audit.complete.json"
hardware_activation="${repo_dir}/artifacts/screenqa-train-factorized-v1/ops/proxy-nll-full-hardware-activation-v1.md"
scorer="${repo_dir}/scripts/score_visual_action_answer_nll.py"
score_module="${repo_dir}/src/beyond_entropy/answer_likelihood.py"
merger="${repo_dir}/scripts/merge_visual_action_answer_nll.py"
analyzer="${repo_dir}/scripts/analyze_visual_action_proxy_outcomes.py"
audit_module="${repo_dir}/src/beyond_entropy/proxy_outcome_audit.py"
worker="${repo_dir}/scripts/slurm_screenqa_proxy_nll_full_4090.sh"
manifest_sha256=a2b6941e2a073b24571d2ccb50960f7c1cd70cb0ce53dc8339c7ec44a47f67ec
rollouts_sha256=0437d2a499adccb1b4e19eb0160583789cee00edf244718ecae9e290108bb8c9
frozen_protocol_sha256=42d952c35ac16f5584ae4fa0f6849920ba0bdc86a3b7cdfd92fb8fdc79c3f129
frozen_implementation_sha256=c497ec89317cbfa6cc7fa2097b8be064c21a29132f95e446b649994ac65c117e
frozen_hardware_protocol_sha256=6402862c1b60bc0a62f58b2389ac05422e20ab84ee74ef627535b4aebb177a0e
frozen_hardware_report_sha256=bb1ba6d1e066086bcaebd1713f6ccee796656892087986bb3a8adae6ffc371a8
frozen_hardware_completion_sha256=ea26bf4898f41baa30e90886f372d594a1b3bdc84770d62647c42b1b1ff6e981
frozen_hardware_activation_sha256=fec80e9c97e054ad195fbd482de697db28a91a8422410c8582d3b0a0e966df4e
frozen_scorer_sha256=d278b8cd50a58133d6f512467dce8b53a38a690ade3e874b9721c61adabe523d
frozen_score_module_sha256=10c2b647b6ebbc036d6ce06b046521476b4f3d26e73e66b63b7d3f32382b51e4
model_revision=66285546d2b821cf421d4f5eb2576359d3770cd3

check_hash() {
  local path=$1
  local expected=$2
  local name=$3
  local actual
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ScreenQA full proxy-NLL ${name} hash mismatch" >&2
    exit 2
  fi
}

actual_revision=$(git -C "${repo_dir}" rev-parse HEAD)
if [[ "${actual_revision}" != "${BE_PROXY_EXPECTED_CODE_REVISION}" ]]; then
  echo "ScreenQA full proxy-NLL code revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree must be clean before ScreenQA full proxy-NLL" >&2
  exit 2
fi
check_hash "${scorer}" "${BE_PROXY_SCORER_SHA256}" "scorer"
check_hash "${score_module}" "${BE_PROXY_SCORE_MODULE_SHA256}" "score module"
check_hash "${merger}" "${BE_PROXY_MERGER_SHA256}" "merger"
check_hash "${analyzer}" "${BE_PROXY_ANALYZER_SHA256}" "analyzer"
check_hash "${audit_module}" "${BE_PROXY_AUDIT_MODULE_SHA256}" "audit module"
check_hash "${worker}" "${BE_PROXY_FULL_WORKER_SHA256}" "worker"
check_hash "${manifest}" "${manifest_sha256}" "manifest"
check_hash "${rollouts}" "${rollouts_sha256}" "rollouts"
check_hash "${protocol}" "${frozen_protocol_sha256}" "frozen protocol"
check_hash "${implementation_contract}" "${frozen_implementation_sha256}" "implementation contract"
check_hash "${hardware_protocol}" "${frozen_hardware_protocol_sha256}" "hardware protocol"
check_hash "${hardware_report}" "${frozen_hardware_report_sha256}" "hardware report"
check_hash "${hardware_completion}" "${frozen_hardware_completion_sha256}" "hardware completion"
check_hash "${hardware_activation}" "${frozen_hardware_activation_sha256}" "hardware activation"
if [[ "${BE_PROXY_SCORER_SHA256}" != "${frozen_scorer_sha256}" \
  || "${BE_PROXY_SCORE_MODULE_SHA256}" != "${frozen_score_module_sha256}" ]]; then
  echo "ScreenQA full proxy-NLL scoring components differ from hardware activation" >&2
  exit 2
fi
if [[ "${BE_PROXY_PROTOCOL_SHA256}" != "${frozen_protocol_sha256}" ]]; then
  echo "ScreenQA full proxy-NLL submitted protocol hash mismatch" >&2
  exit 2
fi
if [[ "${BE_PROXY_IMPLEMENTATION_CONTRACT_SHA256}" != "${frozen_implementation_sha256}" ]]; then
  echo "ScreenQA full proxy-NLL submitted implementation contract hash mismatch" >&2
  exit 2
fi
if [[ "${BE_PROXY_HARDWARE_PROTOCOL_SHA256}" != "${frozen_hardware_protocol_sha256}" \
  || "${BE_PROXY_HARDWARE_REPORT_SHA256}" != "${frozen_hardware_report_sha256}" \
  || "${BE_PROXY_HARDWARE_COMPLETION_SHA256}" != "${frozen_hardware_completion_sha256}" \
  || "${BE_PROXY_HARDWARE_ACTIVATION_SHA256}" != "${frozen_hardware_activation_sha256}" ]]; then
  echo "ScreenQA full proxy-NLL submitted hardware contract hash mismatch" >&2
  exit 2
fi
if [[ "$(jq -r '.hardware_decision.selected' "${hardware_report}")" != rtx_4090 ]]; then
  echo "ScreenQA full proxy-NLL hardware report did not select RTX 4090" >&2
  exit 2
fi
for protected in \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/calibration-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/formal-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/reserve-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/untouched-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/validation-manifest-v1" \
  "${repo_dir}/artifacts/screenqa-train-factorized-v1/test-manifest-v1"; do
  if [[ -e "${protected}" && ( ! -d "${protected}" || -n "$(find "${protected}" -mindepth 1 -print -quit)" ) ]]; then
    echo "ScreenQA protected role opened before full proxy-NLL: ${protected}" >&2
    exit 2
  fi
done
if [[ -d "${root}" && -n "$(find "${root}" -mindepth 1 -print -quit)" && "${BE_PROXY_FULL_RESUME}" != 1 ]]; then
  echo "existing ScreenQA full proxy-NLL outputs require audited resume" >&2
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
  echo "Slurm did not expose CUDA_VISIBLE_DEVICES for the four-GPU allocation" >&2
  exit 2
fi
IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_gpus[@]}" -ne 4 ]]; then
  echo "expected exactly four allocated CUDA devices, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

pids=()
for shard_index in 0 1 2 3; do
  shard_output="${shard_dir}/answer-nll-shard-${shard_index}-of-4.jsonl"
  shard_log="${shard_dir}/shard-${shard_index}.log"
  CUDA_VISIBLE_DEVICES="${allocated_gpus[${shard_index}]}" "${python_bin}" "${scorer}" \
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
    --scientific-status "frozen retrospective proxy-to-outcome audit on opened ScreenQA ranker development bank using hardware-audited RTX 4090; not candidate selection or independent validation" \
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
  echo "one or more ScreenQA proxy-NLL score shards failed; complete checkpoints were retained" >&2
  exit 1
fi
for shard_index in 0 1 2 3; do
  shard_provenance="${shard_dir}/answer-nll-shard-${shard_index}-of-4.provenance.json"
  if [[ "$(jq -r '.measurement_config.accelerator_name' "${shard_provenance}")" != *4090* ]]; then
    echo "ScreenQA full proxy-NLL shard ${shard_index} did not run on RTX 4090" >&2
    exit 2
  fi
done

merged="${merged_dir}/answer-nll.jsonl"
merge_args=(
  --output "${merged}"
  --expected-shard-count 4
  --expected-decisions 14511
  --expected-records 72555
  --expected-sources 1510
)
for shard_index in 0 1 2 3; do
  merge_args+=(--shard "${shard_dir}/answer-nll-shard-${shard_index}-of-4.jsonl")
done
"${python_bin}" "${merger}" "${merge_args[@]}"
merged_sha256=$(sha256sum "${merged}" | cut -d ' ' -f 1)
"${python_bin}" "${analyzer}" \
  --scores "${merged}" \
  --protocol "${protocol}" \
  --implementation-contract "${implementation_contract}" \
  --output-dir "${analysis_dir}" \
  --expected-scores-sha256 "${merged_sha256}" \
  --expected-protocol-sha256 "${frozen_protocol_sha256}" \
  --expected-implementation-contract-sha256 "${frozen_implementation_sha256}" \
  --expected-decisions 14511 \
  --expected-sources 1510 \
  --bootstrap-resamples 2000 \
  --bootstrap-seed 20260831 \
  --bootstrap-confidence 0.95 \
  --code-revision "${actual_revision}"

if [[ "$(jq -r '.population.decisions' "${analysis_dir}/report.json")" -ne 14511 \
  || "$(jq -r '.population.zoom_actions' "${analysis_dir}/report.json")" -ne 58044 \
  || "$(jq -r '.outcome_use.calibration_opened' "${analysis_dir}/report.json")" != false \
  || "$(jq -r '.outcome_use.formal_opened' "${analysis_dir}/report.json")" != false ]]; then
  echo "ScreenQA full proxy-to-outcome audit output contract failed" >&2
  exit 2
fi
printf 'screenqa_proxy_outcome_audit_report_sha256=%s merged_sha256=%s decisions=14511 zoom_actions=58044\n' \
  "$(sha256sum "${analysis_dir}/report.json" | cut -d ' ' -f 1)" \
  "${merged_sha256}"
