# Qwen2.5-VL-7B full-run hardware activation v1

Status: activated on 2026-09-01 after the successful endpoint-blind H800 smoke
and before any full 512-state Qwen2.5-VL-7B endpoint was computed.

## Evidence

- Smoke job `199116`: NVIDIA H800, `COMPLETED`, exit `0:0`, zero restarts,
  runtime `00:02:07`, and all-state email enabled.
- Smoke completion SHA-256:
  `e944437165523b4dab5261822abbeb002872f068d0ecd70b7af023688ae64e11`.
- The pinned 7B model loaded in bfloat16 with SDPA and no quantization or
  offload. Exactly 32 states and 160 sibling records completed in each of the
  rollout and likelihood artifacts; byte-stable resume and raw-target
  prohibition passed.
- First-pass rollout/likelihood timing was `67 / 47` seconds, or `114` seconds
  total on one H800. Linear scaling projects `1,824` one-GPU seconds for 512
  states and approximately `456` four-GPU seconds before merge/analysis
  overhead.
- At activation, the account reported `39,471 / 42,000` GPU-minutes used,
  leaving `2,529`. A one-hour four-GPU request reserves at most 240 GPU-minutes,
  more than a tenfold reserve margin over the request.
- Live H800 state showed four free GPUs on `gpucluster-g3` and five on
  `gpucluster-g4`. The sole H100 node had only one free H100, so a same-class
  four-H100 execution was not immediately feasible.

No task endpoint from the smoke was reported or used for this decision.

## Activated hardware

Use exactly four NVIDIA H800 GPUs in `q-h800`, on one node, with 32 CPUs,
384 GiB host memory, and a one-hour limit. This follows the preregistered
preference for an advanced accelerator because:

1. H800 loaded the complete 7B model without compromise;
2. four same-class H800 GPUs were immediately available;
3. measured queue wait was one second and projected runtime is well below the
   one-hour request;
4. the full reserve fits the live quota by more than tenfold;
5. four same-class H100 GPUs were unavailable at activation.

## Pre-execution QOS correction

Immediately before the first full submission on 2026-09-01, the live
association still allowed four generic GPUs and four H800 GPUs, but
`q-hgpu-small` rejected the request before enqueue with
`QOSMaxGRESPerUser`. No job was created and no endpoint was computed. The
dedicated `q-h800` partition exposes the same H800 node class under its
non-debug QOS. A Slurm `--test-only` request for the unchanged four-H800,
32-CPU, 384-GiB, one-hour allocation passed on `gpucluster-g3`; its synthetic
ID `199140` was confirmed absent from the queue. The partition was therefore
corrected to `q-h800` before execution. Model, population, prompt, actions,
measurement, sharding, analysis, quota reserve, and every outcome boundary
remain unchanged.

Every rollout and likelihood process must see exactly one physical H800. Mixed
hardware, model shrinking, quantization, offload, and population reduction are
forbidden. Runtime telemetry must include accelerator name, compute capability,
requested/actual dtype, attention implementation, package versions, wall time,
queue wait, and peak allocated/reserved CUDA memory.

The full orchestration may bind a later clean repository revision only when all
scientific modules retain the hashes in
`backbone-7b-analysis-implementation-v1.md`. The post-smoke scorer/backend
changes solely centralize the already verified measurement configuration and
add provenance-only peak-memory telemetry; they do not change prompts, model
calls, target spans, NLL definitions, actions, or endpoints.

ScreenQA calibration, formal, reserve, untouched, validation, and test roles
remain sealed. All Slurm state changes must email
`yihangc@connect.hku.hk`. No GitHub push is authorized.
