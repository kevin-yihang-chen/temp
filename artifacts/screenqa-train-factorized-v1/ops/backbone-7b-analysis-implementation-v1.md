# Qwen2.5-VL-7B backbone analysis implementation v1

Status: frozen on 2026-09-01 after the endpoint-blind 32-state H800 smoke and
before any full 512-state Qwen2.5-VL-7B report or decision existed. Smoke task
endpoints were neither reported nor used to define this implementation.

This document resolves implementation details of
`backbone-7b-diagnostic-protocol-v1.md` without changing its population,
estimands, or four-condition decision.

## Bound implementation

- implementation revision:
  `40d84d7a29ade63c37952245bbee7b05369bc11a`;
- rollout CLI SHA-256:
  `6512131e7a9bbe55b65f9229a044df43e0fa9c4564e4c20fca060a2a17059346`;
- Qwen backend SHA-256:
  `5ee063fb3d8abe3461186e7185960afd002848f1f31aad7b1fdbc1fc53840acb`;
- rollout merger module/CLI SHA-256:
  `b480e939017774dcd5dab483eeb5864425b046468dbe2356d006408063d347b5` /
  `5ddd3fcbff9d21f036c75efa8591ab70e3cd9a311e7bd6d679dafcb251061744`;
- answer-likelihood module/CLI SHA-256:
  `afcf8ec83e513d855532bf64b7ecc61911a21776b005220d4ec2f8a64e18f470` /
  `230e1cf2d8e264d9092c0b1c390dbd29029049635911455757d52f3ad9062be4`;
- likelihood merger CLI SHA-256:
  `4e5c8f2a97e9bdfed835f592e6cc9e52138134e4b6d1cfcd855c013b05f5974d`;
- proxy analysis module/CLI SHA-256:
  `7ad2fe4a710e60ca3d1d7f69584c9344c2eee6533e176ddb5e28063b16dae5a4` /
  `0147a7215ac4956eb908322cce880512e6961ee8ba1cf6ce4321c5084c22e266`.

## Four-GPU collection and merge

- Deterministically shard the frozen 512-state manifest by the existing
  SHA-256 state-ID sharding algorithm across indices 0 through 3.
- Each shard must contain exactly the state set assigned by that algorithm,
  complete one-ANSWER/four-ZOOM sibling groups, atomic checkpoints, and an
  exact-prefix resume audit.
- Merge rollouts into parent-manifest order and require exactly 512 states and
  2,560 records. The merge must reject missing, duplicate, cross-shard, or
  configuration-inconsistent states.
- Score the merged rollout bank on four deterministic decision-aligned shards,
  then strictly merge exactly 512 decisions, 2,560 score rows, 512 source
  groups, and 2,048 ZOOM actions.
- Every rollout and likelihood shard must report a common H800 accelerator
  class, compute capability, bfloat16 parameter dtype, SDPA implementation,
  package/runtime versions, and positive peak allocated/reserved CUDA memory.
  Peak telemetry is provenance only and is excluded from the numerical score
  configuration hash.

## Estimands and uncertainty

- Use the existing proxy-analysis definitions for answer-loss gap, entropy
  reduction, signed task gain, `lambda=0.05` utility, top-one tie breaking,
  exact uniform random, oracle, rescue, harm, regret, and disagreement.
- The descriptive call-rate grid is emitted for completeness but selects no
  rate or threshold and is excluded from the backbone decision.
- Use exactly 5,000 iid whole-source percentile bootstrap resamples with seed
  `20260903`, two-sided confidence `0.95`, and all 512 one-state source units.
- Require 5,000 valid resamples for every metric consumed by the decision.

## Reporting boundary

Emit JSON, Markdown, and a completion record binding their hashes. The study
label is `ScreenQA Qwen2.5-VL-7B opened development`. The report must declare
opened ranker development and must record candidate search, calibration/formal,
reserve/validation/test, and protected-role use as false. No raw target may be
written. The analysis cannot revise a prior candidate, select a deployment
parameter, or open a sealed role.
