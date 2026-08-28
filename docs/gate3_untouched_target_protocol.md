# Gate 3 untouched-target protocol

## Status

This protocol is registered before any project-model rollout or per-example
answer inspection on the target benchmark. Public dataset metadata, the paper,
the official evaluation code, and aggregate results reported by the benchmark
authors were inspected only to choose the target and define the scorer.

The purpose is deliberately narrow: test whether the already-frozen
**when-to-call** gate transfers to a harder, independently sourced chart QA
benchmark. It does not test a learned crop selector and it cannot establish a
where-to-look claim.

## Primary target and frozen revisions

The primary target is ChartQAPro:

- paper: 1,948 questions over 1,341 charts from 157 sources, including
  infographics and dashboards;
- dataset: `ahmed-masry/ChartQAPro` revision
  `e27c2874825874d6767d2bbc538ed4f0dc2c64c2`;
- official code: `vis-nlp/ChartQAPro` revision
  `4b422c658270aff1d3105fd0fb39b1dd5de9f08c`; and
- primary scorer: byte-semantic parity with the released official evaluator and
  the pinned VLMEvalKit adapter; and
- scorer sensitivity: the paper-specified exact-match rule for Fact Checking
  and Multi Choice.

The sensitivity is registered because both released implementations compute an
`always_use_exact_match` category flag but fail to pass it to the scoring
helper. Consequently their actual behavior applies ANLS unless a `Year` flag
forces exact matching. Released-code parity remains primary for benchmark
comparability; the paper-specified correction is always reported alongside it
and cannot replace the primary after outcomes are inspected.

ChartQAPro is preferred over the public VTool `Refocus_Chart` test because the
latter has exact decoded-RGB-plus-question overlap with the project's ChartQA
development data. It is preferred as the primary over ChartMuseum because its
official scorer is deterministic and does not require an LLM judge.

ChartMuseum is reserved as a secondary robustness target at dataset revision
`462d46deb187d8a40c5a9de4e69e14f1df982e58` and code revision
`c3feaea144fecae71508add5570222dfc83ede6b`. Its result cannot replace the
ChartQAPro primary because the official evaluation uses an LLM judge.

## Pre-outcome identity audit

Before running the VLM, decode every target image to normalized RGB and compute
the existing image and normalized-question hashes. Audit against every image in
the ChartQA development, validation-confirmation, and train-replication
manifests, plus the pinned VTool test audit source.

- Exclude an entire target image group if its RGB hash occurs in any prior
  project split, even when the target question differs.
- Report exact image, question, and joint-key overlaps, duplicates, invalid
  images, and exclusions in a provenance artifact.
- Index, filename, source URL, and fuzzy text are not acceptable identity
  evidence.
- Abort rather than silently repair a malformed or ambiguous identity join.

This audit may read images and question strings but must not read project-model
predictions, correctness, or per-action target outcomes.

## Frozen pilot/formal split

After overlap exclusions, group all remaining rows by normalized RGB hash.
Rank groups by
`sha256("chartqapro-gate3-pilot-v1\0" + rgb_sha256)` and assign the first 200
image groups to the compatibility pilot. Every other image group is the formal
target. This makes the split deterministic, question-count agnostic, and
strictly image-disjoint.

The pilot may be used only to:

- verify image decoding, conversation serialization, and official scoring;
- make the final-answer format compatible with MCQ, conversational,
  hypothetical, and unanswerable questions; and
- measure runtime and memory for Slurm sizing.

It may not be used to refit the stopping model, scaler, regularization,
threshold, cost, crop geometry, or primary criterion. Any prompt or scorer
compatibility change must be frozen in a new provenance record before the first
formal-target rollout. Pilot outcomes are never pooled with the formal result.

## Frozen model, actions, and gate

- Qwen2.5-VL-3B-Instruct revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Deterministic seed 0, SDPA, and a prompt frozen after compatibility testing.
- One answer-now action plus the existing four UG-grid crops.
- Original image retained alongside every crop observation.
- Official ChartQAPro answer scorer, mapped to a per-question score in `[0, 1]`.
- Cost coefficient `lambda=0.05` and 5,000 paired bootstrap resamples.

The primary policy is the byte-frozen factorized context gate:

- model SHA-256
  `5d5c0f781a7141726e786d6ad87b861a6395c489bcf9ad8567a2e9ca825c3330`;
- absolute call threshold `0.45069723964195885`; and
- uniform one-crop expectation across the four frozen siblings when it calls.

The gate has only `ANSWER` and `CALL_VISUAL_TOOL` outputs. It must keep
`spatial_action_id=None`; using a learned refocus program or learned crop ranker
would change the estimand and is forbidden in this protocol.

## Primary estimand and criterion

For question `i`, define

`utility_i = score_i(policy) - score_i(answer_now) - 0.05 * call_i`.

The policy score for a call is the exact mean official score of the four crop
siblings. This removes arbitrary crop-seed variance while retaining the cost of
one deployed call. The primary result is the mean paired utility on the formal
target.

The untouched-target confirmation passes only if the transferred frozen gate
has all of the following:

- positive mean utility;
- a 95% question-bootstrap utility lower endpoint above zero;
- a 95% image-cluster-bootstrap utility lower endpoint above zero;
- positive mean official-score gain before cost; and
- lower tool use than unconditional one-crop and exhaustive four-crop policies.

Answer-only, unconditional uniform one-crop, exhaustive entropy search, the
source-frozen entropy gate, and oracle value are reported under the same formal
rollouts. Question-type strata, call calibration, transition counts, and cost
frontiers are secondary and cannot change the primary decision.

## Decision boundary

A pass would establish that the project's central stopping result transfers
beyond the ChartQA family split used to construct it, and would justify a
bounded VTool Stage A when-to-call evaluation. It still would not validate
spatial action selection or localized action-token credit.

A failure leaves the existing ChartQA high-power stopping replication intact
but limits the paper claim to in-family generalization. The formal target must
not be reused to tune a replacement primary. Representation or calibration
changes require a new development target and another untouched confirmation.

High-cost RL and any spatial-action advantage remain on hold until this
when-to-call transfer is resolved and a spatial selector independently beats
matched random or fixed crops.
