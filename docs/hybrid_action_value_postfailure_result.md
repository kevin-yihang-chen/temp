# Hybrid stopping/ranking development result

Date: 2026-08-28

## Scientific status

This is a **development-only, post-failure architecture diagnostic**. The
TextVQA formal outcome had already been inspected before this feature mode was
implemented, so this experiment cannot replace or revise the frozen TextVQA
primary result. No formal outcomes are used to fit or cross-validate either
model below.

## Hypothesis

The earlier `semantic-context` feature mode used semantic embeddings for both
the state-level stopping model and the action-level rescue/harm models. The new
`hybrid-context-semantic` mode deliberately separates those roles:

- stopping/error prediction uses the 27 compact pre-action context features;
- crop rescue/harm ranking uses the 42 question/global/region semantic and
  geometry features.

This tests whether semantic features help choose *where* to look without
destabilizing the decision of *whether* to look.

## Source-grouped OOF result

All hyperparameters and call thresholds are selected using five-fold
source-grouped out-of-fold predictions. Utility is task-score gain minus
`0.05 * tool_calls`. Intervals use 5,000 source-cluster bootstrap resamples.

| Development bank | Mean gain | Tool rate | Mean utility | 95% utility CI | Top-1 rescue on helpful states | Random rescue |
|---|---:|---:|---:|---:|---:|---:|
| TextVQA, 318 decisions / 200 sources | +0.00503 | 4.40% | +0.00283 | [-0.00241, +0.01079] | 32.0% | 41.0% |
| DocVQA, 824 decisions / 200 sources | +0.00737 | 8.01% | +0.00337 | [-0.00312, +0.01048] | 60.7% | 45.5% |

For comparison, context-only OOF utility is +0.00645 on TextVQA and +0.00363
on DocVQA. Semantic-only state/action utility is +0.00409 and +0.00284,
respectively.

## Interpretation

The separation is technically sound but does not produce a robust aggregate
improvement. It preserves the strong DocVQA semantic ranking signal while
reducing TextVQA tool use, yet TextVQA action ranking remains worse than random
and both utility intervals cross zero. Therefore this is an informative
negative ablation, not a candidate formal policy.

The cross-benchmark disagreement indicates that global pooled semantic
embeddings are not a stable proxy for question-conditioned regional evidence.
The next method should learn region ranking with an explicitly within-state
objective and a token-level cross-modal interaction, while keeping stopping
calibration separate.

## Reproducibility

- code revision: `637d1761bdbd513fd9fc91f743c233d7b55b9925`
- TextVQA report SHA-256:
  `1a401775e2694d220f13550788a226452be369395606e8b9d20a977899e64d8d`
- TextVQA model SHA-256:
  `9c58f9104820f746a7a03ff4fb1c679d90a5f254229f88de9414b04d1b82f009`
- DocVQA report SHA-256:
  `9abdc1644e7c3e06783ac421ce5adde42dc4beef5b34879d1c45a2023ed2e75d`
- DocVQA model SHA-256:
  `82ead9a3d09b16ea0c36cc085638ed6e9672cb5ae9a1f98188a49a819478c3d0`

