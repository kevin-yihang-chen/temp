# DocVQA action-value formal preregistration

Status: frozen before any DocVQA formal-v2 sibling outcomes are generated.

## Claim under test

The primary claim is that a pre-action, factorized estimate of baseline error,
crop rescue, and crop harm can improve DocVQA ANLS while spending less than the
value it creates. The target is counterfactual task value, not entropy change.

## Frozen development artifact

The policy was selected only from the 824-state, 200-source DocVQA development
bank using five source-grouped out-of-fold predictions, then refit once on all
development sources. The validated semantic and spatial feature ablations were
weaker, so the compact context-by-geometry model is primary.

| Artifact | Frozen value |
| --- | --- |
| Development rollouts SHA-256 | `4d3d3a33f644d1f5122aabecd47a8168d2dce2db5014692b508ba76ae4ddbe52` |
| OOF report SHA-256 | `0307851cf4597dcab7299cd716127523289efbe1fd73c0199f43920ceada0aae` |
| Serialized model SHA-256 | `33f2e0b1fd29e52c878bbbf2cd9819cd3c7e65e12afbabbdc5fa1f6687c8496b` |
| Model training code revision | `fcb7ad2a4e45d359921d0dde34fe75039b53beae` |
| Feature mode | `context-geometry` |
| Model family | factorized error/rescue/harm logistic heads |
| OOF folds / seed | 5 / `20260828` |
| Selected regularizer | `alpha=10.0` |
| Frozen call margin | `0.012630662805226643` |
| Cost | one unit per crop, `lambda=0.05` |

The decision rule selects the crop with maximum predicted

`P(error) * P(rescue | error, crop) * rescue_magnitude`

minus

`P(correct) * P(harm | correct, crop) * harm_magnitude + lambda * cost`,

and calls only when that value is at least the frozen margin. Selection uses no
post-action answer, correctness, score, or entropy field.

## Frozen formal bank

The outcome-unseen formal-v2 manifest has 1,608 states from 400 source images at
`data/cross-benchmark-v1/docvqa-formal-v2/manifest.jsonl`, SHA-256
`9ceb28d05df5feecedf6cf61fbbb27ce281b94dd027e5d6d6da43ddc091081ac`.
It excludes the one decoded-RGB collision found in the original candidate pool
and has zero state, source, or decoded-RGB overlap with development. Every state
will produce answer-now plus four released-UG grid crop siblings under the same
Qwen revision, prompts, pixel bounds, seed, and official-compatible ANLS scorer
as development. The job records its exact code revision and sends Slurm mail
for every state transition.

## Primary analysis and pass rule

The serialized model is evaluated exactly once with
`scripts/evaluate_frozen_action_value.py`, 10,000 source-cluster bootstrap
resamples, bootstrap seed `20260828`, and exact model and rollout hashes. The
primary metric is

`mean ANLS gain - 0.05 * mean tool calls`.

The primary claim passes only if mean policy utility is strictly positive and
its source-clustered 95% bootstrap confidence interval has a lower bound above
zero. A positive point estimate with an interval including zero is only a
directional replication; a non-positive estimate is a failed confirmation.
The policy may not be modified after either result.

## Frozen secondary analyses

Secondary analyses report ANLS gain, tool rate, gain per call, unnecessary-call
rate, and correct-stopping rate. The complete sibling bank is summarized for
answer-now, fixed crop, uniform-random crop expectation, and action-and-stopping
oracle. The fixed exhaustive-entropy baseline observes all four candidates,
selects the lowest post-action entropy, and is charged all four visual costs.
These baselines diagnose tool headroom but cannot alter the pass decision.

No feature, threshold, regularizer, crop set, cost, bootstrap unit, or pass
criterion may be selected from formal outcomes.
