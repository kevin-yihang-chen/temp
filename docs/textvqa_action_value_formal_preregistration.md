# TextVQA action-value formal preregistration

Status: frozen before any TextVQA formal sibling outcomes are generated.

## Claim under test

The primary claim is that a pre-action, factorized estimate of baseline error,
crop rescue, and crop harm can improve TextVQA task score while spending less
than the value it creates. This is a test of counterfactual task value, not of
entropy reduction.

## Frozen development artifact

The policy was selected only from the 318-state, 200-source TextVQA development
bank using five source-grouped out-of-fold predictions, then refit once on all
development sources. No formal outcome is an input to this model.

| Artifact | Frozen value |
| --- | --- |
| Development rollouts SHA-256 | `a94c72b1977e86436c6187248f64826a34b791151c52a7c7b73ca89f92b97ddb` |
| OOF report SHA-256 | `2d81ddbcdd6fea2308c4ebe20a3f2ed307846530689d20cbbfec9a436fdd960e` |
| Serialized model SHA-256 | `ca224964aeb429478aeffaa3f084750cab05daf2c56be0b3f70fda68dceadc33` |
| Model training code revision | `5c382cb97eb8ee61b2be90b47dc52ea0aab706b5` |
| Feature mode | `context-geometry` |
| Model family | factorized error/rescue/harm logistic heads |
| OOF folds / seed | 5 / `20260828` |
| Selected regularizer | `alpha=10.0` |
| Frozen call margin | `-0.002274153771013032` |
| Cost | one unit per crop, `lambda=0.05` |

The decision rule selects the crop with maximum predicted

`P(error) * P(rescue | error, crop) * rescue_magnitude`

minus

`P(correct) * P(harm | correct, crop) * harm_magnitude + lambda * cost`,

and calls only when that value is at least the frozen margin. Selection uses no
`answer_after`, `correct_after`, or `entropy_after` field.

## Frozen formal bank

The outcome-unseen formal manifest has 633 states from 400 source images at
`data/cross-benchmark-v1/textvqa-formal/manifest.jsonl`, SHA-256
`847899f91147633186b61a802004c49cfe8ef3258427cb92ea390c891ec5ef2c`.
It has zero state, source, or decoded-RGB overlap with development. Every state
will produce answer-now plus four released-UG grid crop siblings with the same
Qwen revision, prompts, pixel bounds, generation seed, and TextVQA scorer as the
development bank. The collection job records its exact code revision and sends
Slurm mail for every state transition.

## Primary analysis and pass rule

The serialized model is evaluated exactly once with
`scripts/evaluate_frozen_action_value.py`, 10,000 source-cluster bootstrap
resamples, bootstrap seed `20260828`, and the frozen model and formal-rollout
hashes. The primary metric is mean policy utility:

`mean task-score gain - 0.05 * mean tool calls`.

The primary claim passes only if:

1. mean policy utility is strictly positive; and
2. its source-clustered 95% bootstrap confidence interval has a lower bound
   strictly above zero.

A positive point estimate whose interval includes zero is recorded only as a
directional replication. A non-positive point estimate is a failed
confirmation. The policy may not be changed after either result.

## Frozen secondary analyses

Secondary, clearly labeled analyses report task-score gain, tool-use rate,
gain per call, unnecessary-call rate, and correct-stopping rate. The complete
sibling bank is also summarized for answer-now, fixed crop, uniformly random
crop expectation, and action-and-stopping oracle policies. Oracle and random
policies diagnose headroom; they are not competing learned primary methods and
cannot alter the pass decision.

No feature, threshold, regularizer, crop set, cost, bootstrap unit, or pass
criterion may be selected from the formal outcomes.
