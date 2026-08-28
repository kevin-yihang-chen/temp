# Scaled TextVQA risk-controlled acquisition preregistration

Status: frozen on 2026-08-28 while the ranker-training and risk-calibration
rollout jobs are running. Only checkpoint row counts and Slurm state have been
inspected; no risk-calibration rollout content or outcome has been read. The
formal role remains identity-only: no formal manifest or rollout exists.

This protocol defines the primary scaled method, calibration rule, and stop
condition after the failed 200-source TextVQA policy family. It does not revise
or erase any prior formal failure.

## Fixed data roles

- ranker development: 5,000 train sources, 7,912 questions, manifest SHA-256
  `5a93e5279036db874076f0a5109ace91261f2416a48c3d397bc592d7d03c4468`;
- independent risk calibration: 3,000 train sources, 4,712 questions, manifest
  SHA-256
  `423621b83ec3e4103be3ca8782fa659526612a231cc0e911c6231e4a2da747c8`;
- one-shot formal: 5,000 allocated train sources whose manifest remains
  unmaterialized until the complete policy is frozen.

Allocation SHA-256 is
`da6d41584bf4f3bfb91426fa9fa3bcb61a659846147c279eaab2aedb776e1657`.
The roles are source- and decoded-RGB-disjoint and have zero RGB overlap with
21 prior manifests.

## Fixed visual action bank

Use Qwen2.5-VL-3B-Instruct revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`, generation seed 0, the existing
four-action UG grid, crop ratio 2.0, the frozen TextVQA prompt adapter, maximum
32 new tokens, and the same 200,704/602,112 pixel bounds as the fresh formal
experiment. Each decision has one answer-now sibling and four crop siblings.
Deployment cost is `lambda = 0.05` for one executed crop.

## Primary pre-action model

The primary model type is
`source_crossfit_pairwise_ranker_call_value_v1`. It has two separated parts:

1. A pairwise logistic crop ranker learns only within-state differences between
   pre-action candidate features. Unequal-gain action pairs are entered in both
   directions, and source-balanced weights prevent images with more questions
   from dominating.
2. A ridge call-value head predicts the gain of the ranker's selected crop.
   Its training actions come from an inner source-held-out ranker. Thus the
   call head never receives actions selected by a ranker fitted on that same
   source's outcomes.

The feature mode is fixed to `semantic-context` using the same label-free
original-image Qwen representation, multimodal question embedding, and top-four
layer question-to-region attention used in the previous attention diagnostic.
No crop outcome, post-action answer, post-action entropy, correctness label, or
target is a feature. `context-geometry`, `spatial-context-geometry`, the old
factorized model, fixed attention, and random/fixed crops are ablations only;
they cannot replace the primary model after calibration is opened.

Use five deterministic whole-source folds with seed 20260828. Ranker
regularization candidates are `C in {0.01, 0.1, 1.0}`; choose the value with
the largest OOF source-balanced selected-action gain, then top-1 rescue rate,
then smaller `C`. Call-head candidates are `alpha in {1, 10, 100}`; choose the
smallest nested-OOF source-balanced squared error, then larger `alpha`. Refit
the ranker on all ranker-development sources. Fit the final call head on
actions selected by a five-fold OOF ranker, not by the in-sample refit.

The 200, 1,000, 3,000, and 5,000-source learning curves are secondary. They use
prefixes of the already registered source hash order and do not select the
primary full-scale architecture.

## Frozen threshold family and risk calibration

Construct at most 32 unique thresholds from the nested-OOF ranker-development
call scores using deterministic evenly spaced score order statistics. No
threshold may be added after risk-calibration outcomes are read. Answer-now is
always available but is not counted as a successful non-degenerate policy.

On the 3,000 calibration sources, average each loss within source before the
finite-sample test. Use the Bonferroni bounded-mean KL learn-then-test
implementation with family error 0.05 across 32 thresholds and both fixed risk
constraints:

- expected induced-harm mass at most `0.005` per source-balanced decision;
- expected net-negative-call mass at most `0.02` per source-balanced decision.

Require calibration source-balanced call rate at least `0.01` and empirical
source-balanced utility at least `0.001`. Among jointly accepted thresholds,
select the largest source-balanced call rate, breaking ties by larger utility
and then the existing deterministic threshold rule. This utility floor is an
empirical non-degeneracy condition, not a finite-sample utility guarantee.

If no threshold satisfies every condition, select answer-now, declare this
method branch failed, and do not open the formal role for this policy.

## Formal freeze and one-shot test

Before materializing the formal manifest, freeze and hash the ranker, call
head, feature contract, 32-threshold family, calibration report, selected
threshold, cost, prompts, collector revision, evaluator, and this protocol.
No component is refit on calibration or formal outcomes.

The primary estimand is source-balanced utility:

`mean_source mean_question C(x) * (Delta(x) - 0.05)`.

Report a two-sided 97.5% percentile interval from 20,000 whole-source bootstrap
resamples. The policy passes only if all of the following hold on the one-shot
formal bank:

- source-balanced utility has a positive point estimate and strictly positive
  97.5% lower endpoint;
- question-weighted utility is positive;
- source-balanced call rate is at least 1%; and
- the policy is the exact non-degenerate threshold selected on calibration.

Formal induced harm, negative-call mass, raw gain, gain per call, unnecessary
calls, correct stopping, ranker rescue, random/fixed/entropy baselines, oracle
utility, and oracle regret are mandatory diagnostics. The exchangeable-source
risk statement applies to the calibration population; transfer to the formal
source bank is an empirical test and must not be described as a guarantee under
arbitrary distribution shift.
