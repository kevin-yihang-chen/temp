# TextVQA factorized fixed-sequence preregistration

Status: frozen before exporting a new calibration manifest or generating any
outcome for the reserve roles defined below. The earlier 3,000-source
calibration outcomes have already been inspected and are permanently excluded
from this branch. The earlier 5,000-source formal allocation remains sealed but
is not the formal role for this post-failure method.

## Scope and provenance

This is a new prospective branch after the scaled pairwise primary failed
independent calibration. It does not revise that result. Development selected
one factorized candidate from the source-balanced 2x2 diagnostic:

- model type: `multidomain_factorized_action_value`;
- training protocol: `source_grouped_oof_domain_source_balanced_v2`;
- feature mode: `hybrid-context-semantic`;
- alpha: `1`;
- folds and seed: five whole-source folds, seed `20260828`;
- cost: `lambda=0.05` for one executed crop;
- raw model SHA-256:
  `2509e844de92f4b37e485ab26328d268318f596da93d80e60c0799351e7e52e9`;
- development report SHA-256:
  `d9bafdb3fc73af00f1691e39c8974fc491594bd05b7b463a8bd01e410fe379ea`;
- development rollouts SHA-256:
  `1c1d5b67010b5ddfbdabe47072291336b34dcc54928e5db7a12727daa4f14c8e`;
- frozen label-free feature SHA-256:
  `93cdfa91b570fcc67f16bdd4e39d59489fa160e26c2797abf16d684f2f44a504`.

The baseline-error head sees only the 27-dimensional pre-action context. Crop
rescue and harm heads use frozen original-image semantic crop features. No
post-action answer, entropy, correctness, target, or crop outcome is an input
at deployment.

## Frozen score thresholds

Refit the already selected heads on all 5,000 development sources exactly as in
the serialized raw model. Apply that serialized model to the development
features without a call threshold. Construct a strict-to-permissive numeric
threshold sequence from the resulting scores only:

1. one floating-point step above the maximum development score;
2. observed score order statistics targeting development call rates
   `0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03`;
3. deduplicate tied numeric thresholds while retaining strict descending order.

This procedure is outcome-free after the model is fixed. No threshold may be
added, deleted, reordered, or numerically changed after the new calibration
manifest is exported.

## New source roles

Reuse the immutable TextVQA train allocation order with namespace
`beyond-entropy-textvqa-train-scale-v1`, seed `20260828`, parent allocation
SHA-256
`da6d41584bf4f3bfb91426fa9fa3bcb61a659846147c279eaab2aedb776e1657`,
and the same decoded-RGB/prior-bank exclusion contract.

| Original hash-rank offsets | Role | Sources | Outcome access |
| --- | --- | ---: | --- |
| 13,000--15,999 | fresh fixed-sequence calibration | 3,000 | risk calibration only |
| 16,000--21,952 | new one-shot formal | 5,953 | only after calibration passes and policy freeze |

Both roles must be source- and decoded-RGB-disjoint from all earlier roles and
prior banks. Only the calibration manifest may be exported initially. The new
formal identities are allocated and audited but its manifest, images,
rollouts, and features remain unmaterialized until the complete successful
policy is frozen.

## Fixed-sequence calibration

For each frozen threshold in strict-to-permissive order, average every loss
within source before testing. Jointly test:

- induced-harm mass at most `0.005`;
- net-negative-call mass at most `0.02`.

Use the implemented bounded-mean Bernoulli-KL lower-tail p-value, family error
`0.05`, and per-risk cutoff `0.05/2 = 0.025`. Continue only while both tests
pass. Stop at the first joint failure and mark every more permissive threshold
untested. Among preceding risk-accepted thresholds, select the most permissive
one satisfying both empirical non-degeneracy conditions:

- source-balanced call rate at least `0.01`;
- source-balanced utility at least `0.001`.

The utility floor is not a finite-sample utility guarantee. If no threshold is
eligible, select answer-now, close this factorized branch, and do not export the
formal manifest.

## One-shot formal decision

If calibration succeeds, freeze and hash the candidate model, selected
threshold, calibration report, source allocation, prompt, collector, scorer,
feature contract, risk implementation, evaluator, and this protocol before
formal export. Evaluate the exact policy once on all 5,953 new formal sources.

The primary estimand is source-balanced utility:

`mean_source mean_question call(x) * (gain(x) - 0.05)`.

Use 20,000 whole-source bootstrap resamples, seed `20260828`, and a two-sided
97.5% percentile interval. The branch passes only if:

- source-balanced utility is positive with a strictly positive 97.5% lower
  endpoint;
- question-weighted utility is positive;
- source-balanced call rate is at least 1%;
- the evaluated threshold is exactly the fixed-sequence calibration choice;
- all artifact and implementation hashes match the policy freeze.

Raw gain, gain per call, induced harm, negative-call mass, unnecessary calls,
correct stopping, random/fixed/entropy baselines, oracle utility, oracle regret,
and crop-ranking rescue are mandatory diagnostics. Failure closes this branch;
no replacement may be selected on the new formal outcomes.

This within-TextVQA confirmation is necessary but not sufficient for a CVPR,
ICCV, or ECCV claim. A second benchmark or base-model family remains required
before making a general multimodal-agent claim.
