# ChartQA train high-power replication protocol

## Status and motivation

This protocol is registered after the frozen 1,918-state ChartQA validation
confirmation and before any rollout is collected on the target defined below.
The validation primary did not pass: its transferred factorized stopping policy
had utility `0.003415`, but the 95% state-bootstrap interval
`[-0.000026, 0.007195]` and image-bootstrap interval
`[-0.000131, 0.007249]` narrowly crossed zero. No threshold, feature, candidate,
cost, or primary criterion is changed here.

Using the validation point estimate and interval width only for approximate
normal power planning gives about 4,400 independent decisions for 80% power to
place a two-sided 95% lower endpoint above zero if the effect replicates. The
new target therefore contains 4,500 decisions and exactly one decision per
image. This is a post-near-miss independent replication, not a retroactive
extension of the failed validation primary.

## Frozen target

The source is the official `HuggingFaceM4/ChartQA` train split at revision
`b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5`. All images occurring in either
the 2,500-state development test manifest or the 1,918-state validation
confirmation manifest are excluded by normalized RGB content hash before
sampling. Thirty-one source rows are excluded by this rule.

The target is selected with seed 29 and contains:

- 4,500 states from 4,500 unique images;
- 2,250 human and 2,250 augmented questions;
- zero image overlap with development or validation; and
- manifest SHA-256
  `72db6feaa4bc042e98741a48dd55421c5246c1b48c84b1fd75740d1d072ca621`.

The three pinned source Parquet SHA-256 values are recorded in
`data/chartqa-train-replication-4500/manifest.provenance.json`.

## Frozen rollout and policy

- Qwen2.5-VL-3B-Instruct revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- One answer-now action plus the same four UG-grid crops per state.
- Deterministic generation seed 0, `max_new_tokens=16`, SDPA, and the same
  concise final-answer-only prompt.
- Additive original-image-plus-crop observations and the ChartQA scorer.
- Cost coefficient `lambda=0.05` and 5,000 bootstrap resamples.

The primary deployment model remains byte-identical to the validation freeze:

- model SHA-256
  `5d5c0f781a7141726e786d6ad87b861a6395c489bcf9ad8567a2e9ca825c3330`;
- source report SHA-256
  `1f05ddeef52fa9abced549479cdb8fa386578d12600fb874a964a12a4d927462`;
- absolute call threshold `0.45069723964195885`; and
- uniform-random one-crop expectation when the gate fires.

No target quantile, target label, target outcome, or target-derived threshold is
permitted.

## Primary criterion

The replication succeeds only if the frozen factorized policy has:

- positive mean utility at `lambda=0.05`;
- a 95% state-bootstrap utility lower endpoint above zero;
- a 95% image-bootstrap utility lower endpoint above zero;
- positive accuracy gain; and
- lower tool use than unconditional one-crop and exhaustive entropy policies.

Because there is exactly one state per image, the state and image resampling
units should be equivalent; both are retained as an implementation check.

The same pre-registered fixed-quadrant, source-only action-ranker, and text-only
gate variants are reported as secondary analyses. They cannot alter the primary
criterion. Human and augmented strata are also secondary.

## Completed outcome

The rollout completed all 4,500 states and 22,500 action records under the
frozen protocol. Its SHA-256 is
`f32d8ab8d5ad46ba264de97667540d41022b52aaae8ef3f0ce3a2df939cc36f9`.
The primary policy has utility `0.003633`, state-bootstrap interval
`[0.000700, 0.006689]`, image-bootstrap interval
`[0.000700, 0.006711]`, accuracy gain `0.006833`, and 6.4% tool use. Therefore
every registered primary condition is true and the replication **passes**.

The 2,250 human-question states have utility `0.006844` with interval
`[0.001111, 0.012756]`; the 2,250 augmented-question states have utility
`0.000422` with interval `[-0.000911, 0.001933]`. These secondary strata do not
change the aggregate primary decision, but they localize the confirmed effect
mainly to human questions. The report SHA-256 is
`ba4b3fa6e45da23fa217470a0ca2be5867634208f16665e7abc83a55b8976c30`.
