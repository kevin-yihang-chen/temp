# Dual-proposer union factorized gate protocol v1

Status: frozen on 2026-09-01 after the proposal-conditioned factorized result
and union-coverage diagnostic were recorded, but before fitting or scoring this
candidate. All method development uses opened DocVQA data. ScreenQA and every
protected role remain sealed.

## Registered hypothesis

The incumbent and loss-only OOF proposers have complementary candidate recall.
For each decision, construct the set containing their two proposed action IDs
and remove an exact duplicate. Train one factorized candidate scorer only on
this one-or-two-action set. At inference, score the same two proposals, choose
the higher score with lexicographic action-ID tie break, and use its score for
the stop/call tail.

This changes neither underlying proposer. It does not reopen all four crops,
use target-answer loss at inference, inspect protected data, or tune a call
rate.

## Frozen inputs

- rollouts SHA-256:
  `9109d5c896e59959937c2171e897dd81341237ff51fc5745fb4c56325ddd8fc3`;
- label-free semantic features SHA-256:
  `05b3aaeabeef84993787c0ce676d175ab51a95bae34cabb5e950b0bd4f906686`;
- loss-only OOF predictions SHA-256:
  `d73b976b72101f2815dc89fd9d472ac91b680aa195beb032deef116600db572e`;
- full loss-proposer model container SHA-256:
  `a69a3d1a58e5bbac525035c10b2d76ea9d652b858567ce4191fbec846cf023f3`;
- audited OOF incumbent action/score rows SHA-256:
  `e82ea5b94f45c7e5c961ec61248b9cca514897a0b2b291e5c9110fd1c4673cc3`;
- incumbent factorized model/report SHA-256:
  `ce8e534cfcf5bf4f08e565b4b88112c1768d6b7d9cc0a1eaa8226ffc424b697e`
  / `fd17ef0863ea21ad6eae646e08deb936da449f5ee05dda9cb94d6d1687aef888`;
- preceding factorized-conditioned report/score report/model SHA-256:
  `32c4181a0149c9a245c676974e7c0792b79287e5e8f8b16b55070950532be539`
  / `c9777479eb5cca8d4b4aa127e456acaea95eba719bea156cd68d0bfc0950fea7`
  / `8beecd917dec7b75d91ceff4aae8aa37bc9b5cef6a7e7bf243647f74630b266d`.

Require exactly 3,500 sources, 13,580 decisions, four rollout crops per
decision, one loss proposal, one incumbent proposal, 4,875 equal proposal
pairs, 8,705 unequal pairs, and 22,285 unique union rows. Proposal and audited
score inputs may contain no task outcome.

## Frozen OOF model

Use five whole-source folds with seed `20260908`. Every gate training row must
exclude the test source, and both proposal identities are already bound OOF
outputs excluding their own source.

Fit three independent `LogisticRegression` heads with `C=1.0`, L2,
`solver=liblinear`, `max_iter=2000`, and random state `seed + fold`:

1. error head on one 27-dimensional pre-action state vector per decision,
   target `1[correct_before < 0.5]`;
2. rescue head on the 46-dimensional hybrid-context-semantic features of every
   unique union candidate from error decisions, target
   `1[delta_success > 0]`;
3. harm head on the same candidate schema from non-error decisions, target
   `1[delta_success < 0]`.

Each head uses its own fold-local `StandardScaler`. Error weights give equal
domain/source/decision mass. Candidate-head weights give equal domain, source,
decision, then unique candidate mass, normalized to their head row count. Do
not class-balance, tune `C`, select features, calibrate probabilities, add
layers, or change correctness thresholds.

Compute positive rescue and harm magnitudes with the corresponding registered
candidate weights. For each candidate, score:

`P(error) * P(rescue | error, candidate) * rescue_magnitude`

`- (1 - P(error)) * P(harm | correct, candidate) * harm_magnitude`

`- 0.05 * tool_cost`.

Choose the maximum-scoring proposal. After OOF completion, refit once on all
bound union rows for deployment composition with both frozen full proposers.

## Matched comparison and advancement

Candidate and incumbent each call exactly 225 decisions, using scores and
identities only with complete ties preserved. The incumbent call set and
frozen pooled gain, utility, and call rate must reproduce before evaluation.

Report the existing source- and question-balanced gain, utility, call rate,
gain per call, harm, negative-value call mass, helpful precision/recall,
proposal recovery, action disagreement, and gate disagreement. The primary
candidate-minus-incumbent source-balanced utility interval uses 20,000 iid
whole-source percentile resamples, seed `20260908`, and 95% confidence.

Advance only if all hold:

1. utility is at least incumbent plus `0.00025`;
2. paired interval lower endpoint is above `-0.0005`;
3. gain per call is strictly higher;
4. induced harm and negative-value call mass are each no greater;
5. helpful-call precision is no lower;
6. all input, union cardinality, feature, weighting, source-exclusion, OOF,
   matched-call, incumbent-reproduction, finite-score, and leakage audits pass.

Failure yields `dual_proposer_union_factorized_gate_not_advanced` and keeps
ScreenQA sealed. Passing permits only a separately frozen ScreenQA calibration
sequence. Every Slurm state change emails `yihangc@connect.hku.hk`. No GitHub
push is authorized.
