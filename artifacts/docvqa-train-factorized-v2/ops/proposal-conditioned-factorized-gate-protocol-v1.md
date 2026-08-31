# Proposal-conditioned factorized gate protocol v1

Status: frozen on 2026-09-01 after the unconditional proposal-conditioned
result and diagnosis were recorded, but before fitting or scoring this model.
All development is on opened DocVQA data. ScreenQA calibration, formal,
reserve, untouched, validation, and test roles remain sealed.

## Registered hypothesis and sole change

Keep the registered OOF `loss_only` proposal exactly unchanged. Fit the gate
on those proposed actions, but factorize state correctness from conditional
action effects instead of subtracting two independently class-balanced
unconditional probabilities.

For each decision define:

- `error = 1[correct_before < 0.5]`;
- on error rows only, `rescue = 1[delta_success > 0]`;
- on non-error rows only, `harm = 1[delta_success < 0]`.

No proposal architecture, loss target, proposal fold, crop set, feature, call
budget, comparator, or protected-role boundary may change.

## Frozen inputs

- rollouts SHA-256:
  `9109d5c896e59959937c2171e897dd81341237ff51fc5745fb4c56325ddd8fc3`;
- label-free semantic features SHA-256:
  `05b3aaeabeef84993787c0ce676d175ab51a95bae34cabb5e950b0bd4f906686`;
- loss-proposer OOF predictions SHA-256:
  `d73b976b72101f2815dc89fd9d472ac91b680aa195beb032deef116600db572e`;
- full loss-proposer model container SHA-256:
  `a69a3d1a58e5bbac525035c10b2d76ea9d652b858567ce4191fbec846cf023f3`;
- audited incumbent/decoupled score rows SHA-256:
  `e82ea5b94f45c7e5c961ec61248b9cca514897a0b2b291e5c9110fd1c4673cc3`;
- incumbent factorized model/report SHA-256:
  `ce8e534cfcf5bf4f08e565b4b88112c1768d6b7d9cc0a1eaa8226ffc424b697e`
  / `fd17ef0863ea21ad6eae646e08deb936da449f5ee05dda9cb94d6d1687aef888`;
- preceding unconditional conditioned report/score report/model SHA-256:
  `6b1ea97b1de0cd1ebb61036aa6e026151afa8c2e9999d8ed3d7f5cdb1624c160`
  / `e04e883eaaa0550cd488d370f97cd25960dcb2e64da09d49a4efdac7391edb5d`
  / `d088e5c24c3dbb5202af492374d72992d9908cfdb2bddc390bd1ed431561dad9`.

The population must be exactly 3,500 sources, 13,580 decisions, four crops per
decision, one bound OOF loss-proposed action, and one audited incumbent action
and score. Prediction and comparison score inputs may contain no task outcome.

## Frozen model and OOF construction

Use five whole-source folds with seed `20260907`. Every test source is excluded
from all three gate heads, and every selected action is itself from the bound
loss proposer that excluded its source.

Fit three independent `LogisticRegression` heads with `C=1.0`, L2,
`solver=liblinear`, `max_iter=2000`, and random state `seed + fold`:

1. an error head on the existing 27-dimensional pre-action state context;
2. a rescue head on the existing 46-dimensional hybrid-context-semantic
   selected-action feature, using only error rows;
3. a harm head on the same selected-action schema, using only non-error rows.

Each head has its own fold-local `StandardScaler`. Give every domain, then
source within domain, then row within source equal mass, normalized to the
number of head-training rows. Do not class-balance, tune `C`, select features,
fit a calibration layer, or change label thresholds.

Let `rescue_magnitude` be the source-balanced mean positive `delta_success` on
rescue-training positives. Let `harm_magnitude` be the source-balanced mean
absolute negative `delta_success` on harm-training positives. The score is:

`P(error) * P(rescue | error, proposed action) * rescue_magnitude`

`- (1 - P(error)) * P(harm | correct, proposed action) * harm_magnitude`

`- 0.05 * tool_cost`.

After all OOF scores exist, refit the three heads once on all bound OOF
proposal rows for deployment composition with the frozen full loss proposer.

## Matched comparison and decision

Select exactly 225 calls independently for candidate and incumbent from scores
and identities only, preserving complete ties and preferring fewer calls on an
equal count error. Reproduce the audited incumbent call set and frozen pooled
gain, utility, and call rate before evaluation.

Use the same source- and question-balanced metrics as the preceding gate. The
primary estimand is candidate minus incumbent source-balanced utility with
20,000 iid whole-source percentile resamples, seed `20260907`, and two-sided
95% interval.

Advance only if every clause holds:

1. candidate source-balanced utility is at least incumbent plus `0.00025`;
2. paired 95% lower endpoint is above `-0.0005`;
3. candidate gain per call is strictly higher;
4. candidate induced harm and negative-value call mass are each no greater;
5. candidate helpful-call precision is no lower;
6. every hash, population, schema, source-exclusion, OOF coverage, matched-call,
   incumbent-reproduction, finite-score, and leakage audit passes.

Failure yields `proposal_conditioned_factorized_gate_not_advanced` and keeps
ScreenQA sealed. Passing permits only a separate pre-outcome ScreenQA
calibration freeze. Every Slurm state change emails
`yihangc@connect.hku.hk`. No GitHub push is authorized.
