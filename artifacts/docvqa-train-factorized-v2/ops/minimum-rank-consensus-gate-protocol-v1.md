# Minimum-rank consensus gate protocol v1

Status: frozen on 2026-09-01 after the cost-sensitive direct action-value
result and fixed-result call/action decomposition were recorded, but before
computing or evaluating any consensus score. Development uses only opened
DocVQA data. ScreenQA and every protected role remain sealed.

## Registered hypothesis

The cost-sensitive candidate is the first method in this branch with a positive
source-balanced utility difference, higher gain per call, higher helpful-call
precision, lower negative-call mass, and 70.18% proposal recovery. It fails
because its call set has higher induced harm. Require agreement between its
direct net-utility score and the independently trained incumbent factorized
score, without fitting another model or choosing a mixture weight.

## Frozen inputs

- sibling rollouts SHA-256:
  `9109d5c896e59959937c2171e897dd81341237ff51fc5745fb4c56325ddd8fc3`;
- cost-sensitive report/score-report/model/score-row SHA-256:
  `941181e02e48352ea4f10ca20f9b10b3ed85afa790ba7d25b2138c0be984c464`
  / `5ae3623879a96c46ac5ffce40b4acc0b84b53e30c58dc2fe2d3f8006fbedf83d`
  / `762005d1124ea68ca993f5a58e56317bdec0e026f24751504b8d5e295c3e6bb1`
  / `9512d000ca3c2567fd36711f20eca10619acd390622ccf1f05cc9930145dcaec`;
- audited incumbent score-row SHA-256:
  `e82ea5b94f45c7e5c961ec61248b9cca514897a0b2b291e5c9110fd1c4673cc3`;
- incumbent report/model SHA-256:
  `fd17ef0863ea21ad6eae646e08deb936da449f5ee05dda9cb94d6d1687aef888`
  / `ce8e534cfcf5bf4f08e565b4b88112c1768d6b7d9cc0a1eaa8226ffc424b697e`.

Require exactly 3,500 sources, 13,580 decisions, complete unique identities,
the cost-sensitive and incumbent OOF scores/actions/call flags reproduced from
their bound artifacts, and no outcome-bearing field in either score input.

## Frozen outcome-free consensus score

For a method's finite score vector of length `N=13,580`, define the upper
empirical percentile of score `s` as

`R(s) = count(score_i <= s) / N`.

All equal raw scores receive the same percentile. Compute `R_incumbent` and
`R_cost` independently over their complete OOF vectors. Do not use outcomes,
fold metrics, source identity, action identity, or existing call flags in the
rank transformation.

For every decision define

`consensus_score = min(R_incumbent, R_cost)`.

Retain the frozen cost-sensitive action ID exactly. Select an outcome-blind
complete-tie threshold on `consensus_score` that matches exactly 225 pooled
calls. If no complete-tie threshold yields 225, fail closed. There is no mean,
maximum, product, weighted sum, learned stacker, veto threshold, fallback,
per-fold rank, source-specific rank, or tie splitting. No model is refit.

Serialize only identity, source, retained action, both raw scores, both
percentiles, consensus score, consensus call flag, and the audited incumbent
action/score/call fields. Correctness, answers, gain, harm, reward, target,
post-action entropy, oracle action, and utility are forbidden.

## Frozen evaluation and advancement

Reconstruct the exact incumbent 225-call set and pooled metrics. Report source-
and question-balanced utility, raw gain, gain per call, induced harm,
negative-value call mass, helpful-call precision/recall, proposal helpful-state
recovery, action disagreement, gate disagreement, and raw/percentile score
agreement. Use 20,000 iid whole-source percentile bootstrap resamples, seed
`20260916`, two-sided 95% intervals, for consensus-minus-incumbent
source-balanced utility.

Advance only if every condition holds:

1. consensus utility is at least incumbent utility plus `0.00025`;
2. paired interval lower endpoint is above `-0.0005`;
3. consensus gain per call is strictly higher;
4. consensus induced harm and negative-value call mass are each no greater;
5. consensus helpful-call precision is no lower; and
6. all hash, population, identity, bound-score reproduction, finite-score,
   percentile monotonicity/tie, min-rule, exact-call, incumbent-reproduction,
   serialization, and no-leakage audits pass.

Failure yields `minimum_rank_consensus_gate_not_advanced` and keeps every
ScreenQA role sealed. Passing permits only a separate deployment and ScreenQA
calibration freeze; it does not authorize opening formal outcomes.

If this rule fails, do not evaluate alternate means, maxima, products, weights,
veto cutoffs, fallback rules, call budgets, or action combinations on this
opened DocVQA population. Further method selection must move to a newly frozen
development population; otherwise reposition the paper around the established
mechanism, risk, and negative-results evidence.

Every submitted compute task must email `yihangc@connect.hku.hk` for all state
changes. No GitHub push is authorized by this protocol.
