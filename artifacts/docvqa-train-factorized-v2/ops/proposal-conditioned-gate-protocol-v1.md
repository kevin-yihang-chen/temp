# Proposal-conditioned rescue/harm gate protocol v1

Status: frozen on 2026-09-01 after the naive decoupled gate result was opened,
but before fitting or scoring the proposal-conditioned gate. All work is on
opened DocVQA development data; ScreenQA calibration, formal, reserve, and
untouched roles remain sealed.

## Single registered change

Keep the registered `loss_only` OOF crop proposal unchanged. Replace only the
misaligned incumbent gate with a gate trained on the actions that this proposer
actually emits. Do not change proposal features, teacher target, MLP, seed,
folds, epoch count, or full refit.

## Frozen inputs

- rollouts SHA-256:
  `9109d5c896e59959937c2171e897dd81341237ff51fc5745fb4c56325ddd8fc3`;
- label-free semantic features SHA-256:
  `05b3aaeabeef84993787c0ce676d175ab51a95bae34cabb5e950b0bd4f906686`;
- loss-proposer OOF predictions SHA-256:
  `d73b976b72101f2815dc89fd9d472ac91b680aa195beb032deef116600db572e`;
- full loss-proposer model container SHA-256:
  `a69a3d1a58e5bbac525035c10b2d76ea9d652b858567ce4191fbec846cf023f3`;
- audited incumbent/decoupled OOF score rows SHA-256:
  `e82ea5b94f45c7e5c961ec61248b9cca514897a0b2b291e5c9110fd1c4673cc3`;
- incumbent factorized model/report SHA-256:
  `ce8e534cfcf5bf4f08e565b4b88112c1768d6b7d9cc0a1eaa8226ffc424b697e`
  / `fd17ef0863ea21ad6eae646e08deb936da449f5ee05dda9cb94d6d1687aef888`.

Inputs must contain exactly 3,500 sources, 13,580 decisions, four crops per
decision, one audited loss-proposed action, and one reproduced incumbent action
and score. Input prediction/score rows may not contain task outcomes.

## Sole gate model

For each decision, construct the existing 46-dimensional
`hybrid-context-semantic` feature vector for its OOF loss-proposed crop. Define:

- `rescue = 1[delta_success > 0]`;
- `harm = 1[delta_success < 0]`.

Fit two independent `LogisticRegression` heads:

- `C=1.0`, L2, `solver=liblinear`, `max_iter=2000`;
- five whole-source folds, seed `20260906`, head random state `seed + fold`;
- fold-local `StandardScaler`;
- equal source then equal row weights;
- class-balanced sample weights within each head, with total positive and
  negative mass each equal to one half while preserving relative source mass;
- no alpha grid, feature search, calibration model, neural layer, or early
  stopping.

The gate score is

`P(rescue | state, proposed action) - P(harm | state, proposed action) - 0.05`.

Every OOF score must come from heads that excluded the decision's entire
source. Refit the two heads once on all OOF proposed-action rows only after OOF
scores are complete. The full refit composes with the already frozen full
loss-proposer refit for deployment.

## Matched-budget comparison

Use exactly 225 calls for both the proposal-conditioned candidate and the
audited incumbent. Select thresholds independently from scores and identities
only, preserving ties and minimizing call-count error; break ties toward fewer
calls and then the higher threshold. No task outcome may enter either
threshold.

Report source- and question-balanced gain, utility, call rate, gain per call,
induced harm, negative-value call mass, helpful-call precision/recall, proposal
recovery, action disagreement, and gate disagreement. The primary paired
estimand is proposal-conditioned minus incumbent source-balanced utility with
20,000 whole-source percentile resamples, seed `20260906`, and two-sided 95%
interval.

## Mechanical advancement rule

Advance to candidate serialization only if all conditions hold:

1. source-balanced utility is at least incumbent plus `0.00025`;
2. paired 95% lower endpoint is above `-0.0005`;
3. gain per call is strictly higher than incumbent;
4. induced harm and negative-value call mass are each no greater than
   incumbent;
5. helpful-call precision is no lower than incumbent;
6. all source-exclusion, input-hash, feature-schema, class-balance,
   matched-call, incumbent-reproduction, and score-leakage audits pass.

Failure yields `proposal_conditioned_gate_not_advanced` and ScreenQA remains
sealed. Passing permits a separate freeze of the composed full refits and a
finite ScreenQA calibration threshold sequence; it is not itself independent
validation.

Every submitted task uses all-state email to `yihangc@connect.hku.hk`. No
GitHub push is authorized.
