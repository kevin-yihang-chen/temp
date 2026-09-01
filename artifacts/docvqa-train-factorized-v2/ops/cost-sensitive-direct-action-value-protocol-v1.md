# Cost-sensitive direct action-value protocol v1

Status: frozen on 2026-09-01 after the externally fixed pairwise signed-value
result and its fixed-result decomposition were recorded, but before fitting or
scoring this candidate. Development uses only opened DocVQA data. ScreenQA and
every protected role remain sealed.

## Registered hypothesis

The full-four-action pairwise proposer recovers 68.08% of helpful states, but
its ridge mean-gain head produces a harmful top-225 tail. Replace both the
pairwise ordering and mean regression with one direct, cost-sensitive surrogate
for the sign of each concrete crop's realized net utility. This avoids
multiplying rare-event probabilities and makes large gains and large harms
matter more than the numerous cost-only neutral crops.

This classifier is a policy component, not a standalone novelty claim. The
paper-level contribution remains the counterfactual visual-tool audit, signed
gain/harm decomposition, and independently calibrated acquisition evidence.

## Frozen inputs

- sibling rollouts SHA-256:
  `9109d5c896e59959937c2171e897dd81341237ff51fc5745fb4c56325ddd8fc3`;
- label-free semantic features SHA-256:
  `05b3aaeabeef84993787c0ce676d175ab51a95bae34cabb5e950b0bd4f906686`;
- audited incumbent OOF score rows SHA-256:
  `e82ea5b94f45c7e5c961ec61248b9cca514897a0b2b291e5c9110fd1c4673cc3`;
- incumbent report/model SHA-256:
  `fd17ef0863ea21ad6eae646e08deb936da449f5ee05dda9cb94d6d1687aef888`
  / `ce8e534cfcf5bf4f08e565b4b88112c1768d6b7d9cc0a1eaa8226ffc424b697e`;
- preceding external-pairwise report/score-report/model/score-row SHA-256:
  `4774df95e5cbbff657d4c24edcaeaf32851b8c29e9c2a2ecd2167bb718b27077`
  / `517443986655dd953328d37fb61bc5baaf574b2579dce2fa342aa993f8ba8c6f`
  / `c502bac4d0dfedeee55cdfa0aee2c7521345c4bb9b9e5b7b9e330f6469e4a546`
  / `d19d1c78a161502f59f9470f27d681011fb33916ba20786b268fcb45455b446a`.

Require exactly 3,500 sources, 13,580 decisions, one ANSWER and four ZOOM
siblings per decision, 54,320 ZOOM rows, 1,442 positive-net-utility rows,
52,878 negative-net-utility rows, and zero exactly-zero-utility rows at
`lambda=0.05`. The raw gain counts must remain 1,604 positive, 1,535 negative,
and 51,181 neutral. Semantic storage must be label-free and align all four
action IDs exactly.

No ScreenQA, DocVQA calibration/formal/reserve, or official benchmark
validation outcome may enter fitting or candidate selection.

## Frozen features, target, and weights

For every candidate crop use exactly the existing 46-dimensional
`semantic-context` action vector. It contains only normalized pre-action
question/answer uncertainty features, original-image/question semantic
summaries, candidate-region semantic summaries, and candidate geometry.
Target answer, correctness, gain, post-action answer/entropy, and teacher
likelihood are forbidden at inference and in serialized OOF rows.

For training only, define

`u(x,a) = correct_after(x,a) - correct_before(x) - 0.05 * tool_cost(x,a)`

and binary target `y(x,a) = 1[u(x,a) > 0]`. Every row has raw importance
`|u(x,a)|`. Within each source, divide each raw importance by the sum across
all of that source's training rows. Then multiply all weights by one global
constant so their total equals the number of training rows. Thus every source
has equal total mass while large benefits and severe harms receive more mass
inside that source. No class weight or class resampling is allowed.

## Frozen model and OOF policy

Use five deterministic whole-source folds with seed `20260914`. In each fold:

1. construct weights using only the outer-training sources;
2. fit `StandardScaler` on the 46 training features only;
3. fit L2 `LogisticRegression(C=0.01, solver=liblinear, max_iter=2000,
   random_state=seed+fold)` with the registered source/utility weights;
4. score all four actions for every outer-test decision with the raw linear
   decision function;
5. select the maximum-score action, breaking exact ties toward the
   lexicographically smaller action ID; and
6. use that maximum action score as the decision's call score.

There is no probability calibration, model search, alternate C, PCA, feature
selection, neural layer, early stopping, answer-loss target, factorized head,
pairwise head, or separately fitted stopping model. After OOF scores are
complete, refit the same single model once on all development rows for possible
future deployment serialization. A failed candidate is not deployed.

## Frozen evaluation and advancement

Serialize only identity, source, outer fold, selected action ID, selected raw
score, call flag, and the audited incumbent identity/action/score/call fields.
Choose an outcome-blind complete-tie threshold that matches exactly 225 pooled
candidate calls. Reconstruct and require the exact audited incumbent 225-call
set and pooled metrics.

Report source- and question-balanced utility, raw gain, gain per call, induced
harm, negative-value call mass, helpful-call precision/recall, proposal
helpful-state recovery, action disagreement, and gate disagreement. Use 20,000
iid whole-source percentile bootstrap resamples, seed `20260914`, two-sided
95% intervals, for candidate-minus-incumbent source-balanced utility.

Advance only if every condition holds:

1. candidate utility is at least incumbent utility plus `0.00025`;
2. paired interval lower endpoint is above `-0.0005`;
3. candidate gain per call is strictly higher;
4. candidate induced harm and negative-value call mass are each no greater;
5. candidate helpful-call precision is no lower; and
6. all hash, population, utility-count, semantic alignment, feature dimension,
   fold exclusion, source-mass, global-weight-mass, convergence, OOF coverage,
   finite-score, matched-call, incumbent-reproduction, serialization, and
   no-leakage audits pass.

Failure yields `cost_sensitive_direct_action_value_not_advanced` and keeps
every ScreenQA role sealed. Passing permits only a separate deployment and
ScreenQA calibration freeze; it does not authorize opening formal outcomes.

Every submitted compute task must email `yihangc@connect.hku.hk` for all state
changes. No GitHub push is authorized by this protocol.
