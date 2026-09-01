# Externally fixed pairwise signed-value protocol v1

Status: frozen on 2026-09-01 after the high-dimensional union result and its
stopping-versus-action decomposition were recorded, but before fitting or
scoring this candidate on DocVQA. Development uses only opened DocVQA data.
ScreenQA and every protected role remain sealed.

## Registered hypothesis

The high-dimensional union increases global helpful-state proposal recovery
but loses primarily through its call set. Replace the product of three
rare-event probabilities with two separately cross-fitted components:

1. a pairwise ranker over all four frozen crop candidates, trained on signed
   within-state gain ordering; and
2. a continuous ridge head predicting the selected crop's signed realized
   gain from an inner-OOF selected action.

The deployment score is predicted signed gain minus the fixed tool cost. No
binary error, rescue, or harm probability is multiplied into this score.

## Frozen inputs

- DocVQA sibling rollouts SHA-256:
  `9109d5c896e59959937c2171e897dd81341237ff51fc5745fb4c56325ddd8fc3`;
- label-free semantic features SHA-256:
  `05b3aaeabeef84993787c0ce676d175ab51a95bae34cabb5e950b0bd4f906686`;
- audited incumbent OOF score rows SHA-256:
  `e82ea5b94f45c7e5c961ec61248b9cca514897a0b2b291e5c9110fd1c4673cc3`;
- incumbent report/model SHA-256:
  `fd17ef0863ea21ad6eae646e08deb936da449f5ee05dda9cb94d6d1687aef888`
  / `ce8e534cfcf5bf4f08e565b4b88112c1768d6b7d9cc0a1eaa8226ffc424b697e`;
- external TextVQA development report/model SHA-256:
  `2479f52890cfe6e0bd324bd8da36d8eaa045ca240c3ddc8a57347c797c839ccb`
  / `e628bb0b5216242e52ab3e5e6a9e738c8a53e8d795ea1fd8bf5d16467ee63fe6`.

The external report contains 5,000 TextVQA development sources and 7,912
decisions, uses no calibration or formal outcome, selects
`semantic-context`, ranker `C=0.01`, and call-value `alpha=100`. These settings
are transferred exactly; they are not candidates on DocVQA.

Require exactly 3,500 DocVQA sources, 13,580 decisions, one ANSWER and four
ZOOM siblings per decision, and label-free semantic action alignment. No
ScreenQA, DocVQA calibration/formal/reserve, or official benchmark validation
outcome may enter fitting or selection.

## Frozen features and action ranker

Use the existing `semantic-context` state and action features exactly. They
contain only the pre-action state, question/original-image semantic summaries,
candidate-region semantic summaries, and candidate geometry. Target answer,
correctness, task gain, post-action answer/entropy, and teacher likelihood are
forbidden at inference and in serialized OOF rows.

The bound real-input preflight yields exactly 60 state features, 46 action
features, and therefore 110 call features (state, selected action, and four
ranker-summary scalars). Any other dimension fails closed.

For every training state, enumerate all unordered pairs of its four crops.
Drop exact signed-gain ties. For every unequal pair, enter both feature
difference directions with complementary labels. Weight rows to give every
source equal total mass. Fit `StandardScaler` and L2 `LogisticRegression` with
`C=0.01`, `solver=liblinear`, `max_iter=2000`, and the fold seed. Rank all four
crops by the resulting linear decision score; break exact ties by
lexicographically smaller action ID.

## Nested source-held-out signed-value head

Use five deterministic whole-source outer folds with seed `20260911`. For each
outer fold:

1. fit five inner whole-source rankers only on the outer-training sources;
2. obtain one inner-OOF selected action for every outer-training decision;
3. fit the outer ranker on all outer-training sources and select one crop for
   each outer-test decision;
4. build call features from the pre-action state, selected-crop features,
   selected ranker score, top-two score gap, score mean, and score standard
   deviation;
5. standardize those call features on outer training only and fit weighted
   `Ridge(alpha=100)` to the inner-OOF selected crop's signed
   `correct_after - correct_before` target; and
6. predict signed gain only for the outer-test selected action.

Inner fold assignment and ranker fitting use seeds `seed + 1000 + outer_fold`
and its fold offsets. The outer ranker uses `seed + 2000 + outer_fold`; the
full-development ranker uses `seed + 3000`. These offsets are deterministic
only and are not candidates.

Outer-test sources must have zero overlap with every outer ranker/scaler and
call-value scaler/head. The selected action used as an outer-training call
target must come from an inner ranker that excluded its source. There is no
hyperparameter search, class balancing, PCA, neural layer, early stopping,
calibration model, answer-loss target, or alternate feature mode.

After OOF scoring completes, refit one ranker on all DocVQA development
sources. Fit its deployment call head on actions selected by a separate
five-fold full-development OOF ranker, never on actions selected in-sample by
the full refit. Serialize both full components even if the development
advancement decision is negative, but do not deploy a failed candidate.

## Frozen evaluation

For each OOF decision, serialize only identity, selected action, predicted
gain, cost-adjusted score, fold, and outcome-free ranker diagnostics. Set
`lambda=0.05`. Choose the candidate threshold solely from its scores to match
exactly 225 pooled calls, preserving complete ties. Reconstruct the audited
incumbent and require its exact 225-call set and pooled metrics.

Report source- and question-balanced utility, gain, gain per call, induced
harm, negative-value call mass, helpful-call precision/recall, proposal
helpful-state recovery, action disagreement, and gate disagreement. Use 20,000
iid whole-source percentile bootstrap resamples, seed `20260911`, two-sided
95% intervals, for candidate-minus-incumbent source-balanced utility.

Advance only if every condition holds:

1. candidate utility is at least incumbent utility plus `0.00025`;
2. paired interval lower endpoint is above `-0.0005`;
3. candidate gain per call is strictly higher;
4. candidate induced harm and negative-value call mass are each no greater;
5. candidate helpful-call precision is no lower; and
6. all hash, population, semantic alignment, pair construction, source
   weighting, inner/outer exclusion, feature dimension, finite-score,
   matched-call, incumbent-reproduction, serialization, and no-leakage audits
   pass.

Failure yields `external_pairwise_signed_value_not_advanced` and keeps every
ScreenQA role sealed. Passing permits only a separate deployment and ScreenQA
calibration freeze; it does not authorize opening formal outcomes directly.

Every submitted compute task must email `yihangc@connect.hku.hk` for all state
changes. No GitHub push is authorized by this protocol.
