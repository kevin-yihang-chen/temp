# High-dimensional diagonal-bilinear union protocol v1

Status: frozen on 2026-09-01 after the compact dual-union result was recorded,
but before fitting or scoring any high-dimensional model. Development uses only
opened DocVQA data. ScreenQA and every protected role remain sealed.

## Registered hypothesis and unchanged components

Retain the exact bound incumbent/loss two-proposer union, five-fold OOF
factorization, correctness targets, cost, 225-call comparator, metrics, and
advancement rule. Change only the representation available to the three
logistic heads.

The compact model collapses each 2,048-dimensional semantic relationship into
a few cosine/attention scalars. A diagonal bilinear vector preserves which
frozen Qwen dimensions align while remaining linear and strongly regularized.
No Qwen parameter is trained and no new model inference is performed.

## Frozen inputs

- rollouts SHA-256:
  `9109d5c896e59959937c2171e897dd81341237ff51fc5745fb4c56325ddd8fc3`;
- full label-free Qwen semantic feature artifact SHA-256:
  `05b3aaeabeef84993787c0ce676d175ab51a95bae34cabb5e950b0bd4f906686`;
- loss-only OOF predictions SHA-256:
  `d73b976b72101f2815dc89fd9d472ac91b680aa195beb032deef116600db572e`;
- full loss-proposer container SHA-256:
  `a69a3d1a58e5bbac525035c10b2d76ea9d652b858567ce4191fbec846cf023f3`;
- audited incumbent OOF score/action rows SHA-256:
  `e82ea5b94f45c7e5c961ec61248b9cca514897a0b2b291e5c9110fd1c4673cc3`;
- incumbent model/report SHA-256:
  `ce8e534cfcf5bf4f08e565b4b88112c1768d6b7d9cc0a1eaa8226ffc424b697e`
  / `fd17ef0863ea21ad6eae646e08deb936da449f5ee05dda9cb94d6d1687aef888`;
- compact dual-union report/score report/model SHA-256:
  `268323a18f8d826f302629a3b80a65131710182fdb41275be072bb0d247e1797`
  / `1acfafa8f50429dc091b91431fca47f71ac28a482b0c2dcf06404a732a78ec75`
  / `8820ac4528245db3745582882c62cb9aa2cae0fefa7a2f398120e590e18ee3cb`.

Require the same 3,500 sources, 13,580 decisions, four rollout actions, 4,875
equal proposal pairs, 8,705 unequal pairs, and 22,285 union rows. Each semantic
decision must provide finite `question_embedding` `(2048,)`,
`global_visual_embedding` `(2048,)`, and `region_embeddings` `(4,2048)` aligned
exactly to its action IDs.

## Registered features

L2-normalize question `q`, global image `g`, and each candidate region `r`.
Do not center, rotate, reduce, or learn these frozen embeddings before feature
construction.

- Error head: existing 27 pre-action context features followed by the 2,048
  elementwise products `q * g`, for exactly 2,075 features.
- Rescue/harm heads: existing 46 compact hybrid candidate features followed by
  `q * r` and `g * r`, for exactly 4,142 features.

The products are diagonal bilinear sufficient statistics; their learned
logistic coefficients correspond to one regularized diagonal alignment metric.
No raw answer target or post-action representation is allowed.

## Frozen model and OOF

Use five whole-source folds with seed `20260909`. Reuse the exact union
deduplication and equal domain/source/decision/candidate weighting contract.

Fit three independent `LogisticRegression` heads with:

- `C=0.01`, L2, `solver=liblinear`, `max_iter=4000`;
- random state `seed + fold`;
- independent fold-local `StandardScaler`;
- no class balancing, alpha/C search, PCA, feature selection, neural layer,
  calibration model, early stopping, or alternate embedding combination.

Targets, positive magnitudes, factorized candidate score, maximum-proposal
choice, tie break, and full refit are exactly the compact union protocol. The
score remains:

`P(error) * P(rescue | error, candidate) * rescue_magnitude`

`- (1 - P(error)) * P(harm | correct, candidate) * harm_magnitude`

`- 0.05 * tool_cost`.

## Advancement rule

Candidate and incumbent each call exactly 225 decisions using outcome-blind
thresholds. Use 20,000 iid whole-source percentile resamples, seed `20260909`,
and 95% confidence for candidate-minus-incumbent source-balanced utility.

Advance only if all compact-union clauses remain satisfied:

1. utility at least incumbent plus `0.00025`;
2. paired interval lower endpoint above `-0.0005`;
3. gain per call strictly higher;
4. induced harm and negative-value call mass each no greater;
5. helpful-call precision no lower;
6. every input, embedding/action alignment, feature dimension, weighting,
   source-exclusion, OOF, matched-call, incumbent-reproduction, finite-score,
   and leakage audit passes.

Failure yields `highdim_diagonal_bilinear_union_not_advanced` and keeps
ScreenQA sealed. Passing permits only a separate calibration freeze. All Slurm
state changes email `yihangc@connect.hku.hk`. No GitHub push is authorized.
