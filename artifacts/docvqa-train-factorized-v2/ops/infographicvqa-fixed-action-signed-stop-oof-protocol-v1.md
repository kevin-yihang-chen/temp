# InfographicVQA fixed-action signed-value stop OOF protocol v1

Date: 2026-09-02 (Asia/Hong_Kong)

Status: frozen after the privileged stop-factorization diagnostic and before any
OOF model score or policy result is computed. This opened-train experiment can
authorize a later independent calibration protocol, but it is not itself a
formal or deployable result.

## Hypothesis

The action selected by `argmax(question_region_attention)` has a large positive
stopping ceiling, while entropy, attention maximum, and attention margin fail to
rank its signed realized value. A single low-capacity source-held-out linear
classifier can identify a useful positive-value tail without changing the action
selector.

## Bound inputs

```text
9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e  rollouts
009cdbfa1761f9b53b791a50f70b7e98bdf275eec8743d8bcaf078a52ded8ce8  raw-attention features
6eb313cf1bf4e5f61a8decc0c6ef70605009826c1bf4f815deb7f8111ec7bf40  feature completion
27ba5df9d45f9837f685d64589e32740238de6ff0ce46ce54ce6a1ac21a1d471  feature audit
5c8bced0fdad0a4f7c3ad0dca8bf8cf31d40be4c9d2318c6b42ea72d065366ee  raw-attention evaluation
ea38fb7adb024a1c96a6ec160d921687affb3ac0222aecba3f5d422728a4cbf5  raw-attention evaluation completion
f07eddb658444cd11ab67a62b53143c90ebf81a07026f00c7bba1411a3ad8e1a  stop-factorization diagnostic
0160654dd9173192409b434728c3a654c76a275dd55220e6ecd6ab74d50ef068  stop-factorization completion
17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6  bootstrap indices
5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0  bootstrap source order
```

Require exactly 23,946 decisions, 2,204 sources, 4,406 images, 1,023
positive-net fixed-action states, feature revision
`2020b423f7daa6e8b9a942a02308137136bba548`, and model revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`.

Validation, test, reserve, ScreenQA, and every non-train outcome remain sealed.

## Frozen candidate

- The action is always the raw-attention argmax and is never fitted.
- The target for that action is
  `1[delta_success - 0.05 * tool_cost > 0]`.
- Each training row receives absolute realized-net-utility weight, normalized
  so every training source has equal total weight. There is no class weighting.
- The inference vector concatenates exactly:
  `compact_rescue_features(decision, baseline)` (60 values),
  `compact_action_features(decision, selected_index)` (19 values), and
  `log(question_image_attention_mass)` (one value), for 80 values total.
- Standardization is fitted inside each training fold only.
- The only model is L2 logistic regression with `C=0.01`, `liblinear`, maximum
  2,000 iterations, no class balancing, and no feature, model, regularization,
  calibration, threshold, or ensemble search.
- Five deterministic whole-source folds use seed `20260918`. No source may
  appear in both train and held-out portions of a fold. Every decision receives
  exactly one OOF decision-function score.

## Evaluation and decision

Rank OOF scores descending with `(state_id, replicate_id)` as a deterministic
tie break. Compare the learned stop and entropy stop at exactly the same pooled
call count while keeping the raw-attention action fixed.

The sole primary operating point is 2%: 479 calls. The candidate advances to a
separately frozen calibration stage only if all of the following hold:

1. every input, feature dimension, finite-value, convergence, source-exclusion,
   OOF-coverage, fixed-action, matched-call, and no-protected-data audit passes;
2. source-balanced candidate utility has a paired whole-source-bootstrap 95%
   lower endpoint above zero;
3. candidate-minus-entropy source-balanced utility has a paired 95% lower
   endpoint above zero; and
4. candidate positive-net precision is strictly greater than entropy positive-net
   precision at the same 479 calls.

The 0.5%, 1%, 5%, and 10% points are secondary descriptive robustness results.
They cannot rescue a failed 2% primary decision and cannot be selected after
inspection. Reuse the bound 20,000 whole-source bootstrap samples. Report full
utility, gain, harm, precision/recall, source concentration, folds, coefficients,
and outcome-free OOF score rows.

## Interpretation and operations

Passing would show that a fixed-action signed-value stop is learnable OOF on
opened official train; it would not establish final generalization or novelty.
Failure stops this model family on the current opened outcomes: do not tune C,
features, call rate, weights, seed, or classifier family afterward.

Before the full run, perform a real-input smoke that validates bindings,
feature dimension, finite values, class presence, and fold isolation but does
not fit the full model or expose any OOF policy metric. Every Slurm state emails
`yihangc@connect.hku.hk`. No GitHub push is authorized.
