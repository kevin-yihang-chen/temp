# Externally fixed pairwise signed-value result v1

Status: completed on 2026-09-01 under the frozen opened-DocVQA development
protocol. The mechanical decision is
**`external_pairwise_signed_value_not_advanced`**. ScreenQA and every protected
role remain sealed.

## Bound execution

- Slurm job `199873`, one NVIDIA H800, 12 CPUs, 96 GiB, completed in
  `00:01:09`, exit `0:0`, zero restarts, with all-state email enabled.
- Code revision:
  `a1d37681e24128ff3b45d88ef145876af775e199`.
- 3,500 sources, 13,580 decisions, and 54,320 full-four-action rows.
- Five outer whole-source folds, five inner folds per outer-training
  population, 20,000 whole-source bootstrap resamples, seed `20260911`.
- Report SHA-256:
  `4774df95e5cbbff657d4c24edcaeaf32851b8c29e9c2a2ecd2167bb718b27077`.
- Score-report SHA-256:
  `517443986655dd953328d37fb61bc5baaf574b2579dce2fa342aa993f8ba8c6f`.
- Outcome-free score rows SHA-256:
  `d19d1c78a161502f59f9470f27d681011fb33916ba20786b268fcb45455b446a`.
- Model SHA-256:
  `c502bac4d0dfedeee55cdfa0aee2c7521345c4bb9b9e5b7b9e330f6469e4a546`.
- Completion SHA-256:
  `58456cda1b638c63addfeb40d5ec2030b2f3cda706a299bd839c4029a15ea969`.

The run used the externally fixed TextVQA settings `semantic-context`,
ranker `C=0.01`, and call-value `alpha=100`. No DocVQA hyperparameter or
feature-mode search occurred.

## Registered result

Both methods execute exactly 225 pooled calls. Source-balanced metrics are:

| Metric | Incumbent | External pairwise signed value |
|---|---:|---:|
| Utility | 0.00317324 | 0.00164063 |
| Raw gain | 0.00397661 | 0.00243654 |
| Gain per call | 0.247496 | 0.153067 |
| Helpful-call precision | 0.394407 | 0.393659 |
| Induced harm | 0.000341563 | 0.00113534 |
| Negative-value call mass | 0.00977110 | 0.00969944 |
| Helpful-state proposal recovery | 0.516410 | 0.680827 |

The candidate-minus-incumbent source-balanced utility difference is
`-0.00153260`, with 95% interval `[-0.00272904, -0.000492265]`. Five
performance clauses fail; all input, dimension, inner/outer source exclusion,
OOF coverage, convergence, matched-call, incumbent-reproduction, finite-score,
and no-leakage audits pass.

The proposal result is useful but insufficient. Full-four-action recovery
reaches 68.08%, the best deployable OOF proposal in this branch, while the
top-225 call set produces 3.3 times the incumbent induced harm. A ridge mean
target therefore does not order the sparse positive-utility tail safely.

## Post-hoc fixed-result decomposition

A deterministic 20,000-source-bootstrap decomposition with seed `20260913`
changed no score, action, or threshold. It is descriptive and does not replace
the registered result.

| Composition | Source-balanced utility | Difference from incumbent (95% CI) |
|---|---:|---:|
| External call set + incumbent action | 0.00262909 | -0.000544143 [-0.00152850, 0.000311233] |
| Incumbent call set + external action | 0.00266306 | -0.000510179 [-0.00126399, 0.00000908809] |
| Incumbent call set + oracle of the two actions | 0.00328610 | +0.000112859 [0.0000105820, 0.000287992] |

Both stopping and action choice contribute. The external call set becomes much
less harmful when paired with the incumbent action, but remains below the
incumbent. The external action is also weaker within the incumbent call set.
Even an oracle between only those two actions improves by `0.000112859`, below
the registered `0.00025` margin.

## Consequence

Do not tune Ridge alpha, pairwise C, or the semantic feature mode on this
opened result. The next candidate should train one cost-sensitive action-value
surrogate directly over all four candidate rows:

- target the sign of realized net utility `delta_success - 0.05`;
- weight each row by absolute realized utility while normalizing every source
  to equal total mass;
- score all four actions with one source-held-out model, choose the maximum,
  and apply the same outcome-blind 225-call comparison;
- use a single fixed regularization value and no search.

This is an engineering response to the observed tail-ranking failure, not a
standalone novelty claim. Existing selective VQA, learning-to-defer, adaptive
visual-acquisition, uncertainty-guidance, and RL tool-use work prevents such a
generic classifier from being presented as new by itself. The paper-level
claim must remain the counterfactual tool-benefit audit, signed gain/harm
decomposition, and independently calibrated acquisition evidence.

No GitHub push is authorized by this result.
