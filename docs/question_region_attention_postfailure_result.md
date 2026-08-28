# Question-conditioned region attention result

Date: 2026-08-28

## Scientific status

This is a development-only, post-failure architecture study. The TextVQA
formal result was already known, while no DocVQA formal outcome had been
inspected when these models and hashes were frozen. The DocVQA result therefore
selects a secondary frozen policy but cannot revise the original context-only
primary policy.

## Intervention

The frozen Qwen baseline is replayed on only the original image and question.
Question-token attention from the final four language layers is averaged over
layers, heads, and exact question tokens, restored to the visual grid, and ROI
pooled over the four released-UG candidate boxes. No candidate crop is executed
to construct this feature. The same pass also provides an original-image-
conditioned question representation.

At deployment these signals should be harvested from the baseline prompt
forward pass. The current offline replay isolates the signal without changing
the frozen answers; its compute overhead must be reported separately and is not
silently counted as zero wall-clock cost.

## Zero-shot ranking

Intervals use 5,000 source-cluster bootstrap resamples.

| Development bank | Attention Top-1 gain | Random-crop gain | Attention minus random | 95% CI | Top-1 rescue on helpful states | Random rescue |
|---|---:|---:|---:|---:|---:|---:|
| TextVQA, 318 decisions | -0.00126 | +0.00212 | -0.00338 | [-0.02014, +0.01320] | 56.0% | 41.0% |
| DocVQA, 824 decisions | +0.01204 | -0.00009 | +0.01212 | [+0.00324, +0.02101] | 67.9% | 45.5% |

DocVQA establishes a stable within-state localization signal. Always executing
the selected crop is still uneconomic: its utility is -0.03796 because it pays
for a crop on every state. TextVQA remains a negative cross-domain result.

## Source-grouped OOF action value

The factorized model predicts baseline error, conditional rescue, and harm for
each action, subtracts `0.05 * tool_cost`, and calibrates a no-call margin from
source-grouped OOF predictions.

| Development bank / feature role | Gain | Tool rate | Utility | 95% utility CI |
|---|---:|---:|---:|---:|
| TextVQA semantic state/action | +0.00912 | 9.43% | +0.00440 | [-0.00394, +0.01494] |
| DocVQA semantic state/action | +0.00655 | 3.88% | **+0.00461** | **[+0.000056, +0.01026]** |
| DocVQA context stopping + semantic action | +0.00941 | 7.89% | +0.00547 | [-0.000046, +0.01200] |

The lower-capacity DocVQA `semantic-context` model is the first development
policy whose cost-adjusted source-bootstrap interval is strictly positive. The
hybrid has a larger point estimate but misses the interval criterion and is not
selected. The simpler fixed-attention ranker plus separate stopping gate also
fails, showing that the gain comes from action-specific rescue/harm prediction,
not an unconditional attention heuristic.

## Frozen development artifacts

- code revision: `96076cc0321c1813f6e9a3dad74bdbf27ab888f7`
- DocVQA rollouts SHA-256:
  `4d3d3a33f644d1f5122aabecd47a8168d2dce2db5014692b508ba76ae4ddbe52`
- attention features SHA-256:
  `a4055bc8306321c0ca98577407e3d9ed1f4b983474178cb68422702bd6f9407a`
- selected OOF report SHA-256:
  `a3e1387c04c760c5bac2483da724a36a144f9c5dc21807d4f8e743786e7a3420`
- selected serialized model SHA-256:
  `1f8b6cf5d026bcd9921434c1c6ef0c753259d36504dedc040b8145c76bd06ff3`
- selected regularizer / call margin: `alpha=10.0` /
  `0.03311381598522578`

## Interpretation

This result validates the core methodological direction on one development
domain: question-conditioned pre-execution regional evidence can support a
cost-aware candidate value model. It does not yet establish cross-domain
robustness, and the positive lower bound is narrow. The next decisive evidence
is a label-free frozen evaluation on the untouched DocVQA formal partition,
with multiplicity correction because the original context policy remains the
registered primary analysis.

## Baseline-forward reuse audit

A post-freeze engineering diagnostic recomputed the multimodal question state
inside the same eager forward pass used for attention instead of running a
separate SDPA pass. On all 824 DocVQA development decisions:

- region-attention vectors were bit-identical (maximum absolute difference 0);
- question embeddings had mean cosine similarity `0.999669`;
- frozen predicted values differed by `0.000219` on average;
- 823/824 complete stop/action decisions agreed (`99.879%`), with 41 versus 42
  calls.

Thus attention and question conditioning can practically share one original-
image forward pass. This does not alter the separately frozen formal policy,
whose exact feature construction remains the preregistered two-pass replay.
The joint-equivalence report SHA-256 is
`7a1f972989ddaae682ab139bb6b1efd45e8b59dff16b755f0ec5aa798de1bafd`.
