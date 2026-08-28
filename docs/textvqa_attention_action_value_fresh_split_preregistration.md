# TextVQA attention action-value fresh-split preregistration

Frozen: 2026-08-28 14:04 Asia/Hong_Kong, after the earlier TextVQA context
failure and DocVQA secondary failure, but before exporting, rolling out, or
scoring any target source at offset 600 or later.

## Scientific role

This is a prospective post-failure confirmation on new source groups, not a
revision of either earlier formal result. The earlier TextVQA formal sources at
offsets 200--599 remain a negative result and are never used for fitting or
calibration. The policy below was already serialized before the DocVQA
secondary outcome was opened and was fit only on the original 200-source
TextVQA development bank.

Because this confirmation follows multiple prior attempts, it uses a
conservative two-sided 97.5% source-bootstrap interval. It confirms only if
both mean utility and the 97.5% lower bound are strictly positive. Failure is
retained and cannot cause tuning or a second test on these sources.

## Frozen policy

The policy is the TextVQA-only source-grouped OOF factorized
`semantic-context` model:

- state head: compact original-image semantic state and pre-action context;
- action heads: original-image question/global/ROI similarities, fixed crop
  geometry, and final-four-layer question-to-region attention;
- value: predicted rescue minus predicted harm and `0.05 * tool_cost`;
- decision: select the maximum-valued released-UG crop only when its value is
  at least the frozen margin.

| Artifact or parameter | Frozen value |
|---|---|
| Development rollouts SHA-256 | `a94c72b1977e86436c6187248f64826a34b791151c52a7c7b73ca89f92b97ddb` |
| Development semantic features SHA-256 | `560538364f43467118d776178e8a3b6797ff5f22ddda469426992e4b79b4eada` |
| OOF report SHA-256 | `69c8b010a8a14d374d797169c9c2388758dcc12eccdf1a87fa7fc6ac4f89f1a0` |
| Serialized model SHA-256 | `f9b5dc897c5e8499ea5a245b0c512684579a5c6756da9196b628148ccf2c9a76` |
| Training code revision | `cfbf437870745e7946ea361c7268d566a2624e4c` |
| Feature mode | `semantic-context` |
| OOF folds / seed | 5 / `20260828` |
| Regularizer / call margin | `alpha=100.0` / `0.07408840253027031` |
| Development utility | `+0.00440252`, 95% source CI `[-0.00393939, +0.01493788]` |
| Development gain / tool rate | `+0.00911950` / `9.43396%` |

No policy component may change after this freeze.

## Frozen target selection

The target is 2,000 whole TextVQA validation image sources from the same pinned
dataset revision `9c0699cd19768ac5ab97568f6b3cbac4c0062884`. Selection uses the already
registered ranking

`SHA256("beyond-entropy-cross-benchmark-v1" + NUL + 20260828 + NUL + source_id)`

with `--source-group-offset 600 --source-group-count 2000`. Offsets 0--199 are
development and offsets 200--599 are the earlier formal bank. Every question
from the selected sources is retained. RGB collisions against either earlier
bank are excluded without reading targets or model outputs and deterministically
backfilled by the next hash-ranked source. The final manifest size, exclusions,
image bundle, and hashes are recorded after this outcome-independent export.

The 2,000-source size was chosen before export from the development source
bootstrap variance. A normal approximation gives an expected z-score `2.89`
and approximately 74% probability of clearing the stricter 97.5% lower bound
if the development effect is stable. This is prospective power planning, not a
claim about the unseen target effect.

## Frozen inference and feature contract

Rollouts use the same Qwen, prompt, scorer, generation seed, pixel bounds,
released-UG four-crop action set, and additive original-image-plus-crop
protocol as the earlier source-disjoint banks. Each decision produces one
answer-now and four sibling crop records.

Semantic features are built from the original-image prompt before executing a
candidate crop. The final file must contain no correctness, answer-after,
score-delta, or post-action entropy fields and must pass the strict label-free
audit. Offline feature replay cost and the practical baseline-forward reuse
path are reported separately from visual-tool utility.

## Exact evaluation

Evaluate the serialized model exactly once with:

- primary metric: mean TextVQA soft-accuracy gain minus
  `0.05 * mean tool calls`;
- 20,000 bootstrap resamples of whole `source_id` groups;
- two-sided confidence 97.5% and seed `20260828`;
- exact model, rollout, manifest, and feature SHA-256 checks; and
- mandatory label-free feature validation.

Secondary diagnostics are raw score gain, tool rate, gain per call,
unnecessary-call rate, correct-stopping rate, learned versus attention-only
ranking, and oracle headroom. None may change the pass decision.

