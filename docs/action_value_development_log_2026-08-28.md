# Action-value development log (2026-08-28)

Status: hypothesis-generating, source-group-held-out development evidence.
ChartQAPro formal, DocVQA/TextVQA formal, and HRBench outcomes are excluded.

## Why the original transfer failed

The frozen ChartQAPro primary learned a state gate but selected a uniformly
random crop whenever it called. Its gain per call (`0.02543`) remained below
the registered cost (`0.05`), and 87.4% of calls were unnecessary. At the same
time, formal oracle utility was `0.07977`, so the next development problem is
jointly calibrating *whether* to call and *which* crop is unlikely to harm the
answer.

## Development iterations

All rows below use a fixed 80/20 source-group split within ChartQA and the
allowed 309-question ChartQAPro pilot. Domain utility is computed separately;
the pilot column is never replaced by the formal split.

| Version | Value model / features | Pooled utility | Tool rate | ChartQA utility | Pilot utility | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v1 | direct ridge, context x crop geometry, zero margin | 0.00470 | 0.2192 | 0.00606 | -0.01638 | rejected: severe cross-domain overcalling |
| v2 | direct ridge plus worst-domain no-call calibration | 0.00193 | 0.0031 | 0.00206 | 0.00000 | safe but nearly abstains |
| v4 | direct ridge, frozen Qwen ROI similarities | 0.00000 | 0.0000 | 0.00000 | 0.00000 | rejected: semantic ranker does not beat random |
| v5 | factorized error/rescue/harm, context x geometry | 0.00517 | 0.0219 | 0.00550 | 0.00000 | current strongest hypothesis; recall still missing |
| v6 | factorized error/rescue/harm, frozen Qwen ROI similarities | 0.00170 | 0.0018 | 0.00190 | 0.00000 | precise but too conservative |

The robust calibration rule always includes answer-now as a zero-utility
candidate, and selects worst-domain utility before domain-balanced mean
utility. Thus zero pilot utility in v2/v4/v5/v6 means abstention, not evidence
of positive transfer.

## Where-to-look diagnostic

The explicit action ranking diagnostic separates crop selection from stopping.
On the v5 ChartQA validation sources, learned top-1 rescued 79.5% of helpful
states versus 71.6% for a random crop. On the much smaller pilot validation
slice, it rescued 33.3% versus 25.0% for random, but its always-call mean gain
remained negative because selected crops sometimes damaged a previously useful
answer. This is why the harm head and abstention margin are necessary.

Frozen Qwen cosine/ROI features did not improve ranking: v4 pooled top-1 rescue
was 50.0% versus 56.7% for random, and pilot top-1 rescue was 0% versus 25%.
Simple one-pass similarity is therefore retained as a negative ablation, not
presented as a learned-localization contribution.

## TextVQA development evidence

The registered TextVQA development bank completed with 318 decisions from 200
source images (1,590 sibling records). Its rollout SHA-256 is
`a94c72b1977e86436c6187248f64826a34b791151c52a7c7b73ca89f92b97ddb`.
The bank supplies two distinct pieces of evidence:

- A uniformly random crop has mean task-score gain `0.00212` and mean utility
  `-0.04788`, whereas the action-and-stopping oracle has mean utility `0.05236`
  with source-clustered 95% CI `[0.03225, 0.07414]`. The visual tool therefore
  has real counterfactual headroom but blind use is decisively wasteful.
- A target-domain, five-fold source-grouped OOF factorized model using only the
  compact context-by-geometry features attains task-score gain `0.01730`, tool
  rate `0.2170`, and mean utility `0.00645`. The source-clustered 95% CI is
  `[0.00510, 0.03146]` for task-score gain and `[-0.00538, 0.02026]` for
  cost-adjusted utility. Its OOF top crop rescues 52% of helpful states versus
  41% for a uniformly random crop.

The OOF result is encouraging but not yet a confirmation: its utility interval
includes zero and the no-call margin and regularizer were selected on the OOF
bank. The serialized model is refit on all 318 development decisions only for
a future outcome-unseen formal evaluation. Repeated single 80/20 splits were
unstable (several selected answer-now everywhere), which is why the
source-grouped OOF selection and full-development refit replace a cherry-picked
split as the candidate-freezing procedure.

Adding TextVQA to one shared three-domain fit did not produce positive TextVQA
calls: the domain-robust model abstained on the TextVQA and ChartQAPro pilot
validation slices and retained positive calls only on ChartQA. This is evidence
against a single uncalibrated universal head, not against per-domain action
value learning. Explicit spatial-word features and simple frozen-Qwen ROI
cosines are retained as negative ablations; neither improved the OOF context
model.

## Reproducibility anchors

| Version | Report SHA-256 | Model SHA-256 |
| --- | --- | --- |
| v1 | `a9bcd21e795eed8935f4ee534a720791688424aeadf04c09036a9475911429e4` | `ea65ca3ad102e15a53e88b2d188579aec2521f0de34e4925a4f64160ff51f9e2` |
| v2 | `5e256922e1b671e49b20943d3df3216b6005c4a6497e84873cf1589ef3cc3890` | `be0f2c5fa9b751b8829ce2a92061683b2d5f06fc9fd686cac2a5e8ca5c4652d5` |
| v4 | `a81dbb88fdc8d08b73869fbd74055039f8823afd8cce39ad4a3789447b729790` | `3a8ad5180c291892b4ed034e1af148fbcfecf86817ec9d13e539db7b1526dbde` |
| v5 | `ab224498e2fee0a8436dd9f42d5fd9bf573bba986e607fa141ca3bce675641e4` | `fd0a8b36b7805e22f12f524f04a809473581ab6f367fda26410812ba6784b4d6` |
| v6 | `f6c1290c73736eeb948f8252a301761b336cc841ad552b42d741efb444a11743` | `744e54c9410204699c90328b5cddccd07d13dfa77317dc186205354953c4e3ac` |
| TextVQA OOF context v13 | `2d81ddbcdd6fea2308c4ebe20a3f2ed307846530689d20cbbfec9a436fdd960e` | `ca224964aeb429478aeffaa3f084750cab05daf2c56be0b3f70fda68dceadc33` |

## Next evidence required

TextVQA now demonstrates non-chart oracle headroom and a positive OOF point
estimate, but not a cost-adjusted confidence interval above zero. Registered
DocVQA development siblings are still required to test whether the same pattern
recurs with denser document text and more development states. A new formal
protocol will freeze the exact per-domain or shared calibration choice before
any DocVQA/TextVQA formal outcomes are generated; the existing formal outcomes
remain untouched.
