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

## Reproducibility anchors

| Version | Report SHA-256 | Model SHA-256 |
| --- | --- | --- |
| v1 | `a9bcd21e795eed8935f4ee534a720791688424aeadf04c09036a9475911429e4` | `ea65ca3ad102e15a53e88b2d188579aec2521f0de34e4925a4f64160ff51f9e2` |
| v2 | `5e256922e1b671e49b20943d3df3216b6005c4a6497e84873cf1589ef3cc3890` | `be0f2c5fa9b751b8829ce2a92061683b2d5f06fc9fd686cac2a5e8ca5c4652d5` |
| v4 | `a81dbb88fdc8d08b73869fbd74055039f8823afd8cce39ad4a3789447b729790` | `3a8ad5180c291892b4ed034e1af148fbcfecf86817ec9d13e539db7b1526dbde` |
| v5 | `ab224498e2fee0a8436dd9f42d5fd9bf573bba986e607fa141ca3bce675641e4` | `fd0a8b36b7805e22f12f524f04a809473581ab6f367fda26410812ba6784b4d6` |
| v6 | `f6c1290c73736eeb948f8252a301761b336cc841ad552b42d741efb444a11743` | `744e54c9410204699c90328b5cddccd07d13dfa77317dc186205354953c4e3ac` |

## Next evidence required

The current two domains are both chart-centric and the pilot has only three
helpful states in its 58-decision validation slice. Registered DocVQA and
TextVQA development sibling banks are therefore being collected to determine
whether the factorized improvement is a chart-specific artifact, to estimate
rescue/harm heads from non-chart tasks, and to run leave-one-domain-out tests.
No new formal protocol will be registered unless at least one revised model has
non-negative utility in every development domain and positive domain-balanced
utility without collapsing to zero calls everywhere.
