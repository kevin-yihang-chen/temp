# Frozen-Qwen diagnostic results — 2026-08-28

> Scientific status: exploratory frozen-model diagnostics, not a final benchmark
> claim. The held-out sets were used repeatedly during method development after
> the first registered evaluations, so subsequent variants are hypothesis-
> generating rather than confirmatory.

## Frozen protocol

- Model: `Qwen/Qwen2.5-VL-3B-Instruct` at revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Dataset: ChartQA test Arrow file with SHA-256
  `b71823482e02f4dce7ae59e56b0a4dc3b030232200fd07233fb7337a09a16af7`.
- Initial balanced slice: 1,000 states, 500 human and 500 augmented, manifest
  SHA-256 `925ee5ca34c0349e8642bf88e75279e4755ce15049d4127afc45a1710aa18640`.
- Full ChartQA test replication: 2,500 states, 1,250 human and 1,250
  augmented, 1,509 unique images; manifest SHA-256
  `3c485aa5c09cc9491f866ba5737a78c2b79c3539c6de2663c964b2cff90d814a`.
- Rollouts: one answer-now action and four UG grid crops per state; original image
  plus crop are supplied additively; deterministic generation seed 0;
  `max_new_tokens=16`.
- Concise system prompt: `Answer with only the final answer: a single number,
  word, or short phrase. Do not explain.`
- Frozen 1,000-state rollout SHA-256:
  `eeee102e8de77ee35026e221b51053535b16d37dada94652023b07b14d595414`.
- Frozen 2,500-state rollout SHA-256:
  `881526ccd3ff03753127307128c84dcf9dfa217f06635934ed2c5bca6d93973c`.
- Confidence intervals resample complete `state_id` clusters. Actions from one
state are never treated as independent observations.

## Full 2,500-state replication

At `lambda=0.05`, the complete ChartQA test split reproduces the original
diagnostic almost exactly:

| Policy | Accuracy | Accuracy gain [95% CI] | Calls/state | Utility [95% CI] |
|---|---:|---:|---:|---:|
| Answer now | 0.8128 | 0.0000 | 0.00 | 0.0000 |
| Uniform random crop (exact expectation) | 0.8262 | 0.0134 [0.0059, 0.0213] | 1.00 | -0.0366 [-0.0441, -0.0287] |
| Fixed-center crop | 0.8256 | 0.0128 [0.0040, 0.0216] | 1.00 | -0.0372 [-0.0460, -0.0284] |
| Entropy search | 0.8320 | 0.0192 [0.0104, 0.0284] | 4.00 | -0.1808 [-0.1896, -0.1716] |
| Oracle VOI | 0.8632 | 0.0504 [0.0420, 0.0592] | 0.0504 | 0.0479 [0.0399, 0.0562] |

The main claim is not that crops fail: random, fixed, and exhaustive policies
all have positive accuracy-gain intervals. They fail the registered
accuracy-cost objective because the useful transitions are too sparse to pay
for indiscriminate calls. Oracle VOI retains a clearly positive five-point
frontier improvement, so selective tool use remains meaningful.

The point-estimate break-even costs make the frontier explicit: uniform random
crop breaks even at `lambda=0.0134`, fixed crop at `0.0128`, and four-crop
entropy search at only `0.0048`; oracle selection breaks even at `1.0` because
every oracle call is a successful rescue. Uniform random already has higher
utility than exhaustive entropy search once `lambda` exceeds about `0.0019`.
Thus entropy search is competitive only when visual calls are almost free.

The complete action table contains 320 positive and 186 negative transitions
among 10,000 crops. Of crop actions, 39.74% increase confidence, but only 5.41%
of confidence-increasing actions improve correctness; 37.59% increase confidence
without improving the answer. Entropy reduction and success gain have Pearson
correlation 0.126 [0.045, 0.206].

| Full-test stratum | Answer now | Entropy search | Oracle | Helpful states | Oracle utility [95% CI] |
|---|---:|---:|---:|---:|---:|
| Human test (1,250) | 0.6880 | 0.7232 | 0.7760 | 0.0880 | 0.0836 [0.0692, 0.0988] |
| Augmented test (1,250) | 0.9376 | 0.9408 | 0.9504 | 0.0128 | 0.0122 [0.0068, 0.0182] |

This stronger replication confirms that aggregate opportunity is dominated by
human-authored questions; a single aggregate number obscures a nearly sevenfold
difference in helpful-state rate.

## Initial 1,000-state diagnostic

At cost coefficient `lambda=0.05`:

| Policy | Accuracy | Accuracy gain [95% CI] | Calls/state | Utility [95% CI] |
|---|---:|---:|---:|---:|
| Answer now | 0.826 | 0.000 | 0.00 | 0.000 |
| Random crop | 0.835 | 0.009 [-0.005, 0.023] | 1.00 | -0.041 [-0.055, -0.027] |
| Fixed-center crop | 0.839 | 0.013 [0.000, 0.026] | 1.00 | -0.037 [-0.050, -0.024] |
| Entropy search | 0.846 | 0.020 [0.005, 0.035] | 4.00 | -0.180 [-0.195, -0.165] |
| Oracle VOI | 0.876 | 0.050 [0.037, 0.064] | 0.05 | 0.0475 [0.0352, 0.0608] |

Extra evidence can improve accuracy, but evaluating every candidate is not
cost-effective. The oracle headroom is both non-zero and sparse: only 5.0% of
states contain a positive-cost crop at this lambda.

Entropy reduction is also a noisy usefulness proxy:

- 40.43% of crop actions increase confidence, but only 5.88% of those actions
  improve task correctness.
- 38.05% of all crop actions increase confidence without improving correctness.
- Entropy reduction and success gain have Pearson correlation 0.209.
- Entropy's within-state top choice disagrees with a success-optimal choice on
  2.3% of states. This number is bounded by the low 5.0% helpful-state rate and
  should not be interpreted without that denominator.

## Strata

| Stratum | Answer now | Entropy search | Oracle | Helpful states | Oracle utility |
|---|---:|---:|---:|---:|---:|
| Human test (500) | 0.702 | 0.736 | 0.784 | 0.082 | 0.0779 |
| Augmented test (500) | 0.950 | 0.956 | 0.968 | 0.018 | 0.0171 |

The aggregate opportunity is concentrated in human-authored questions. The
augmented split is nearly saturated and should be reported separately rather
than allowed to hide this heterogeneity.

## Prompt and decoding controls

The original helpful-assistant prompt produced a protocol artifact on the first
200-state ChartQA slice: 99/1,000 action generations hit the 16-token cap, and
all capped generations were scored wrong. Increasing the cap from 16 to 32
changed 98/175 targeted outputs but changed zero correctness labels. Nevertheless,
entropy-search accuracy on that subset changed from 0.314 to 0.371 solely because
continuation-token entropy changed.

On the same 35-state targeted subset, the concise prompt removed all 99 cap hits,
changed 63 wrong answers to correct and only two correct answers to wrong, and
raised answer-now accuracy from 0.486 to 0.714. The default-prompt ChartQA run is
therefore retained only as a negative protocol-control, not as primary evidence.

## Candidate-count ablation

A matched 200-state ablation compares the four-quadrant proposer with a 3-by-3
nine-crop grid. The answer-now records are byte-equivalent after canonical
sorting (identical answer, correctness, and entropy hashes), so the difference
is attributable to the candidate set rather than decoding drift. The nine-crop
rollout SHA-256 is
`0bf4e13d2b87b8229b4b2b8967ec9e071ae57da7f8a629e655acb32845e88dba`.

| Candidate set | Helpful states | Harmful states | Expected random gain [95% CI] | Entropy gain [95% CI] | Calls | Entropy utility [95% CI] | Oracle gain [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Four crops | 0.045 | 0.070 | -0.0013 [-0.0275, 0.0262] | 0.010 [-0.025, 0.050] | 4 | -0.190 [-0.225, -0.150] | 0.045 [0.020, 0.075] |
| Nine crops | 0.060 | 0.065 | 0.0056 [-0.0200, 0.0322] | 0.015 [-0.015, 0.045] | 9 | -0.435 [-0.465, -0.405] | 0.060 [0.030, 0.095] |

The denser grid exposes three additional rescuable states and raises oracle
headroom by 1.5 points, but exhaustive entropy search captures only 0.5 point of
additional accuracy while paying for five more calls. Candidate coverage is a
real bottleneck, yet naive search expansion makes the accuracy-cost frontier
strictly worse at the registered cost.

A paired state-bootstrap comparison (nine minus four) gives entropy-search gain
`0.0050 [-0.0150, 0.0300]` and utility difference
`-0.2450 [-0.2650, -0.2200]`. The oracle-gain difference is
`0.0150 [0.0000, 0.0350]`, while the exact uniform-random difference is
`0.0068 [-0.0022, 0.0157]`. Thus only the cost degradation is statistically
resolved on this small matched slice; the apparent extra task headroom remains
a diagnostic point estimate.

## Held-out value-model diagnostic

The fixed image-grouped split uses 701 outer-train and 299 test decisions. At
`lambda=0.05`:

| Policy | Accuracy gain | Tool rate | Utility [95% CI] |
|---|---:|---:|---:|
| Entropy gate + uniform random crop (exact expectation) | 0.0117 | 0.134 | 0.0050 [-0.0092, 0.0209] |
| Scalar ridge | -0.0033 | 0.117 | -0.0092 [-0.0217, 0.0017] |
| Scalar ridge + grouped OOF threshold | -0.0033 | 0.090 | -0.0079 [-0.0209, 0.0030] |
| Semantic MLP | 0.0000 | 0.000 | 0.0000 |
| Structured semantic success difference | 0.0000 | 0.043 | -0.0022 [-0.0033, -0.0010] |
| Semantic similarity ridge | 0.0000 | 0.040 | -0.0020 [-0.0120, 0.0079] |
| Oracle VOI | 0.0502 | 0.050 | 0.0477 [0.0254, 0.0731] |

The MLP's best validation epoch was 1 and its monotone affine calibration slope
was zero. Direct regression on sparse `{-1, 0, +1}` action gains collapsed to
the dominant zero target. More data or a structured rescue/harm objective is
required before this head can support the method claim.

The entropy-gated random-crop row is the exact counterfactual expectation of a
uniform one-crop policy, not one lucky pseudo-random draw. Its positive point
estimate is not statistically resolved. When a concrete random action and an
inner-validation threshold are evaluated across 100 deterministic policy seeds,
mean utility is 0.00213, only 47/100 seeds are positive, and seed 17 is the
maximum at 0.0100. Random-seed variance must therefore not be presented as a
method improvement.

## Full-test scalar stopping diagnostic

On the full rollout, an image-grouped 70/30 split contains 1,741 outer-train and
759 test decisions. All thresholds are selected outside the test groups.

| Policy | Accuracy gain | Tool rate | Utility [95% CI] |
|---|---:|---:|---:|
| Entropy gate + uniform random crop (exact expectation) | 0.00725 | 0.120 | 0.00125 [-0.00876, 0.01113] |
| Entropy gate + fixed crop | 0.00922 | 0.120 | 0.00323 [-0.00843, 0.01482] |
| Scalar learned VOI | 0.00264 | 0.0856 | -0.00165 [-0.01061, 0.00744] |
| Oracle VOI | 0.05007 | 0.0501 | 0.04756 [0.03379, 0.06258] |

More data improves the point estimates of simple gates, but none has a resolved
positive held-out utility. The scalar learned model remains below zero at its
point estimate.

The corresponding semantic experiment uses 1,408 model-train, 333 validation,
and the same 759 held-out decisions. Sparse gain regression now trains for 23
best epochs and the structured success-difference objective for 69, so the old
epoch-1 collapse is reduced, but policy quality does not improve:

| Full-test semantic policy | Accuracy gain | Tool rate | Utility [95% CI] |
|---|---:|---:|---:|
| Semantic gain + validation threshold | 0.0000 | 0.0224 | -0.00112 [-0.00514, 0.00283] |
| Structured semantic success difference | -0.00132 | 0.0264 | -0.00264 [-0.00732, 0.00152] |
| Semantic similarity ridge | 0.0000 | 0.00659 | -0.00033 [-0.00435, 0.00356] |
| Entropy gate + fixed crop | 0.01054 | 0.112 | 0.00494 [-0.00573, 0.01528] |
| Oracle VOI | 0.05007 | 0.0501 | 0.04756 [0.03379, 0.06258] |

The larger sample therefore preserves the Gate 2 blocker: every learned
semantic policy has a non-positive point estimate, while the best simple gate
has an unresolved positive point estimate.

Five repeated image-grouped splits confirm substantial instability. Because
their test sets overlap, the ranges below are robustness summaries, not a new
confidence interval:

| Policy | Mean utility [min, max] | Positive splits |
|---|---:|---:|
| Validation-thresholded semantic gain | 0.00091 [-0.00404, 0.00928] | 2/5 |
| Structured semantic success difference | -0.00035 [-0.00264, 0.00147] | 1/5 |
| Semantic similarity ridge | -0.00011 [-0.00060, 0.00084] | 1/5 |
| Entropy gate + uniform random expectation | 0.00073 [-0.00437, 0.00510] | 3/5 |
| Scalar learned VOI | 0.0012 [-0.0016, 0.0043] | 3/5 |
| Oracle VOI | 0.0441 [0.0318, 0.0497] | 5/5 |

## Compact two-stage rescuability gate

Failure localization suggests separating the state-level question "can any crop
rescue this answer?" from action selection. A low-capacity logistic gate is
therefore trained on model-train groups using only baseline signals, frozen
question/global cosine, per-region question/global cosines, and candidate boxes.
Regularization and the call threshold are chosen on inner validation utility.
If the gate fires, deployment samples one uniform crop; evaluation uses the exact
sibling expectation only to remove arbitrary action-seed variance.

Across the same five image-grouped seeds, this gate has mean accuracy gain
0.00677, mean tool rate 0.0952, and mean utility 0.00201. Four of five split point
estimates are positive, with utility range [-0.00203, 0.00573]. Helpful-state
ROC-AUC ranges from 0.781 to 0.877. However, every split's state-bootstrap
utility interval crosses zero. This is the strongest learned direction so far,
but it remains promising rather than resolved evidence.

## Factorized stopping gate

The stronger stopping formulation factorizes pre-action value into (1) the
probability that answer-now is wrong and (2) conditional rescuability given a
wrong answer. In nested five-fold image-grouped OOF evaluation on all 2,500
development states, the full context model has accuracy gain 0.0130, tool rate
0.1276, and utility 0.00662. Its state-bootstrap 95% interval is
`[0.00142, 0.01198]`; the stricter image-bootstrap interval is
`[0.00125, 0.01228]`. This is leakage-safe development evidence, but variants
were selected after repeated use of the split and therefore remain
hypothesis-generating.

A feature ablation localizes the useful signal. The text/form-only conditional
rescue model remains positive at utility 0.00582 with state interval
`[0.00100, 0.01066]` and image interval `[0.00095, 0.01097]`. Uncertainty alone
has utility 0.00132, question-only 0.00070, and answer-plus-uncertainty 0.00222;
all three intervals cross zero. Question and baseline-answer surface structure,
not entropy alone, drives most of the stopping signal.

Composing the OOF state gate with a question-type-by-quadrant action ranker gives
utility 0.00882, versus 0.00842 for the same gate with fixed crop 0 and 0.00662
for uniform-random crop expectation. However, the paired ranker-minus-random
difference is only 0.00220 with state interval `[-0.00060, 0.00510]` and image
interval `[-0.00060, 0.00499]`. The ranker-minus-fixed-crop difference also
crosses zero. Development evidence therefore supports stopping more strongly
than spatial action selection.

## Independent ChartQA validation confirmation

The full-context factorized deployment model, scaler, regularization, absolute
threshold, source report, and secondary baselines were frozen before any target
outcomes were inspected. The target is the official ChartQA validation split
after excluding two questions whose image appeared in development: 1,918 states
from 1,054 images, with rollout SHA-256
`a0d11b785ee6683dc34277740e3abfcd7d84323a740d88da5ef68ddb2eb98257`.

The registered primary **does not pass**, although it is a near miss. Its utility
is 0.003415, accuracy gain 0.006257, and tool rate 0.05683. The 95% state interval
is `[-0.000026, 0.007195]` and the image interval is
`[-0.000131, 0.007249]`. Positive utility, positive gain, and lower tool use than
unconditional policies pass, but the registered state-CI lower-bound condition
does not. The report SHA-256 is
`8e5dcb337161b6ff50e3e4a3cab301195430f5656d5a1049d745b01e276d7d55`.

| Frozen validation policy | Accuracy gain | Tool rate | Utility [95% state CI] | Image CI |
|---|---:|---:|---:|---:|
| Full-context gate + random crop expectation (primary) | 0.00626 | 0.0568 | 0.00342 [-0.00003, 0.00719] | [-0.00013, 0.00725] |
| Same gate + fixed crop 0 (secondary) | 0.00730 | 0.0568 | 0.00446 [0.00078, 0.00868] | [0.00060, 0.00864] |
| Same gate + learned quadrant ranker (secondary) | 0.00574 | 0.0568 | 0.00289 [-0.00149, 0.00743] | [-0.00135, 0.00738] |
| Text-only conditional-rescue gate (secondary) | 0.00600 | 0.0714 | 0.00242 [-0.00109, 0.00631] | [-0.00123, 0.00635] |
| Frozen source entropy gate | 0.00352 | 0.0746 | -0.00021 [-0.00370, 0.00334] | — |
| Always random crop | 0.01538 | 1.0000 | -0.03462 [-0.04322, -0.02589] | — |
| Exhaustive entropy | 0.02398 | 1.0000 | -0.17602 [-0.18644, -0.16559] | — |
| Oracle VOI | 0.04953 | 0.0495 | 0.04705 [0.03764, 0.05647] | — |

The primary is heterogeneous: on 958 human questions it has utility 0.00736
`[0.00094, 0.01456]`, while on 960 augmented questions it has utility -0.00052
`[-0.00229, 0.00193]`. This stratum result is secondary and cannot be used to
retroactively restrict deployment.

Post-hoc paired contrasts clarify the fixed-crop secondary without changing the
primary conclusion. Fixed crop 0 minus uniform-random expectation is +0.00104,
but both the state interval `[-0.00065, 0.00287]` and image interval
`[-0.00066, 0.00286]` cross zero. The learned ranker minus random is -0.00052
with intervals crossing zero. Thus fixed crop 0 is a positive pre-registered
deployment variant, but there is no resolved evidence that its action choice is
better than random. The post-hoc report SHA-256 is
`420c020b982131db62389c459aa4e9a2a29b5a5ecc711ac7a00c8883dccb7257`.

## High-power independent replication in progress

The validation interval motivated a new replication rather than extending or
retuning the failed primary. Before target rollout, 4,500 official ChartQA train
states were frozen with exactly one state per image, balanced 2,250/2,250 across
human and augmented strata, and with zero image overlap against both development
and validation. Its manifest SHA-256 is
`72db6feaa4bc042e98741a48dd55421c5246c1b48c84b1fd75740d1d072ca621`.
The model, threshold, four-crop UG proposal, cost, prompt, and pass criteria are
unchanged. This is explicitly a post-near-miss replication; its protocol is in
`docs/replication_protocol_chartqa_train.md`.

## Image-disjoint chart-layout confirmation

A chart-specific four-crop proposer appeared promising on a 200-state
development slice: its uniform-random one-crop utility exceeded the matched UG
candidate set by 0.01625 with interval `[0.00625, 0.02750]`. The proposer and a
go/no-go rule were therefore frozen before collection on a target excluding all
189 development images.

The 2,137-state, 1,320-image confirmation **does not pass**. The treatment-minus-
UG uniform-random difference is 0.001755, with state interval
`[-0.001524, 0.005030]` and image interval `[-0.001420, 0.004951]`. Fixed-center,
entropy-search, and oracle candidate-set differences also cross zero. The large
development effect therefore does not replicate; the small positive target
point estimate is unresolved. The report SHA-256 is
`e87e16f0f6da83efe0efb4fcf893a9e67ae010ab02269561b47743c7c832e1e3`.

The pre-frozen follow-up rule required both state and image lower endpoints
above zero before launching a 4,500-image chart-layout treatment. Because the
rule fails, that treatment is not launched. Proposal design and per-question
action localization remain open.

## Failure localization

- Across 1,000 states there are 50 helpful states and 39 harmful states. Crop
  actions contain 118 positive and 78 negative transitions out of 4,000.
- Baseline entropy predicts whether the current answer is wrong well on the
  fixed test split (ROC-AUC 0.910), so generic error detection is not the main
  bottleneck.
- Among the 56 baseline-wrong held-out states, only 15 are rescuable by the four
  proposed crops; entropy's helpful-versus-nonhelpful AUC is only about 0.64.
- For 224 crop actions attached to baseline-wrong states, individual pre-action
  semantic similarities have AUCs of roughly 0.52–0.60 for rescue. Post-action
  entropy reduction reaches 0.77, but requires executing the crop and is still
  not equivalent to task success.
- Helpful states often have more than one useful quadrant: 33/50 have at least
  two successful crops. This indicates that stopping/rescuability prediction is
  currently a larger bottleneck than exact quadrant ranking.

The full replication strengthens this diagnosis. It contains 126 helpful and 88
harmful states; 87/126 helpful states have at least two successful crops, and 42
are rescued by all four. On the seed-17 image-grouped test split, 38/759 states
are helpful. Entropy predicts baseline error with ROC-AUC 0.874, but among the
147 baseline-wrong states its helpful-versus-nonhelpful ROC-AUC is only 0.687
(average precision 0.448). More observations improve label support without
turning generic uncertainty into a reliable rescuability score.

## Current decision

Do not enter VTool-R1/GRPO yet. Gate 1 establishes sparse counterfactual
headroom and the cost failure of exhaustive entropy search. The factorized gate
is the first method with positive nested OOF state- and image-bootstrap intervals,
and its independent validation point estimate transfers, but the registered
validation interval narrowly crosses zero. A fixed-crop secondary passes on
validation, while paired contrasts provide no evidence that fixed or learned
action selection beats random. The independently tested chart-layout proposer
also fails its state/image confirmation. Gate 2 is therefore partially supported
for stopping and still open for both independent replication and spatial action
selection. The frozen 4,500-image replication must finish before the stopping
claim is upgraded; no chart-layout follow-up is launched.

## Artifacts

- `artifacts/gate1-chartqa-1000/qwen3b-c4-concise-seed0/pilot_report.json`
- `artifacts/gate1-chartqa-1000/qwen3b-c4-concise-seed0/pilot_report.md`
- `artifacts/gate1-chartqa-2500/qwen3b-c4-concise-seed0/pilot_report.json`
- `artifacts/gate2-chartqa-2500/scalar-seed17-lambda005-v1/report.json`
- `artifacts/gate2-chartqa-2500/qwen3b-roi-concise-seed17/semantic-model-oof-v1-success/report.json`
- `artifacts/gate2-chartqa-2500/qwen3b-roi-concise-multiseed/robustness_report.json`
- `artifacts/gate2-chartqa-2500/compact-rescue-gate-multiseed-v1/report.json`
- `artifacts/gate2-chartqa-2500/factorized-context-ablation-v12-image-bootstrap/report.json`
- `artifacts/gate2-chartqa-2500/composed-factorized-context-quadrant-v13-image-bootstrap/report.json`
- `artifacts/confirmation-chartqa-val-1918/frozen-factorized-context-v1/report.json`
- `artifacts/confirmation-chartqa-val-1918/posthoc-action-contrasts-v1/report.json`
- `artifacts/confirmation-chart-layout-2137/matched-comparison-v1/report.json`
- `docs/replication_protocol_chartqa_train.md`
- `artifacts/gate1-chartqa-200/qwen3b-c9-concise-seed0/pilot_report.json`
- `artifacts/gate1-chartqa-200/c4-vs-c9-concise-seed0/candidate_ablation.json`
- `artifacts/gate2-chartqa-1000/qwen3b-roi-concise-seed17/features.pt`
- `artifacts/gate2-chartqa-1000/qwen3b-roi-concise-seed17/semantic-model-oof-v1/report.json`
- `artifacts/gate2-chartqa-1000/qwen3b-roi-concise-seed17/semantic-model-oof-v2-success/report.json`
- `artifacts/gate2-chartqa-1000/scalar-seed17-lambda005-v3/report.json`
