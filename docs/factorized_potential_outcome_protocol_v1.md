# Factorized Potential-Outcome Visual Acquisition Protocol v1

Status: method definition fixed before the first GPU smoke. This is a new
development transaction and does not revise any earlier NO-GO result.

## One-week research question

Can end-to-end post-training learn a deployable visual-acquisition score by
separating three asymmetric events that direct gain classification conflates?

For the same pre-action state, let `Y0` be STOP correctness and `Y1` be
CONTINUE correctness. The model predicts

\[
e(s)=P(Y_0=0\mid s),\quad
r(s)=P(Y_1=1\mid Y_0=0,s),\quad
h(s)=P(Y_1=0\mid Y_0=1,s).
\]

The expected acquisition gain is the identifiable decomposition

\[
\widehat G(s)=e(s)r(s)-(1-e(s))h(s),
\]

and deployment chooses CONTINUE only when
`G_hat - lambda * incremental_cost > 0`. Cost is not a training label.

The decomposition is motivated by current evidence, not by a post-hoc claim on
old results. At matched 25% cost, the earlier Outcome-only arm improved
ChartQA/DocVQA accuracy over uncertainty baselines, while direct paired-gain
training was worse and collapsed naturally to CONTINUE. On the training pairs,
direct gain receives only 63 non-neutral labels out of 512. The factorized loss
uses every state for error risk and one observable conditional event.

## Inputs and leakage boundary

The deployable input remains exactly the existing typed partial-prefix view:
original image, one already acquired crop, question/prompt, proposed action ID
and bbox, and proposed cost. The proposed crop is not executed. Neither branch
answer, correctness, entropy, reward, gain, sibling outcome nor ground truth is
accepted by the model forward method.

Train and validation remain source/image disjoint. The current 256-train and
128-validation banks for ChartQA and DocVQA are development-only because their
validation outcomes were seen by earlier routes. No result from them is a
formal paper claim. A fresh held-out allocation is permitted only after the
Phase-B transition rule passes.

## Model and objective

The encoder, Qwen2.5-VL-3B revision, pixel budget, trainable vision merger,
last language block/norm, optimizer, schedule, steps and seed match the two
existing controls. The only architectural difference is a three-logit head for
`error`, `rescue|error` and `harm|correct` instead of a two-action head.

For each paired state the loss is

\[
L={1\over2}\big[
\operatorname{BCE}(e,1-Y_0)
+\mathbb 1[Y_0=0]\operatorname{BCE}(r,Y_1)
+\mathbb 1[Y_0=1]\operatorname{BCE}(h,1-Y_1)
\big].
\]

No inverse-frequency weight, focal loss, auxiliary gain loss, temperature,
threshold fitting or hyperparameter search is allowed in Phase A/B. The three
arms are:

1. Outcome-only final-reward control;
2. direct Counterfactual Utility preference control;
3. Factorized Potential Outcomes (proposed).

This is not claimed as a new causal-inference estimator. Shared potential-
outcome heads are established in TARNet/CFR-style work. The candidate novelty
is the rescue/harm conditional factorization as a deployable, cost-aware visual
acquisition objective, supported by paired sibling executions. It differs from
VTool-R1 outcome-only RL, AdaptVision turn-level objective decoupling,
AdaTooler-V question-level benefit scaling, ToolVision evidence filtering, and
MED's checkpoint diagnosis. A broader novelty claim is not allowed without a
complete collision audit.

Primary references:

- Shalit et al., *Estimating individual treatment effect: generalization bounds
  and algorithms*, ICML 2017: https://proceedings.mlr.press/v70/shalit17a.html
- AdaptVision, CVPR 2026:
  https://openaccess.thecvf.com/content/CVPR2026/html/Lin_AdaptVision_Efficient_Vision-Language_Models_via_Adaptive_Visual_Acquisition_CVPR_2026_paper.html
- VTool-R1, ICLR 2026: https://arxiv.org/abs/2505.19255
- AdaTooler-V: https://arxiv.org/abs/2512.16918
- ToolVision: https://arxiv.org/abs/2608.08907
- MED: https://arxiv.org/abs/2602.01334

## Frozen stages and decisions

### Phase A: engineering smoke

- same outcome-independent 5--10% subsets as the previous smoke;
- 64 optimizer steps, seed 17;
- all three arms run concurrently on one GPU each;
- require finite loss/scores, nonzero gradients and updates in every trainable
  group, decreasing fixed-audit loss, no proposed-crop execution, matched
  schedule, no test access and noncollapsed validation scores.

An engineering failure may be fixed only if it does not change the estimand,
loss, data, schedule or decision rule.

### Phase B: bounded development pilot

- ChartQA and DocVQA each use all 256 train and 128 validation states;
- 512 matched optimizer steps, seed 17;
- exact 25% call rate is primary; call-rate frontier is secondary;
- whole-source paired bootstrap uses 10,000 samples.

Phase B advances only if all three conditions hold:

1. Factorized exceeds the strongest matched uncertainty baseline by more than
   `+1pp` on one domain;
2. it remains above `-0.5pp` versus that baseline on the other domain;
3. its mean accuracy delta over Outcome-only across the two domains is positive.

It is an immediate NO-GO if it is nonpositive against the strongest uncertainty
baseline on both domains, or nonpositive against Outcome-only on both domains.
Any result that misses the joint transition rule is NO-GO for this candidate.

### Phase C: only after Phase-B GO

Within the one-week window, create larger paired train banks and a fresh
held-out allocation, then run three fixed seeds on ChartQA, DocVQA and HRBench.
The formal method requires positive direction on at least two domains, a
source-bootstrap lower endpoint above zero against Outcome-only on at least one
domain, and no material accuracy-cost regression on the other successful
domain. Semantic controls must show dependence on image, question and proposed
region. Otherwise the route stops and is reported honestly.

## One-week stop boundary

No RL, 7B, continuous bbox, free-form tool syntax, second acquisition, new
visual tool, prompt search or loss family search is allowed before Phase C.
If Phase B is NO-GO, this route ends; the remaining time is used to consolidate
the negative empirical finding, not to cycle through seeds or renamed losses.

