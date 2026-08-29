# Beyond Entropy related-work and claim map

Status: outcome-blind paper-positioning note, written while the fresh TextVQA
calibration rollout was still running. This document does not change the frozen
candidate, threshold sequence, risks, pass rule, or formal allocation.

## Defensible central claim

The strongest current positioning is **cost-sensitive counterfactual visual
acquisition**, not merely adaptive tool use and not merely a causal audit:

> Learn the pre-action, action-specific change in downstream task success from
> acquiring visual evidence, explicitly subtract acquisition cost and predicted
> harm, and deploy only a prospectively risk-calibrated non-degenerate tail.

This claim has four separable parts:

1. A sibling counterfactual action bank measures rescue, harm, and no-effect
   outcomes for multiple crops from the same pre-action state.
2. A factorized error/rescue/harm model estimates action-specific net value
   using deployment-available features only.
3. Abstention is part of the policy: an action is executed only when its
   predicted net value clears a threshold selected by fixed-sequence risk
   calibration.
4. The claim is falsifiable through a source-disjoint calibration bank and a
   one-shot formal population with a predeclared 97.5% source-bootstrap rule.

## Nearest work and non-overlapping scope

### Training-free Uncertainty Guidance (UG)

[Kim et al., 2025/2026](https://arxiv.org/abs/2510.00705) use an MLLM's
intrinsic response uncertainty to score candidate visual inputs and localize
useful evidence without fine-tuning. Their result means this project must not
claim that entropy is generally ineffective or that it is the first method to
use uncertainty proactively for visual search.

The non-overlapping question is whether uncertainty reduction is the same as
task-success value under cost and harm. UG scores evidence after presenting a
candidate visual input. Beyond Entropy predicts net downstream value before
executing the crop, can answer now, and charges every visual acquisition. The
formal evaluator therefore needs both:

- a matched-call-rate pre-action entropy gate;
- a UG-style exhaustive candidate entropy search charged for every evaluated
  crop, plus its raw accuracy gain.

### VTool-R1

[Wu et al., 2025/2026](https://arxiv.org/abs/2505.19255) train VLMs with
outcome-reward reinforcement fine-tuning to interleave text and visual editing
steps. It already supports a broad "learn when and how to use visual tools"
claim, so Beyond Entropy must not use that as a first-of-kind statement.

The distinction is objective and evidence design. VTool-R1 optimizes the
generative tool trajectory through RFT. Beyond Entropy estimates explicit
counterfactual action value, separates rescue from induced harm, includes a
costed no-call action, and prospectively controls deployment risk. It can be
presented as a policy-selection/calibration layer complementary to tool-RL,
not as a replacement for multimodal chain-of-thought training.

### AdaTooler-V

[Wang et al., 2026](https://aclanthology.org/2026.findings-acl.898/) train
adaptive image/video tool use with a Tool Benefit Score, SFT data, and
tool-benefit-scaled RL rewards. It already targets unnecessary tool invocation
and demonstrates broad multi-benchmark adaptive behavior.

Beyond Entropy is narrower in model scale and current benchmark coverage. Its
potentially stronger axis is statistical and causal specificity: action-level
counterfactual gain minus harm and cost, pre-action-only deployment features,
an explicit answer-now action, fixed-sequence calibration, and a one-shot
formal decision. Avoid claiming broader empirical generality unless a second
benchmark or model family succeeds.

### MED: Measure--Explain--Diagnose

[Ma et al., 2026](https://arxiv.org/abs/2602.01334) decompose tool-induced
performance differences into gain and harm across tool-RL checkpoints. They
find that improvement is dominated by intrinsic learning while tool-RL mainly
reduces tool-induced harm and has limited correction of intrinsic failures.

This directly validates the importance of harm but precludes claiming the
first gain/harm decomposition. MED is primarily diagnostic after training;
Beyond Entropy attempts to turn rescue and harm into a pre-action acquisition
policy and then prospectively validate its high-value tail. The paper should
report MED-compatible quantities: rescue mass, induced-harm mass, negative
call mass, unnecessary calls, correct stopping, and crop-ranking rescue.

### The Illusion of Visual Tool-Use

[Wang et al., 2026](https://arxiv.org/abs/2608.06270) causally audit visual
tool use at policy, trajectory, and step levels. They identify "Calling Without
Looking" and "Looking Without Planning" and use counterfactual observation
replacement to isolate visual evidence gain.

This work makes a generic "first causal audit" claim untenable. Beyond Entropy
should instead be framed as a deployment response to the diagnosed failure:
learn which concrete acquisition has positive task-success value before the
call and abstain under calibrated risk. The sibling bank is an action-level
counterfactual evaluation design, while the main deliverable is a selective
policy rather than an audit taxonomy.

## Claims to avoid

- "The first method that learns when and where to use visual tools."
- "Entropy cannot identify useful visual information."
- "The first causal analysis of thinking with images."
- "The first method to model tool benefit or induced harm."
- "A general multimodal-agent solution" before a second benchmark/model result.
- Treating development OOF evidence as independent confirmation.
- Reporting accuracy gain without acquisition cost, call rate, harm, and
  source-level uncertainty.

## Minimum experiment matrix for a top-tier claim

The primary one-shot policy must be compared with:

1. answer now;
2. random gate at matched budget;
3. pre-action entropy gate at matched budget;
4. learned gate with random and each fixed crop;
5. entropy gate with learned, random, and each fixed crop;
6. UG-style exhaustive post-crop entropy selection, charging all candidates;
7. always-call random/fixed policies;
8. oracle positive-net-value policy and oracle regret.

Required decompositions are baseline-error detection, helpful-state crop
rescue, harm avoidance, and stopping. Required uncertainty is whole-source
bootstrap, never row bootstrap. All negative earlier ChartQAPro, DocVQA, and
TextVQA calibration/formal results remain visible.

## Paper narrative conditional on the current gate

If calibration and formal both pass, the main story is:

> Entropy can localize evidence after paying to inspect candidates, but
> acquisition decisions require the counterfactual value of downstream task
> success. A factorized and risk-calibrated value policy isolates a sparse set
> of cost-effective visual calls.

If fresh calibration fails, close the current policy branch. The defensible
paper direction becomes an empirical/causal study of why apparent visual-tool
headroom does not translate into independently calibratable acquisition value,
but that package would need broader models/benchmarks to clear a top-tier bar.

If calibration passes and formal fails, retain the formal result and emphasize
distributional instability of sparse tool-benefit tails; do not select a
replacement on the opened formal bank.
