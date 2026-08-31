# Beyond Entropy literature collision and claim repositioning

Date checked: 2026-08-30; refreshed 2026-08-31 after the ScreenQA semantic
candidate failed and two newer August papers were audited.

Scientific status: planning note written while DocVQA factorized-v2 calibration
rollouts were still being collected and before calibration or formal outcomes
were read. It does not modify the frozen experiment.

## Material new overlap

Several 2026 papers materially overlap broad versions of the idea.

**ToolGate: Token-Efficient Pre-Call Control for Tool-Augmented
Vision-Language Agents** studies binary execution control after a ReAct VLM has
already proposed one pending perceptual call. Its external controller consumes
trajectory and pending-call text plus structural features. The default target
is a forced-answer/trajectory proxy rather than an enumerated causal action
effect. Its cross-domain result centers on preserving accuracy while reducing
token and call cost.

**Beacon: Knowing When and How to Perform Agentic Visual Reasoning** makes two
quantities central: Mode Adaptiveness, whether tools are invoked only when
needed, and Tool Effect, whether tools rescue hard cases without damaging easy
cases. It explicitly reports that gains on hard examples can be offset by harm
on examples already solvable without tools, then trains with a
Necessity-Aware Adaptive Reward and Hint-Guided Capability Expansion. This is a
direct collision with any claim that identifying visual-tool harm or
necessity-aware invocation is itself novel.

**ToolVision: Learning When and How to Use Visual Tools with
Capability-Aligned Supervision** is still closer. Before RL, it evaluates the
frozen learner with and without tools per question and assigns stronger tool
reward only where tools show a clear benefit. Its SFT pipeline explores tool
trajectories and uses a student-scale committee to score stepwise evidence
gain. This invalidates a broad claim that paired with/without-tool benefit
supervision or capability-aligned when/how learning is new. It remains
different from a complete, fixed candidate sibling bank: its target is a
question-level necessity tier and successful trajectory supervision, rather
than a signed effect for ANSWER-NOW and every concrete candidate action,
including harmful and zero-value actions.

**MED (Measure--Explain--Diagnose)** evaluates crop-and-zoom RL checkpoints
with and without tool availability, decomposes the tool-induced performance
gap into gain and harm, and further separates call effects from schema-only
effects. Its main finding is that current RL improvements are dominated by
intrinsic capability learning and reduced tool-induced harm rather than
stronger correction of intrinsic failures. This directly collides with any
claim that decomposing visual-tool gain and harm is new. MED is a
checkpoint-level attribution framework, however; it does not supervise or
deploy a pre-action value for every concrete same-state crop candidate.

**BCEA (Budgeted Conformal Evidence Acquisition)** replaces selective
answer/abstain with answer/abstain/acquire and folds a visual-evidence
acquisition policy into a conformal score before recalibration. It shows that
naively adding acquisition after calibration can break the risk guarantee and
that post-acquisition recalibration can restore finite-sample hallucination
control. This directly collides with broad claims around risk-calibrated visual
acquisition or answer-versus-acquire selection. Its estimand is selective
claim-level hallucination risk and coverage under a bounded budget, not the
signed task-score effect and induced harm of each pre-enumerated visual action.

**AdaptVision (CVPR 2026)** trains a Qwen2.5-VL policy to choose direct answer
versus a generated crop through RL, with accuracy-efficiency rewards and
decoupled turn-level credit assignment. Other close systems broaden the
collision: **CropVLM** learns a transferable one-shot crop generator through
RL; **AVA-VLM** teaches domain-specific when/where cropping with region-aware
CoT; **ToolsRL** uses tool-specific curriculum rewards for zoom, rotate, flip,
and drawing; **VTool-R1 (ICLR 2026)** trains visual tool use with outcome-based
RFT; and **UG (ECCV 2026)** uses post-acquisition response uncertainty to
select visual evidence without training.

**GapSight / Learning to Look Again (arXiv:2608.21762, released 2026-08-22)**
is a direct collision with the broadest version of our learning claim.  It
executes a diverse candidate crop bank offline, uses the target model's
teacher-forced answer-NLL reduction or correct-option margin improvement as a
model-specific utility label, and trains a frozen-backbone router to jointly
predict preserve/review, scalar utility, and a continuous crop.  It reports
positive results on six benchmarks and three VLM backbones.  Our realized
task-score sibling target, explicit harmful/zero actions, complete fixed action
bank, source-level risk gate, and one-shot outcome protocol remain different,
but "first model-specific counterfactual crop utility", "first loss-difference
crop supervision", and "first joint when/where one-shot crop router" are no
longer defensible.

**The Illusion of Visual Tool-Use (arXiv:2608.06270)** performs policy-,
trajectory-, and step-level causal interventions on crop-and-zoom agents.  Its
Visual Evidence Gain holds the prefix and action fixed while replacing a
returned crop, isolating the observation-mediated contribution.  It finds that
aggregate gains concentrate in a calibrated minority and that many calls either
do not affect the answer or follow incoherent schedules.  This overlaps the
diagnostic thesis that successful trajectories and tool calls need not imply
useful visual evidence.  It does not learn a deployment-time value over every
same-state candidate or prospectively control action harm, but it prevents us
from claiming the first counterfactual or causal audit of returned visual
observations.

The statements "first to learn when to call a visual tool", "first pre-call
VoI gate", "first adaptive answer-or-crop policy", "first paired tool-benefit
supervision", "first to model or decompose tool harm", "first risk-calibrated
visual acquisition", "first model-specific loss-gap crop supervision", "first
counterfactual audit of returned visual evidence", and "first visual tool credit
assignment" are therefore not defensible.

## Remaining defensible methodological center

The paper should be repositioned around the conjunction below, not around any
single generic pre-call-gating phrase:

1. **Complete signed candidate sibling supervision.** At exactly the same
   pre-action state, materialize ANSWER-NOW and every fixed candidate visual
   action as independent siblings. Directly measure the signed task delta of
   each concrete candidate relative to answering now, retaining rescue, harm,
   and zero effect. This is narrower than generic paired with/without-tool
   necessity supervision.
2. **Joint stop-and-action selection.** Predict a value for every candidate crop
   before executing any crop, then jointly decide whether to stop and which
   single visual action to execute. This differs from accepting or rejecting an
   already proposed call.
3. **Task value rather than proxy confidence.** Learn action-specific task-score
   change minus explicit execution cost, and test the failure of post-action
   entropy as a proxy under matched execution budgets.
4. **Prospective action-harm control.** Use source-grouped OOF fitting followed
   by an independently frozen fixed-sequence gate that constrains induced harm
   and negative-value calls before a one-shot formal test. This is not a
   conformal coverage or hallucination-risk guarantee and must not be described
   as one.
5. **Auditable counterfactual bank.** Preserve all sibling outcomes and exact
   provenance so action ranking, stopping, oracle regret, and exhaustive-search
   cost can be evaluated from the same states without changing the deployed
   gate.

No one component alone is enough for a novelty claim. The contribution is the
combination of direct sibling action effects, joint where/when selection, and
prospective harm-controlled evaluation.

## Recommended paper framing

Working title (safer after the MED/BCEA collision audit):

> **Prospectively Harm-Controlled Counterfactual Action Value for Selective Visual
> Acquisition**

Core contrast sentence:

> Existing methods train adaptive visual-tool policies with trajectory,
> necessity-aware, or with/without-tool rewards; gate one call already proposed
> by an agent; or select acquired evidence using uncertainty. We instead retain
> ANSWER-NOW and every fixed crop as same-state siblings, supervise the signed
> task effect of each concrete action including harm, learn a cost-aware joint
> stop/action value, and independently calibrate deployment risk before one-shot
> evaluation.

The phrase **Beyond Entropy** remains useful as a subtitle or empirical finding,
but it should no longer carry the full novelty claim.

## Evidence still required for a competitive paper

The current formal protocol already covers entropy/random/fixed/exhaustive UG
controls. If its primary test passes, the next evidence gap is a separately
declared comparator family under the same opened reserve action bank:

- ToolGate-style binary execute/skip over a separately proposed crop;
- ToolVision-style question-level with/without-tool necessity tiers;
- Beacon-style mode-adaptiveness and tool-effect diagnostics;
- MED-style call-gain, call-harm, and schema-only attribution;
- BCEA-style answer/abstain/acquire control where its claim-level conformal
  estimand is meaningful, reported separately from task-value utility;
- proxy labels based on immediate forced-answer transitions where required;
- the same source splits and matched call budget;
- reporting action-selection error, gate error, rescue, and easy-case harm;
- preferably a second backbone or an independently preregistered reserve/test
  population.

This comparator must not be inserted into the ongoing one-shot branch or used
to change its policy. It should be a separately declared follow-up, because the
present implementation and formal evaluator were frozen before calibration.

## Primary sources

- ToolGate: https://arxiv.org/abs/2606.03054
- Beacon: https://arxiv.org/abs/2607.28595
- ToolVision: https://arxiv.org/abs/2608.08907
- MED: https://arxiv.org/abs/2602.01334
- BCEA: https://arxiv.org/abs/2606.16667
- AdaptVision: https://arxiv.org/abs/2512.03794
- CropVLM: https://arxiv.org/abs/2511.19820
- AVA-VLM: https://arxiv.org/abs/2607.05859
- ToolsRL: https://arxiv.org/abs/2604.19945
- VTool-R1: https://arxiv.org/abs/2505.19255
- UG framework/paper: https://github.com/ExplainableML/ug-framework
- GapSight / Learning to Look Again: https://arxiv.org/abs/2608.21762
- The Illusion of Visual Tool-Use: https://arxiv.org/abs/2608.06270
