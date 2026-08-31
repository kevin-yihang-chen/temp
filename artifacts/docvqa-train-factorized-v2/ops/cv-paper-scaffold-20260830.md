# CV paper scaffold: harm-controlled counterfactual visual acquisition

Status: updated after the independent DocVQA calibration PASS and the frozen
formal positive-point/interval-FAIL result. The formal population is now opened
and may be used only for locked reporting and diagnostics, never for revision
selection or a renewed claim on the same population.

## Working title

**Prospectively Harm-Controlled Counterfactual Action Value for Selective Visual
Acquisition**

Optional subtitle: **Beyond Entropy for Tool-Using Vision-Language Agents**

## Precise thesis

Visual-tool policies need to predict the signed downstream task effect of a
specific acquisition before executing it. Entropy after acquisition, binary
gating of an already proposed call, and trajectory-level outcome reward answer
related but different questions. Paired action siblings provide direct
supervision for joint stop/action value; an independent fixed-sequence gate
tests whether its sparse high-value tail satisfies predeclared induced-harm and
negative-call constraints before one-shot evaluation. This is not a conformal
risk or coverage guarantee.

## Abstract skeleton

Vision-language agents can zoom or crop to acquire additional evidence, but a
tool call may be unnecessary, harmful, or confidently misleading. Existing
approaches select observations using uncertainty, train tool trajectories from
outcome rewards, gate a call already proposed by an agent, decompose aggregate
tool gain and harm, or conformally calibrate answer/abstain/acquire decisions.
We study a more specific decision: before observing any candidate crop, should
the agent answer now, and if not, which single crop has positive task value
after cost? We build
paired sibling rollouts that execute answer-now and every candidate action from
the same state, yielding signed rescue and harm supervision. A factorized model
predicts baseline error, action-conditional rescue, and induced harm using only
pre-action features, while a source-level fixed-sequence procedure calibrates a
non-degenerate deployment threshold. On DocVQA ranker-training data, the OOF
policy calls tools on 1.66% of decisions, obtains 0.2205 task gain per call, and
has positive mean utility 0.00283 with a 95% source-bootstrap interval
[0.00181, 0.00390]. A prospective TextVQA calibration does not pass the frozen
deployment gate, revealing that sparse tool-benefit tails need not transfer.
On 2,500 independent DocVQA calibration sources, the frozen sequence selected
threshold -0.0080312 at 1.4857% calls and +0.1310 percentage-point utility while
passing both registered risk constraints. On 3,500 one-shot formal sources, the
same policy obtains +0.1304 percentage-point source-balanced utility at 1.4393%
calls, but its registered 97.5% whole-source interval
[-0.0203, 0.2970] percentage points crosses zero; the formal gate therefore
fails. Our evaluation charges every acquisition and
compares matched-budget entropy, random, fixed-crop, and exhaustive uncertainty
baselines. These results position counterfactual action value as a
risk-controlled acquisition layer rather than a generic claim that visual-tool
use is universally beneficial.

## Contributions that remain defensible

1. **Direct signed action supervision.** A paired sibling bank measures the
   task effect of answer-now and each crop at the same pre-action state,
   separating rescue, harm, and no effect.
2. **Joint where/when value.** A factorized model ranks concrete crops and
   abstains based on cost-adjusted action value before acquiring any candidate.
3. **Prospective action-harm gate.** Whole-source OOF fitting, fixed-sequence
   calibration, and a one-shot formal decision prevent threshold selection on
   formal outcomes and explicitly constrain induced harm and negative calls.
4. **Cost-faithful diagnostics.** Matched-call controls and exhaustive UG-style
   search are charged for actual candidate executions; entropy/task discordance,
   crop-ranking rescue, stopping, and oracle regret are reported together.

Do not claim first pre-call gating, first answer-or-crop learning, first visual
tool gain/harm decomposition, first risk-calibrated acquisition, first visual
tool credit assignment, or general cross-benchmark effectiveness.

## Evidence ledger

### Established before DocVQA calibration

- DocVQA OOF: 13,580 decisions from 3,500 sources.
- OOF tool rate: 0.0165685.
- OOF task gain: 0.0036537.
- OOF utility at lambda 0.05: 0.0028252; source-bootstrap 95% interval
  [0.0018100, 0.0038967].
- Gain per selected call: 0.2205182.
- Any helpful crop exists in 1,015 / 13,580 states (7.47%).
- Learned top-crop rescue within helpful states: 50.84%; random crop: 39.51%.
- Post-action entropy/success correlation: 0.214; entropy action-selection
  regret: 0.01789.
- TextVQA prospective calibration: registered FAIL,
  `no_non_degenerate_safe_threshold`; formal remained sealed. The best safe
  near-floor point had source call rate 1.25% and utility 0.0009917, below the
  frozen 0.001 floor.

### Established by independent DocVQA calibration

- 9,762 decisions from 2,500 source-disjoint calibration sources.
- Frozen selected threshold: -0.00803116602423646.
- Source-balanced call rate: 0.0148571.
- Source-balanced utility: 0.00131013 (+0.1310 percentage points).
- Source-balanced induced harm: 0.0005335; net-negative call mass: 0.010358.
- Calibration audit PASS with `formal_outcomes_used=false`, bound to code
  revision `2042c2e1f01f28b24b38d78b7d9c9bbfccb78232`.

### Established by the frozen DocVQA formal evaluation

- DocVQA 3,500-source formal utility: 0.00130439; 97.5% interval
  [-0.00020303, 0.00296951]; preregistered FAIL.
- Question-weighted utility: 0.00103988; source-balanced call rate: 0.01439297.
- Strong positive paired differences over random, unconditional, and charged
  exhaustive UG controls.
- No positive paired lower endpoint over no call or matched-budget entropy gate
  with the learned crop.

### Established by the frozen reserve comparator

- On 688 reserve sources and 2,585 decisions, signed Policy A has
  source-balanced utility 0.0028453 versus 0.0024627 for the frozen
  ToolGate-style Policy B.
- The registered paired difference is +0.0003826 with 95% interval
  [-0.0012354, 0.0019681]; the lower-endpoint rule fails, so this is a
  null/negative ablation rather than evidence that A is superior.
- The gates disagree on only 0.781% of source-balanced decisions. The shared
  proposal misses a helpful crop in 49.27% of helpful states, identifying
  action selection as the larger observed bottleneck.
- A separate opened-development Qwen2.5-VL-7B/H800 mechanism diagnostic passes
  its four frozen replication conditions, but it is not an independent policy
  validation.

### Pending on unopened follow-up populations

- A proposal-improved joint model developed only on non-ScreenQA populations,
  followed by frozen ScreenQA calibration and one-shot formal evaluation.
- ToolVision-style necessity and Beacon-style mode/tool-effect diagnostics if
  they can be implemented without reopening candidate selection.

### Unsupported without more evidence

- Cross-benchmark general effectiveness.
- Deployable-policy generality beyond Qwen2.5-VL-3B; the 7B evidence is
  mechanism-only.
- Multi-turn or arbitrary-tool effectiveness.
- SOTA benchmark accuracy.
- Causal identification beyond the enumerated deterministic sibling action set.

## Main-paper structure

### 1. Introduction

- Show one state where entropy decreases but task correctness worsens.
- Separate three decisions: post-acquisition evidence scoring, binary gating of
  a pending call, and pre-acquisition joint stop/action value.
- State the narrow deployment question and the need to model both rescue and
  harm.
- Preview prospective calibration, including the negative TextVQA gate as a
  boundary rather than hiding it.

### 2. Related work

- Training-free uncertainty guidance and visual re-perception.
- RL-trained visual tool use and adaptive visual-token acquisition.
- Pre-call tool gating and proxy supervision.
- Value of information, selective prediction, conformal evidence acquisition,
  and the distinction between guarantees and prospective harm gates.
- Counterfactual/paired evaluation and visual-tool harm diagnostics.

### 3. Method

1. State, enumerated visual actions, answer-now, and costed utility.
2. Sibling rollout construction and signed task delta.
3. Factorized baseline-error/rescue/harm heads.
4. Joint crop argmax and abstention threshold.
5. Source-grouped OOF training.
6. Fixed-sequence risk calibration and formal pass rule.

### 4. Experimental protocol

- Qwen2.5-VL-3B, frozen UG-grid crops, deterministic decoding.
- Source-disjoint identity allocation and RGB collision audit.
- Ranker/calibration/formal access sequence.
- Baselines: answer-now, random, entropy, fixed crop, learned/random crop
  crossings, exhaustive UG charged four calls, and oracle.
- Whole-source bootstrap and exact cost accounting.

### 5. Results

- Development mechanism diagnostics, clearly labeled non-independent.
- TextVQA negative prospective calibration.
- DocVQA calibration PASS with the exact frozen threshold and risk diagnostics.
- DocVQA formal positive point estimate with registered interval FAIL.
- Reserve ToolGate-style comparator: positive point difference but registered
  95% lower-endpoint FAIL; report as a null/negative supervision ablation.

### 6. Analysis and limitations

- Why useful calls are sparse.
- Entropy reduction versus task value.
- Crop-ranking error versus stopping error.
- Distribution shift of high-value tails.
- Fixed one-step grid, a 3B deployment policy, external gate, and chosen
  lambda; separate 7B evidence is mechanism-only.

## Planned figures and tables

1. **Figure 1:** one pre-action state branching into answer-now and four sibling
   crops; training sees all branches, deployment executes at most one.
2. **Figure 2:** entropy change versus task delta, highlighting confidently
   wrong and task-improving quadrants.
3. **Figure 3:** utility/call/harm frontier with the frozen calibration sequence
   and stopping point.
4. **Figure 4:** source-disjoint evidence ladder: OOF, TextVQA calibration FAIL,
   DocVQA calibration PASS, and DocVQA formal interval FAIL.
5. **Table 1:** UG, VTool-R1, AdaptVision, CropVLM, ToolsRL, ToolGate,
   Beacon, ToolVision, MED, BCEA, and this method by intervention point,
   supervision, candidate coverage, joint action choice, cost, and risk
   semantics.
6. **Table 2:** formal matched-budget policies and 97.5% source intervals.
7. **Table 3:** error/rescue/harm, action-feature, and binary-proxy ablations.

## Likely reviewer objections and required answers

- **Only 1%-3% calls.** Show the full development cost/call frontier, all-call
  negative utility, gain per selected call, and formal matched-budget baselines.
- **ToolGate/AdaptVision/Beacon/ToolVision already decide when to call; MED
  decomposes gain/harm; BCEA calibrates evidence acquisition.** Do not claim
  generic necessity, harm awareness, paired with/without-tool supervision, or
  generic risk-calibrated acquisition. Center the complete same-state candidate
  sibling bank, signed effect for every crop including harm/zero outcomes,
  joint crop ranking, and prospective fixed-sequence action-harm gate; include
  the reserve comparator family.
- **One benchmark/backbone.** A DocVQA pass is necessary but not sufficient for
  broad generality. Add an independently declared benchmark or backbone before
  making that claim.
- **Offline counterfactual bank is not a free online tool.** State clearly that
  siblings are training/evaluation supervision; deployment uses one pre-action
  score pass and at most one crop.
- **Lambda is arbitrary.** Report the frozen lambda=0.05 primary result plus
  development-only cost sensitivity and physical latency/token counts without
  retuning formal policy.
- **Fixed crop grid is limited.** Treat continuous proposals and multi-step
  acquisition as future work, not implied capability.

## Result-dependent publication decision

### Counterfactual route: DocVQA calibration and formal pass

Submit as a selective visual-acquisition method with one large prospective
positive result, one transparent negative calibration boundary, and the reserve
ToolGate-style comparator. Before claiming generality, prioritize one new clean
benchmark or second backbone replication.

### Current route: calibration passes but formal fails

Do not tune on formal. Report the positive point estimate and failed 97.5%
lower-endpoint clause unchanged. Reframe around instability of sparse
visual-tool value under distribution shift; a top-tier submission now requires
broader predeclared replications and a stronger diagnosis of sparse-tail
precision versus recall. The completed reserve comparator has a positive but
inconclusive A-minus-B point estimate and shows that shared crop-proposal error,
not another gate threshold, is the next method target.

### If calibration fails

Close the policy branch. The remaining package is an empirical study showing
why OOF visual-tool headroom fails independent risk calibration across TextVQA
and DocVQA. That can be scientifically useful but is below the intended main-
track bar without broader models, benchmarks, or a new method developed on
untouched populations.

## Authoritative local evidence

- `artifacts/docvqa-train-factorized-v2/ranker-training/factorized-oof-v1/report.json`
- `artifacts/docvqa-train-factorized-v2/frozen-candidate/model.json`
- `docs/textvqa_factorized_v2_independent_calibration_result_2026-08-29.md`
- `docs/docvqa_train_factorized_v2_preregistration.md`
- `artifacts/docvqa-train-factorized-v2/ops/literature-collision-repositioning-20260830.md`
- `artifacts/docvqa-train-factorized-v2/ops/reserve-toolgate-comparator-protocol-20260830.md`
- `artifacts/docvqa-train-factorized-v2/ops/reserve-toolgate-comparator-result-20260901.md`
- `artifacts/screenqa-train-factorized-v1/ops/backbone-7b-result-and-next-step-v1.md`
