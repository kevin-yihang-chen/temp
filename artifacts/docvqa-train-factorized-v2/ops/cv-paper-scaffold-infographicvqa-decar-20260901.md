# Prospective CV paper scaffold: InfographicVQA DECAR

Status: written on 2026-09-01 while full generation job `200130` was running,
before official-train OOF predictions or endpoints existed and while official
validation/test remained sealed.  Every result slot below is intentionally
blank.  This document cannot turn a negative registered decision into a paper
claim.

## Working identity

Working title:

> **Signed Task Effects and Prospective Harm Control for Selective Visual
> Re-Reading**

Method name: **DECAR**, used descriptively for the frozen loss-distilled
where / task-effect-and-harm-aware when architecture.  **Beyond Entropy** may
remain a subtitle or empirical diagnostic, but it is not the novelty claim.

One-sentence research question:

> Given a fixed visual-action bank and a low-resolution global state, can a
> policy use model-specific loss gaps to identify *where* to look while using
> realized signed task effects to decide *whether* looking is worth its cost,
> with prospective control of crop-induced harm?

## Collision-safe thesis

GapSight already probes an offline crop bank, learns from model-specific
answer-NLL or option-margin gaps, jointly predicts preserve/review and a
continuous crop, and reports rescue and harm.  MED already decomposes
tool-induced gain and harm.  ToolGate controls execution of a pending call;
Beacon and ToolVision learn necessity-aware tool behavior; BCEA calibrates a
different answer/abstain/acquire risk estimand.  Therefore none of the
following is novel alone: crop probing, loss-gap supervision, joint when/where
routing, adaptive visual tokens, tool-harm measurement, pre-call gating, or
risk-aware evidence acquisition.

The only defensible methodological center is the conjunction:

1. retain ANSWER-NOW and **every fixed candidate action** as auditable
   same-state siblings;
2. retain each candidate's realized signed task-score effect, including rescue,
   zero effect, and harm, instead of reducing the bank to only a proxy-best crop
   and a binary review label;
3. combine loss-distilled action ranking with a separate task-effect and harm
   model for joint stop/action selection under explicit execution cost; and
4. evaluate with source-grouped nested OOF predictions and an outcome-free
   prospective harm gate before any sealed endpoint is opened.

This is an operational and evaluation distinction, not a claim that generic
visual re-reading is new.

## Exact hypothesis tested by the frozen branch

At one of the registered nominal call rates `0.5%`, `1%`, `2%`, `5%`, or
`10%`, DECAR must simultaneously:

- make at least 100 calls spanning at least 50 source components;
- have a strictly positive 95% whole-source-bootstrap lower endpoint for
  source-balanced utility;
- have a higher source-balanced utility point estimate than every feasible
  non-oracle matched-budget baseline;
- have higher utility than `task_value_only`, `loss_only`, and `no_harm_head`
  at the identical call count;
- have no greater induced harm or negative-utility-call mass than both the
  no-harm ablation and the strongest feasible non-oracle baseline; and
- pass every source, leakage, action-coverage, cost, tie, serialization, and
  bootstrap audit.

Passing some clauses is not partial success for the method claim.  Failure of
any clause yields `decar_not_advanced`, leaves validation sealed, and triggers
only the registered failure decomposition.

## Abstract skeleton, with no result laundering

Vision-language models can re-read a local crop when a compressed global view
loses answer-critical detail, but additional evidence can also waste compute or
damage an already correct answer.  Recent routers learn model-specific crop
utility from answer-loss gaps, while audits show that apparent visual-tool
gains often coexist with harmful or causally ineffective calls.  We study a
narrower deployment problem over a fixed candidate bank: predict the value of
ANSWER-NOW and every concrete crop before executing any crop, then execute at
most one action.  DECAR uses answer-loss gaps to learn candidate ranking but
uses complete same-state sibling outcomes to learn signed task effect and harm
for stopping.  We evaluate outcome-free, source-grouped nested-OOF predictions
with matched execution budgets, whole-source bootstrap intervals, and a
predeclared harm-aware advancement rule on 23,946 InfographicVQA training
questions from 2,204 source components.  **[Insert the registered train and
sealed-validation decisions verbatim; if DECAR does not advance, this abstract
is not valid as a positive method abstract.]**

## Experimental population and intervention

- Backbone: pinned Qwen2.5-VL-7B-Instruct, bfloat16, deterministic decoding.
- Population: 23,946 official-train questions, 2,204 source-connected
  components, 4,406 images; official validation/test sealed.
- Pre-action input: original-image global representation, question
  representation, region-pooled candidate features, crop geometry, and 16
  frozen scalar signals.  No target, answer correctness, crop outcome, or
  post-action entropy is an inference feature.
- Sibling action bank: ANSWER-NOW plus four frozen UG-grid crops from the same
  pre-action state.
- Cost: `0.05` per executed crop; exhaustive four-crop selection pays `0.20`.
- Prediction protocol: five outer source folds, nested four-way inner
  cross-fitting, 65 deterministic 200-epoch fits, outcome-free serialized OOF
  predictions, then one evaluation join.
- Uncertainty: exactly 20,000 shared whole-source bootstrap resamples, seed
  `20260917`.

## Required comparisons

### Learned variants at identical call count

1. `decar`: loss-distilled where plus three-way rescue/neutral/harm when.
2. `task_value_only`: signed task-delta where, with the same nested triage.
3. `loss_only`: proxy loss-gap where with no task-effect/harm triage.  Describe
   this only as a **GapSight-style fixed-bank loss-gap ablation**, not as a full
   reproduction of GapSight's continuous-box router.
4. `no_harm_head`: loss-distilled where plus rescue-versus-other stopping,
   without an explicit harm class or veto.

### Non-learned matched-budget baselines

- ANSWER-NOW;
- entropy-gated exact uniform random crop;
- entropy-gated fixed `ug-grid-00` crop;
- entropy-gated four-crop UG selection charged for four executions; and
- charged exhaustive UG over all questions as a static cost reference.

Task-oracle one-crop and oracle-stopping results are diagnostic ceilings, never
deployable comparators.

### External comparison still required before a top-tier claim

If and only if DECAR advances on train, freeze a same-backbone held-out
comparison against the closest executable external methods.  At minimum:

- GapSight or an author-code-faithful reproduction, including its continuous
  crop and reported token accounting;
- a ToolGate-style execute/skip controller under a shared proposer;
- the frozen UG implementation; and
- always-high-resolution or always-second-view compute controls.

Published numbers from another backbone or 1,000-example subset may provide
context but cannot substitute for a same-population comparison.

## Main result table shells

### Table 1: registered OOF advancement table

For every call rate and policy, report:

```text
policy | nominal rate | actual calls | called sources | executed crops
       | ANLS gain | utility [95% source CI] | gain/call
       | helpful precision | helpful recovery | induced harm
       | harmful-call mass | negative-utility-call mass | audit status
```

Mark exactly one selected operating point only if all six advancement clauses
pass.  Do not bold the highest point estimate if its interval or audit fails.

### Table 2: mechanism ablations at matched calls

```text
variant | where target | when target | harm veto | utility delta vs DECAR
        | action regret | gate FP loss | gate FN loss | induced harm
```

Use paired source-bootstrap intervals for differences even where the train
advancement rule uses point-estimate ordering.

### Table 3: sealed validation and external methods

Create only after a train pass and a new validation protocol.  It must bind the
selected operating point, all code/data hashes, exact external implementations,
execution cost units, and one-shot pass/fail rule before validation outcomes are
read.

## Figure plan

1. **Sibling bank and deployment asymmetry.** Training/evaluation retain
   ANSWER-NOW plus four independently executed siblings; deployment scores all
   candidates but executes at most one.
2. **Proxy versus task effect.** Candidate loss gap against realized ANLS
   delta, partitioned into proxy-correct, proxy-misranked, rescue, neutral, and
   harm regions.
3. **Matched-budget frontier.** Source utility, induced harm, and helpful-state
   recovery versus executed crops, with complete-tie operating points.
4. **Failure anatomy.** Action-choice regret, gate false-positive loss, gate
   false-negative loss, and source call concentration.  Include this figure
   whether the method passes or fails.
5. **Evidence ladder.** Historical opened DocVQA/ScreenQA negatives as
   motivation, registered InfographicVQA train OOF, and sealed validation only
   if opened prospectively.

## Reviewer objections that must be answered

- **"This is GapSight with a discrete crop bank."** Show the exact difference
  between proxy-best binary labels and retained per-action realized task
  effects; require DECAR to beat `loss_only`, then run the external method.
- **"Harm reporting is already in GapSight/MED."** Agree.  The claim is
  prospective harm-conditioned action selection and its registered gate, not
  the discovery or measurement of harm.
- **"The grid handicaps the strongest crop routers."** Treat the fixed bank as
  a controlled action-value experiment.  A positive result still requires an
  external continuous-crop comparison; a negative result may indicate proposal
  limitations rather than disprove adaptive re-reading.
- **"Sparse call rates make the gain trivial."** Report raw calls, called
  sources, gain/call, source concentration, all-call and exhaustive controls,
  and the full cost frontier.  Never hide the no-call baseline.
- **"Train OOF is not held-out evidence."** It is an advancement screen only.
  No paper-level positive claim is allowed without a frozen sealed-validation
  result.
- **"Only one dataset and one backbone."** Even a validation pass supports a
  narrow claim.  A CVPR/ICCV/ECCV main-track submission should add either a
  second clean dataset or a second backbone under a predeclared protocol.

## Result-contingent publication routes

### No train operating point qualifies

The positive DECAR method route closes.  Publish no success language.  Use the
frozen decomposition to identify whether the binding failure is proposal
quality, action ranking, stopping precision, missed rescues, source
concentration, or lack of tool headroom.  A successor must use a new population
and dated protocol.  The remaining artifact is a rigorous negative study, not
yet a three-main-conference method paper.

### Train qualifies but sealed validation fails

Report train advancement and validation failure unchanged.  The main result is
instability of sparse signed action value, not a deployable method.  Do not tune
on validation.  Continue only with a separately frozen dataset/backbone or a
mechanistically different successor.

### Train and sealed validation both pass

The narrow method claim becomes supportable: within the fixed action bank,
separating proxy-based where from task-effect/harm-aware when yields positive
cost-adjusted utility under prospective source-level control.  Before a top-tier
submission, complete the external continuous-crop comparison and one additional
predeclared generalization axis.

## Authoritative inputs

- `infographicvqa-decar-method-protocol-v1.md`
- `infographicvqa-decar-full-generation-freeze-v1.md`
- `infographicvqa-decar-oof-evaluation-freeze-v1.md`
- `literature-collision-repositioning-20260830.md`
- `gapsight-comparator-feasibility-audit-20260901.md`
- `infographicvqa-decar-oof-runtime-benchmark-freeze-v1.md`

Historical DocVQA and ScreenQA results may motivate the question and show prior
failure modes, but they may not be pooled with the registered InfographicVQA
population or used to select its model, operating point, or claim.
