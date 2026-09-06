# Factorized Potential Outcomes: claim map v1

Status: frozen before Phase-C held-out access. This document constrains paper
language; it does not turn a development result into a publication claim.

## Defensible method claim

The narrow method claim is:

> For one-step visual acquisition with real same-state STOP/CONTINUE sibling
> outcomes, decompose bounded task-score change into remaining reward mass,
> rescued fraction, and harmed fraction. Train a pre-action VLM selector with
> dense current-answer supervision and mass-weighted asymmetric conditional
> supervision, then apply visual cost only at policy time.

The exact technical distinction is the combination of:

1. realized official task rewards rather than entropy or teacher-forced NLL;
2. a fixed pre-action state and a fixed outcome-blind proposed visual action;
3. retention of beneficial, harmful, and neutral sibling pairs;
4. the bounded-reward identity
   `Y1-Y0=(1-Y0)r-Y0h`, including soft metrics such as ANLS;
5. dense risk supervision on every pair, with conditional rescue/harm losses
   weighted by their observed reward mass;
6. an `O(1)` selector that does not execute the proposed crop before deciding;
7. independently frozen multi-domain, multi-seed, matched-cost evaluation.

The method predicts task effect, not a cost-dependent label. Cost is applied
only by a threshold or budgeted ranking policy. In the registered experiment
all CONTINUE actions have unit cost, and exact call counts are used to compare
ranking quality without held-out threshold fitting.

No individual item above is claimed as novel in isolation. The paper claim is
eligible only if the frozen Phase-C report establishes an empirical advantage
over Outcome-only SFT, direct signed-gain SFT, and uncertainty gates.

## Claims that are forbidden

Do not claim any of the following:

- the first visual tool-use policy or answer-versus-tool gate;
- the first paired with/without-tool supervision;
- the first counterfactual, potential-outcome, treatment-effect, rescue, or
  harm decomposition;
- the first cost-aware visual acquisition method;
- the first method to learn when or where to crop;
- a general causal effect outside the registered sibling intervention;
- continuous region proposal, multi-turn planning, or online adaptation;
- formal generalization from HRBench equivalent to the fresh-source guarantee
  of ChartQA and DocVQA;
- superiority inferred from Phase B, training loss, oracle headroom, a single
  seed, or an unregistered frontier point.

## Nearest-work distinction table

| Work family | Material overlap | Remaining distinction of this method |
|---|---|---|
| TARNet/CFR-style potential outcomes | shared representations and multiple potential-outcome heads | visual acquisition target is a bounded task-reward mass identity with explicit rescue/harm conditionals and a cost-separated deployment rule |
| ToolVision-style capability-aligned supervision | with/without-tool benefit and evidence-gain filtering | every fixed same-state sibling, including harm and neutral outcomes, is retained; the deployed selector predicts action effect before crop execution |
| MED-style diagnosis | decomposes tool-induced gain and harm | MED is checkpoint-level attribution; this method uses the decomposition as post-training supervision for a deployable selector |
| GapSight / loss-gap crop routing | offline candidate execution, utility labels, preserve/review, rescue/harm analysis | uses realized official task reward rather than NLL/margin proxy, does not collapse to only a proxy-best crop, and tests mass-factorized supervision against matched Outcome/direct-gain controls |
| The Illusion of Visual Tool-Use | fixed-prefix observation intervention and visual evidence gain | diagnostic causal audit rather than this bounded-reward post-training objective and pre-action effect selector |
| AdaptVision / VTool-R1 / visual-tool RL | adaptive visual acquisition trained from outcome or efficiency reward | current method is supervised, one-step, fixed-action, explicitly factorized, and does not use online RL |
| Entropy/UG-style evidence selection | uncertainty-based decision or post-acquisition search | selector is trained on realized task effect and makes the call without executing all candidate crops |

Primary-source URLs currently registered in the repository include Shalit et
al. (ICML 2017), AdaptVision, VTool-R1, ToolVision, MED, GapSight, and The
Illusion of Visual Tool-Use. Before submission, titles, versions, conference
status, and exact quotations must be checked again against the primary papers.

## Evidence-to-claim map

| Proposed statement | Required authoritative evidence |
|---|---|
| The factorization learns more useful rankings than dense imitation | Factorized minus Outcome-only primary paired delta is positive on at least two domains, with one source-cluster CI lower endpoint above zero |
| The decomposition adds value beyond direct counterfactual preference | Factorized minus direct-CF is positive on at least two domains and the frontier is not explained only by a different call count |
| The method is competitive with uncertainty | on every successful domain, it is no worse than the strongest entropy/confidence/margin gate by more than the registered 0.5pp tolerance |
| The selector uses image-question-region semantics | all question/image/region shuffle gates pass under the frozen thresholds |
| The method is visually efficient | proposed-crop executions remain zero during scoring and accuracy-cost gains occur at fixed call count |
| The result generalizes | the registered direction holds on at least two of ChartQA, DocVQA, and HRBench; HRBench's weaker allocation guarantee is disclosed |

If any required evidence is absent, the corresponding sentence must be removed
or rewritten as a limitation rather than softened with unregistered secondary
metrics.

## Packaging after the formal result

### If the machine decision is GO

The paper center may be the exact reward-mass factorization and its explanation
of why neutral-heavy paired data defeats direct gain preference. The primary
table, source-cluster confidence intervals, accuracy-cost frontiers, and all
three semantic controls must appear in the main paper. Phase B is motivation
only. Additional experiments may replicate the already frozen claim but may
not tune on the consumed held-out set.

### If the machine decision is NO_GO

Do not present the factorization as a successful CVPR/ICCV/ECCV method. Preserve
the negative evidence and diagnose which registered condition failed. Any new
method must receive a new estimand and new untouched evaluation allocation; the
Phase-C outcomes may only be used as retrospective evidence. A post-hoc rate,
lambda, seed, ablation threshold, or domain subset cannot rescue the claim.

## Current status

Development Phase B was a marginal transition pass, not paper evidence. The
three Phase-C selector seeds are still being trained and the held-out sibling
outcomes remain unopened. Therefore every positive empirical sentence remains
provisional until the immutable formal report is available.
