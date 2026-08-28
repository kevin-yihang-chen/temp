# Literature position and novelty boundary

Snapshot date: 2026-08-28

This note records the current positioning decision. It is not a complete
survey and should be refreshed before submission.

## Closest current work

1. **Training-free Uncertainty Guidance for Complex Visual Tasks with MLLMs**
   (ECCV 2026) evaluates candidate crops by executing them and selecting the
   observation with the lowest response uncertainty. It is the direct
   entropy-search baseline and motivates testing whether lower post-action
   entropy is worth its execution cost.
   <https://arxiv.org/abs/2510.00705>
2. **VTool-R1** (ICLR 2026) trains interleaved visual tool use with
   outcome-based reinforcement learning. It learns an end-to-end trajectory
   policy, rather than an explicit pre-action counterfactual value model.
   <https://arxiv.org/abs/2505.19255>
3. **What Does Vision Tool-Use Reinforcement Learning Really Learn?** (MED,
   ICML 2026) decomposes observed tool-induced performance into correction
   gains and harms and finds that current crop-and-zoom RL mainly reduces harm.
   This is very close to our empirical rescue/harm decomposition, but MED is a
   checkpoint-level diagnostic rather than a learned pre-action acquisition
   policy.
   <https://arxiv.org/abs/2602.01334>
4. **CropVLM** (CVPR 2026 GRAIL-V workshop) learns a low-cost crop policy with
   reinforcement learning and reports out-of-domain transfer. It raises the
   bar for learned zoom selection, but does not make calibrated
   task-improvement-minus-cost prediction the central object.
   <https://arxiv.org/abs/2511.19820>
5. **AdaTooler-V** (Findings of ACL 2026) is the closest adaptive-tool-use
   training baseline. It labels each query with a Tool Benefit Score computed
   from eight teacher runs with tools and eight without tools, then uses that
   query-level score to rescale GRPO rewards. It already establishes that
   outcome improvement should supervise when-to-call. Our remaining
   distinction must therefore be action-specific and pre-execution: predicting
   *which* unexecuted observation will rescue or harm the current model, with
   explicit deployment-time cost and no 72B teacher requirement.
   <https://arxiv.org/abs/2512.16918>
6. General value-of-information work formalizes whether the expected utility
   of an observation exceeds its acquisition cost, including recent work on
   clarification questions. This supplies decision-theoretic precedent but is
   not a visual acquisition method.
   <https://arxiv.org/abs/2601.06407>
7. **AdaptVision** learns adaptive coarse-to-fine visual acquisition with an
   explicit visual-token efficiency objective and decoupled tool/answer RL
   advantages. It already establishes adaptive when-to-crop under a cost proxy;
   our distinction cannot be generic adaptive acquisition. AdaptVision starts
   from a deliberately compressed image and optimizes an end-to-end policy,
   whereas our target is action-level improvement over the same model's
   answer-now state, calibrated in task units under a user-specified execution
   cost.
   <https://arxiv.org/abs/2512.03794>
8. **Learning to Focus and Precise Cropping** (CVPR 2026) finds that cropping
   agents can invoke tools formally while relying weakly on crop contents. It
   creates a low/high-resolution information gap and adds grounding loss to
   force crop dependence. This makes "tools may be perfunctory" and
   "information gain improves cropping" prior claims; our sibling bank must
   instead quantify which concrete acquisition changes the answer and whether
   that change was worth its cost.
   <https://arxiv.org/abs/2603.27494>
9. **Do Multimodal Agents Really Benefit from Tool Use?** compares tool-enabled,
   tool-free, format-only, and result-only variants of Thyme and DeepEyesV2. It
   finds little consistent aggregate improvement and small tool-only solved
   sets. This is the closest reliability/negative-evidence paper: a submission
   from this project cannot rely only on showing unnecessary tool calls. The
   remaining empirical separation is exhaustive same-state action siblings,
   explicit induced harm and net acquisition utility, pre-action prediction,
   and source-disjoint prospective confirmation.
   <https://arxiv.org/abs/2606.02357>

## Defensible contribution boundary

The paper should not claim that it is the first method to use task improvement
to learn when to invoke visual tools; AdaTooler-V already does this at the
query level. It also should not claim to be the first method to learn where to
zoom. The defensible claim is narrower:

> Learn a calibrated, pre-execution estimate of counterfactual task-success
> gain for each *candidate visual action*, explicitly subtract acquisition
> cost, and use sibling rollouts to supervise both rescue and harm before any
> candidate action is run at deployment.

The empirical contribution is equally important:

> Post-action entropy search can improve raw accuracy while destroying net
> utility, and current learned policies fail primarily through false-positive
> stopping and distribution-sensitive regional ranking.

This connects the UG assumption (uncertainty as relevance) to the MED finding
(tool correction versus tool harm), then turns the decomposition into a
decision rule rather than only a retrospective analysis. Relative to
AdaTooler-V, the method must demonstrate region/action discrimination,
model-specific counterfactual supervision, and cheaper deployment or training
rather than merely adaptive tool invocation.

The two completed DocVQA confirmations now show that this distinction is real
but technically unresolved: the counterfactual oracle remains strongly
positive while both pre-action policies transfer negatively. That evidence is
consistent with MED and the 2026 tool-benefit study, so negative results alone
are no longer sufficient novelty. A top-tier method claim now depends on the
fresh TextVQA confirmation or a materially stronger invariant stopping method
tested on another untouched bank.

## What a top-tier version still needs

- a token-level question-region interaction that improves within-state action
  ranking over random on multiple domains;
- calibrated stopping that transfers without target-label threshold tuning;
- at least one untouched formal benchmark where utility is positive and its
  source-cluster confidence interval excludes zero;
- cost curves, strong RL/tool-use baselines, entropy-search accounting for all
  executed candidates, AdaTooler-V-style tool-benefit supervision, and
  rescue/harm ablations;
- ideally, integration of the learned counterfactual value as a process reward
  or critic for tool-use post-training, showing data-efficiency or safety gains
  beyond a standalone gate.

Until these are met, the project has a publishable problem and rigorous
negative evidence, but not yet a CVPR/ICCV/ECCV-level method result.
