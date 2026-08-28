# Submission-readiness checkpoint — 2026-08-28

Target: CVPR / ICCV / ECCV main conference.

This is an evidence and decision tracker, not a paper claim. Formal failures
remain part of the record and are not overwritten by later experiments.

## Current verdict

The project has a credible problem, a differentiated measurement object, a
reproducible counterfactual data pipeline, and strong oracle headroom. It does
not yet have the transferable positive cost-sensitive method result required
for a competitive main-conference submission. The fresh TextVQA confirmation
produced a positive raw-gain interval but failed its utility interval, so Branch
B below is now active.

| Requirement | Status | Evidence or missing item |
| --- | --- | --- |
| Clear problem | ready | Visual acquisitions can help, harm, or waste cost; entropy reduction is not task improvement. |
| Distinct technical object | provisional | Pre-execution, answer-now-relative value for each sibling visual action, including explicit no-call and cost. |
| Reproducible data/evaluation | ready | Frozen manifests, source/RGB overlap audits, artifact hashes, label-free feature audits, source-cluster bootstrap, and one-shot evaluators. |
| Diagnostic empirical finding | ready | Exhaustive entropy search can improve raw score while having strongly negative utility; oracle sibling value remains positive. |
| Confirmed stopping | partial | ChartQA high-power replication supports when-to-call for a bounded gate; cross-domain action-specific stopping has failed. |
| Learned action ranking | not ready | Development attention ranking improved rescue diagnostics, but formal DocVQA ranking and utility were negative. |
| Positive untouched utility | not ready | No action-specific policy currently has a strictly positive formal confidence interval. |
| Cross-domain/model breadth | not ready | Current action-value evidence is Qwen2.5-VL-3B and benchmark-specific; a second model family is still needed. |
| Strong contemporary baselines | partial | UG-style entropy and random/fixed/oracle controls exist; matched AdaptVision/CropVLM/AdaTooler-V-style comparisons and outcome-only RL are incomplete. |
| Agent post-training contribution | not ready | Counterfactual visual credit has not yet improved tool-use RL under a matched budget. |

## Decision after fresh TextVQA confirmation

Observed outcome: **Branch B**. The frozen policy gained `+0.004675` raw score
with a strictly positive 97.5% interval, but utility was `+0.000032` with
97.5% CI `[-0.003541, +0.003852]`. The pass criterion was not met.

### Branch A — registered pass

A pass requires mean utility above zero and a strictly positive lower endpoint
of the frozen two-sided 97.5% source-bootstrap interval. If it passes:

1. lock and report the simple frozen policy before any additional analysis;
2. run the registered action-ranking, stopping, harm, cost-frontier, and oracle
   diagnostics without changing the primary conclusion;
3. reserve a second untouched benchmark and one additional VLM family for
   replication;
4. compare against matched entropy search, fixed/random crop, absolute
   post-crop reward, and query-level tool-benefit supervision;
5. integrate the fixed counterfactual value as a visual-action critic/process
   reward and compare with outcome-only post-training under equal rollouts and
   GPU budget.

The likely paper story would be a positive selective-acquisition method backed
by exhaustive sibling diagnostics and a cost-aware formal result. One TextVQA
pass alone is necessary but not sufficient for a main-conference claim.

### Branch B — registered failure

If either the point estimate or lower bound fails:

1. close the current semantic-attention policy family on every opened TextVQA
   and DocVQA formal bank;
2. retain the result as evidence that development OOF action value is not a
   reliable stopping certificate under source shift;
3. implement the separately frozen risk-controlled acquisition protocol using
   development/calibration sources only;
4. pre-register the harm tolerance, finite threshold family, calibration unit,
   and non-degenerate utility criterion;
5. evaluate once on a newly reserved source/RGB-disjoint target, never on an
   already opened formal bank for a paper-level claim.

If risk control only achieves safety by making essentially zero calls, the
method branch also fails. At that point the defensible product is a rigorous
diagnostic/resource paper, which is unlikely to be sufficient for the target
venues without a substantially stronger benchmark or training contribution.

## Minimum main-conference evidence package

Before drafting a submission as a method paper, require all of the following:

- at least one prospective untouched test with positive cost-adjusted utility
  and a source-cluster confidence interval strictly above zero;
- a second benchmark or model-family replication with the same qualitative
  conclusion;
- action ranking above matched random/fixed baselines, not only oracle
  headroom or better stopping;
- full accuracy/utility versus visual-cost curves and explicit accounting of
  candidate evaluation cost;
- rescue, induced harm, unnecessary-call rate, gain per call, and oracle regret;
- all negative formal attempts and multiplicity decisions disclosed;
- matched modern visual-tool baselines; and
- either a meaningful post-training improvement or a clear statistical risk
  guarantee that is itself central enough to carry the method contribution.

## Immediate execution order

1. Implement source-level risk-control calibration and synthetic unit tests.
2. Export outcome-independent TextVQA train source partitions and audit decoded
   RGB overlap against every prior bank.
3. Collect the 5,000-source ranker-training bank and 3,000-source calibration
   bank with restart-safe, emailed Slurm jobs.
4. Compare fixed low-capacity, listwise/pairwise ranking, separated call heads,
   and risk-controlled thresholds without touching formal sources.
5. Freeze one policy and evaluate it once on the reserved 5,000-source formal
   bank under a multiplicity-aware source interval.
