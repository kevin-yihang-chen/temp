# Factorized Phase-C one-shot formal protocol v1

## Purpose

This transaction decides whether **Factorized Potential-Outcome Visual
Acquisition** is supported as the final method candidate. It is not another
development sweep. The protected sequential outcomes are read once, after all
three selector seeds, code, hashes, metrics, controls, ablations, and decision
rules have been frozen.

The deadline for the final method decision is **2026-09-13 23:59 HKT**.

## Frozen method and controls

The state contains the original image, question, and exactly one already
acquired crop. The only deployable decision is `STOP` or `CONTINUE` with the
single outcome-blind `sequential-opposite-ug-v1` proposed crop. The proposed
crop is not executed to produce selector features.

Three matched Qwen2.5-VL-3B selectors are trained for each seed `17/29/47`:

1. `outcome_only`: absolute branch-reward imitation;
2. `counterfactual_utility`: direct signed-gain preference;
3. `factorized_potential_outcomes`: error mass, conditional rescue, and
   conditional harm, combined as `e*r-(1-e)*h`.

All arms use the same outcome-independent 3,072-step, domain-balanced schedule,
optimizer, trainable vision merger, final language block, head width, and pixel
budget. The three trained seeds are evaluated as three independent deployment
runs. Scores are not ensembled before selection.

The formal report also includes Answer-only, deterministic Random gate,
entropy/confidence/margin gates, the two SFT controls, and a privileged Oracle
gain ranking. The strongest uncertainty comparator is the maximum observed
among entropy, confidence, and margin at the registered primary call rate; this
is a conservative comparator, not a selector choice.

## Protected domains

The allocation report is
`data/factorized-phase-c-v1/allocation.report.json`, SHA-256
`3bd94f0cba17b6ea476003d91278370026dd5a7d53bdb88b9d39e8df8efedc08`.
The held-out manifests contain:

- ChartQA: 512 states, new source/RGB relative to historical manifests;
- DocVQA: 522 states from 128 complete documents, new source/RGB;
- HRBench: 92 states from 20 image groups, image-disjoint from Phase-C train.

Allocation used no model outcome. Until the access ledger is created, the
held-out sequential outcomes remain unopened. HRBench inference uses image ID,
not question row, as the bootstrap cluster.

## Frozen policy and statistics

For each seed and domain, selection is exact top-count at call rates
`0/10/25/50/75/100%`. The primary point is 25%; visual cost is swept only in
evaluation at lambda `0/.025/.05/.1/.2`, with `.05` primary. Every policy pays
one incremental visual call when it continues.

Metrics include task score, accuracy gain, incremental and total visual cost,
net utility, answer-selection probability, beneficial-call precision/recall,
harmful and unnecessary call rates, and binary top-1 action regret. The report
contains the full accuracy-cost frontier and score-quantile versus observed-gain
plots.

Primary policy differences first average the paired per-state effect across the
three independently deployed seeds, then use 20,000 paired cluster bootstrap
resamples. ChartQA and DocVQA resample `source_id`; HRBench resamples `image_id`.
No seed, threshold, call rate, lambda, or method is selected after access.

## Semantic controls

Only the proposed Factorized selector is evaluated under three deterministic,
outcome-independent SHA-256 component derangements. Whenever at least two
component values exist, every recipient is assigned a donor whose registered
component differs; donor reuse is permitted when a one-to-one permutation is
impossible because component frequencies differ. A constant component remains
unchanged and therefore fails the semantic ranking/call-set gate rather than
aborting the one-shot report:

- question and model-prompt shuffle;
- original-image shuffle (which also changes the already-acquired visual crop);
- proposed-region geometry/action/cost shuffle.

For each ablation, the original must have positive mean 25%-rate task-score
delta across domains and seeds. In at least two domains, mean Spearman score
correlation must be `< .98` and mean selected-set Jaccard must be `< .95`.
This prevents a positive policy result that is unchanged by question, image, or
region information from being called spatial semantic utility learning.

## GO rule

`GO` requires all of the following:

1. Factorized minus Outcome-only mean task-score delta is positive in at least
   two of three domains;
2. its paired source-cluster 95% CI lower endpoint is positive in at least one
   domain;
3. on every domain counted as positive versus Outcome-only, Factorized is no
   worse than the strongest uncertainty baseline by more than 0.5 percentage
   points;
4. all three semantic controls pass.

Any failed condition yields `NO_GO`. A `NO_GO` does not authorize held-out
reuse, post-hoc seed/rate selection, 7B, RL, or method changes on these outcomes.

## Irreversible execution

Before any plan is frozen, a one-GPU runtime smoke must load real completed
selector bytes, recreate the sparse trainable topology, and execute all three
methods plus all Factorized semantic modes on two previously opened development
examples. The smoke must prove finite scores, two observed images, and zero
proposed-crop execution; its report hash is part of the formal plan.

`freeze_factorized_phase_c_formal.py` validates the nine completed selectors,
matched schedules, engineering gates, monitor-only reports, repository
revision, runtime smoke, code bytes, and all input hashes without reading
held-out manifest bytes. It writes a plan whose hash binds every formal output
path.

The Slurm worker requests four RTX 4090 GPUs, `--mail-type=ALL`, and
`--no-requeue`. On start, `execute_factorized_phase_c_formal.py` validates every
frozen byte and writes `access-ledger.json` before any held-out manifest byte is
read. It then:

1. runs four deterministic rollout shards per domain and strictly merges exact
   decision coverage;
2. scores seeds 17/29/47 in parallel on three GPUs, without executing the
   proposed crop;
3. runs the frozen evaluator, figures, and `GO_NO_GO.md`;
4. writes either immutable completion evidence or a failure-after-access
   record.

There is no automatic retry or requeue after protected outcomes are opened.
