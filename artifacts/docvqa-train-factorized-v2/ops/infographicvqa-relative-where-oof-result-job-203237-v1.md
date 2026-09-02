# InfographicVQA relative-where OOF result (job 203237)

Status: official-train source-OOF result; method not advanced. Validation and
test remain sealed.

## Execution and recovery

The combined fit/evaluation job 203099 completed all 20 frozen source-OOF fits
and wrote 23,946 outcome-free prediction rows, but its evaluator failed closed
after bootstrap with a generic frozen-comparator mismatch. It produced no
scientific decision. The frozen recovery reused those predictions byte-for-byte,
moved exact comparator checks before bootstrap, and reported the first exact
difference on failure without adding tolerance or changing any endpoint.

Recovery job 203237 completed on 2026-09-02 HKT with exit code 0, zero restarts,
five seconds of queue wait, and 374 seconds of evaluator runtime. All frozen
comparators matched exactly and all 20,000 paired whole-source bootstrap draws
were reused. Slurm state mail was configured for BEGIN, END, FAIL, REQUEUE, and
related events.

## Frozen decision

`relative_where_train_not_supported`; no operating point qualified.

| Nominal call rate | Calls | Primary source utility (95% CI) | Old DECAR | Privileged NLL teacher | Task oracle | Teacher-action agreement | Oracle-gap closure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5% | 120 | -0.000197 [-0.000464, 0.000098] | -0.000326 | 0.000101 | 0.000374 | 17.4% | 18.5% |
| 1% | 240 | -0.000486 [-0.001178, 0.000083] | -0.000402 | 0.000423 | 0.000794 | 20.0% | -7.0% |
| 2% | 479 | -0.000960 [-0.001941, -0.000040] | -0.001245 | 0.000463 | 0.001279 | 19.3% | 11.3% |
| 5% | 1,198 | -0.002531 [-0.004349, -0.000871] | -0.002767 | 0.002131 | 0.004944 | 22.7% | 3.1% |
| 10% | 2,395 | -0.006502 [-0.009302, -0.003751] | -0.006714 | 0.003770 | 0.007561 | 22.8% | 1.5% |

Every point passed the population, source-disjointness, prediction-leakage,
fixed-call-count, frozen-comparator, and bootstrap audits. Every point failed
the positive-utility lower-bound rule, both paired lower-bound rules against the
old DECAR and task-value rankers, the all-deployable-comparators rule, and the
50% oracle-gap-closure rule. Four of five points also failed the harm rule.

The primary ranker improves the old DECAR point estimate at 0.5%, 2%, 5%, and
10%, but none of those paired improvements has a positive 95% lower bound. Its
exact privileged-teacher action agreement is only 17.4%--22.8%, below the 25%
four-action random reference. In contrast, the privileged teacher itself is
positive at every call rate and the task oracle is stronger still. The remaining
bottleneck is therefore deployable action generalization, not the existence of
useful crops.

## Implementation sanity

The target sign and action alignment are not reversed. The assembled teacher
target is `baseline answer NLL - crop answer NLL`; larger is better. Target
distributions, pairwise comparisons, output columns, and selected action IDs all
use the same frozen `DECAR_ACTION_IDS` ordering. Training loss falls sharply in
each outer fold, while held-out-source agreement remains at or below chance.
This is evidence of source-specific overfit, noisy fine-grained teacher labels,
or missing invariant localization features rather than an optimization-sign bug.

## Bound artifacts

```text
fit predictions:   94e292d33fe9e55e9d50d53eef5f2590c4d5c751567feb2fe736867fc7dd6b6b
fit audit:         256c34ad9d370107950f9edf915c4a65337bf35df0b198fcb6bccf02d56319af
fit report:        f164e1481e09f3bc9be7450b7fc82fd682e0b1177b3c696db264c366c2d0202a
fit complete:      700170914af0e5721479fdd5594696cd872ac4f49ed5fcd5b6bd14649410b677
evaluation:        1c51131d6b8599a3733c3018e0a53570552ff09fff19aa07bcb7bf61b984e61c
decision:          895eaca96d44ae6f4a4a8bb5d35bb2561537452d56b8d0fe4b12eb68566083cb
evaluation complete: e7f1557d7a6b14ef6888b57c873b3574a15b01cdffc175b12893f7153a903afd
execution:         600b6933464b2b5e08b658aef5c70b09399e1fb7c123cdb2dc1121b319b29b09
job log:          71b0b6874e7478a920bca980a13d64e242dadd234d2d6cdd60d4105494d65bea
```

## Next action

Before fitting another action model, run a train-only, source-OOF action
generalization audit on the already fixed predictions. It must report action
priors and confusion matrices, per-fold agreement, teacher best-vs-second NLL
gap, chosen-action NLL regret, and agreement/regret stratified by predicted
confidence, teacher stability, entropy, and source frequency. This diagnostic
may choose the next train-only modeling family, but it cannot change this gate,
open validation/test, or retrospectively select an operating point.
