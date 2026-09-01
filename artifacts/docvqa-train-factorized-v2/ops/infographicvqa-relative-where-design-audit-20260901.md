# InfographicVQA relative-where design audit

Status: exploratory official-train analysis performed after the frozen
job-203078 factorization result and before freezing a new crop ranker. It is a
model-design diagnostic, not a confirmatory endpoint. Validation and test were
not read.

## Bound evidence

```text
9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e  merged rollouts
884de4ebc1ba83226393871cdfff819bc7553a61eee6ffa11697db03332ac646  answer-NLL teacher rows
d0508726a50b4c1e54778392d08329b242a680fc13292cac1ebec8b42a175300  label-free feature bank
c8338b1960ca223c892c3f992b0ccc3027d543113f67c82a734eccd7e3699c4b  DECAR source-OOF predictions
7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6a232b0837a  outer source folds
c520dc3fd9a5d25c0b7e626a88a55e629e56c8a6d1a473feb53d94bad4689cf0  oracle-where result record
```

The analysis joined action outcomes and privileged teacher labels only after
the existing OOF prediction file was fixed. It did not alter any previous
decision.

## What failed

Across all 95,784 candidate rows, teacher answer-NLL gap and realized ANLS
delta have Pearson correlation 0.4287 and Spearman correlation 0.3272. More
importantly for selection, 14,126 candidate pairs have unequal values under
both targets and their ordering agrees 86.34% of the time. Thus the privileged
teacher ranking is imperfect but strongly task-aligned.

The existing OOF ranker does not recover that ranking. Over all states it
selects the exact teacher argmax only 25.02% source-balanced. On the fixed
entropy-selected tails, agreement remains between 20.40% and 26.21%, close to
four-way chance. The direct-task-delta network is not better. Both networks
fit their outer-training objectives to low loss, so this is an out-of-source
generalization failure rather than an optimization crash.

Task top-one accuracy is inflated by crop-outcome ties: 69.9%--77.1% of called
states have multiple task-optimal actions. Utility and regret are therefore
the primary diagnostics, not nominal top-one accuracy.

| Entropy rate | Called states / sources | OOF teacher agreement | OOF chosen task gain | Raw-teacher task top-one | Raw-teacher helpful precision | Raw-teacher task gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5% | 120 / 64 | 20.40% | -0.0441 | 88.15% | 17.95% | +0.0791 |
| 1% | 240 / 110 | 20.87% | -0.0132 | 92.56% | 21.83% | +0.1172 |
| 2% | 479 / 194 | 26.21% | -0.0229 | 90.85% | 20.35% | +0.0921 |
| 5% | 1,198 / 390 | 24.25% | -0.0145 | 91.71% | 23.96% | +0.1233 |
| 10% | 2,395 / 642 | 24.24% | -0.0361 | 92.17% | 22.45% | +0.0882 |

The gains and percentages in this table are source-balanced among sources with
calls. The later formal evaluator must return to the all-source estimand and
the exact registered bootstrap.

## Design consequence

The next branch should retain answer-NLL as privileged supervision but replace
absolute scalar regression with explicit four-candidate relational
distillation:

1. optimize a listwise top-crop distribution and target-difference-weighted
   pairwise ordering, rather than an absolute smooth-L1 target dominated by
   irrelevant within-state offsets;
2. expose each candidate together with its difference from the four-candidate
   mean so the network represents comparative crop evidence directly;
3. source-balance every outer-training fold and deterministically emphasize
   high-answer-entropy states using inference-visible entropy ranks;
4. down-weight states whose privileged candidates are nearly indistinguishable;
5. emit only source-held-out scores/actions, never teacher gaps or task
   outcomes.

The confirmatory family should include fixed absolute-feature, no-entropy-
weight, and direct-task-target ablations. The relational teacher-distilled
variant must be fixed as primary before fitting. Architecture selection after
outer endpoints is prohibited.

## Novelty boundary

GapSight already uses loss-difference crop supervision, so neither privileged
NLL labels nor adaptive crop routing can be claimed as new. The defensible
paper contribution remains the conjunction of complete signed same-state
siblings, a cost-faithful one-crop decision, source-OOF relational action
distillation, prospective harm control, and matched-budget/oracle-regret
auditing. A successful new ranker must be presented within that system, not as
the first loss-gap crop method.

No validation/test opening or GitHub push is authorized by this exploratory
audit.
