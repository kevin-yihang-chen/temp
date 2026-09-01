# InfographicVQA entropy-when / OOF-where hybrid result

Status: terminal official-train-only diagnostic. The frozen decision is
`hybrid_train_not_supported`. Validation and test remained sealed. This record
was written after the endpoint and does not modify the frozen policy family,
budgets, cost, inference, or decision rule.

## Execution

| Field | Value |
| --- | --- |
| Slurm job | `203059` |
| State | `COMPLETED`, exit code `0:0`, zero restarts |
| Start / end | 2026-09-01 20:53:29 / 21:06:32 HKT |
| Queue / evaluation wall time | 4 s / 783 s |
| Allocation | debug; 1 RTX 4090 reserved for QOS admission; 4 CPU; 64 GiB |
| Evaluator device | CPU; the reserved GPU was hidden with an empty `CUDA_VISIBLE_DEVICES` |
| Code revision | `9a365a00e1df5dc05f131b3069ceaa2b5de3f1af` |
| Credentials in worker | false |
| Mail | Slurm `--mail-type=ALL` to `yihangc@connect.hku.hk` |

The job verified the frozen dependency hashes and clean tracked revision before
evaluation, refused overwrite, reused the exact formal bootstrap, and wrote
outputs atomically.

## Registered endpoints

All values below are source-balanced. `U` is utility at the frozen cost
`lambda = 0.05`; brackets are the paired whole-source 95% percentile interval
for the primary utility. `Oracle-where U` is the deterministic diagnostic
identity `primary U + action-selection regret`; it uses observed crop outcomes
and is therefore only an upper bound, not a deployable result.

| Nominal rate | Calls / sources | Primary U [95% CI] | ANLS gain | Helpful precision | Action regret | Task-value-where U | Entropy exhaustive U | Oracle-where U |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5% | 120 / 64 | -0.000326 [-0.000545, -0.000141] | -0.000146 | 4.65% | 0.000701 | -0.000345 | -0.000083 | +0.000374 |
| 1% | 240 / 110 | -0.000402 [-0.001104, +0.000201] | -0.000069 | 12.74% | 0.001196 | -0.000349 | -0.000208 | +0.000794 |
| 2% | 479 / 194 | -0.001245 [-0.002292, -0.000320] | -0.000544 | 10.91% | 0.002524 | -0.001431 | -0.000418 | +0.001279 |
| 5% | 1,198 / 390 | -0.002767 [-0.004660, -0.000898] | -0.000868 | 12.50% | 0.007711 | -0.002353 | -0.001192 | +0.004944 |
| 10% | 2,395 / 642 | -0.006714 [-0.009450, -0.004026] | -0.002704 | 12.22% | 0.014276 | -0.006228 | -0.002246 | +0.007561 |

No registered operating point qualified. Every point met the minimum call and
source requirement and passed every input, OOF, identity, cost, and bootstrap
audit. None had a strictly positive lower confidence endpoint, none beat every
feasible comparator, and none kept induced harm and negative-utility-call mass
no greater than both one-crop entropy baselines. The primary beat the
task-value-where point estimate only at 0.5% and 2%; every paired primary-minus-
task-value interval included zero.

## Audit and interpretation

- The population is exactly 23,946 official-train decisions from 2,204 sources.
- All prediction rows are outcome-free; the nested source-OOF audit contains 55
  zero-overlap checks. Validation/test inputs were not read.
- DECAR, loss-only, and no-harm-head actions agree on all rows. The meaningful
  task-value-only action differs on 17,446 rows, but does not consistently
  improve the entropy hybrid.
- The formal `int32 [20000, 2204]` source bootstrap and sorted source order were
  reused exactly for every policy and paired difference.
- The entropy-selected calls are source-concentrated. At 0.5%, 59 of 64 called
  sources have negative mean utility; at 10%, 536 of 642 do.

The deployable conclusion is negative: neither frozen OOF crop selector makes
the entropy gate useful at the registered budgets. The positive `Oracle-where
U` point estimates nevertheless isolate a testable remaining question: are the
entropy-selected states valuable under an outcome-oracle crop, with positive
source-bootstrap support? If yes, the next investment belongs in crop/action
ranking; if no, this entropy-when line should close. That one-shot,
official-train-only factorization must be frozen before its bootstrap endpoint
is computed and cannot open validation or test.

## Artifact identities

```text
8e7b939254e82fa98c41652e328c47720a7df32ecfba949db457f28b39f59cfd  slurm-infovqa-decar-hybrid-203059.out
ab4a179c2141de60c9a1d173e34a7166d075935e5283569439e3d93424344a62  entropy-where-hybrid-v1/evaluation.json
0597526725eac7efed05392fb652b04798de26b71d6de6d063303b49ec114d42  entropy-where-hybrid-v1/decision.json
4e3b3c2b2b2e1698fcf9bb3c9e71881b11e1357982bb1b6634a12af6a7aa03ac  entropy-where-hybrid-v1/complete.json
7c6a51295dc95624cd4adf07989fddba18f377462c65132fade8ec3dcbf0a0e6  hybrid-execution/job-203059.json
86e61bb0c7a4ad5a259077314be3a83c6c95284d2b219c414016b2280292a8bb  entropy-where-hybrid freeze
4cba9736f8635b57f8d6295e04b6fa2c3f7eef3d03b126ae5bcdb7f2924ef8c2  hybrid resource amendment
17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6  formal bootstrap indices
5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0  formal bootstrap sources
```

No GitHub push is authorized by this result.
