# InfographicVQA entropy-when / oracle-where factorization result

Status: terminal official-train-only diagnostic. The frozen decision is
`where_bottleneck_supported`. This is an outcome-oracle factorization result,
not deployable method evidence. Validation and test remained sealed.

## Execution

| Field | Value |
| --- | --- |
| Slurm job | `203078` |
| State | `COMPLETED`, exit code `0:0`, zero restarts |
| Start / end | 2026-09-01 21:23:45 / 21:29:33 HKT |
| Queue / evaluator time | 4 s / 345 s |
| Allocation | debug; 1 RTX 4090 reserved for QOS; 4 CPU; 64 GiB |
| Evaluator device | CPU; reserved GPU hidden from evaluator |
| Code revision | `5b760f5d900eb4ac9c1dc44c00c81366a5e870ae` |
| Credentials in worker | false |
| Mail | Slurm `ALL` state notifications to `yihangc@connect.hku.hk` |

## Frozen endpoints

All values are source-balanced at `lambda = 0.05` per executed crop. Brackets
are 95% intervals from the exact reused paired whole-source bootstrap. `Delta
DECAR` and `Delta task` are oracle-minus-OOF-selector utility differences.

| Rate | Calls / sources | Oracle utility [95% CI] | ANLS gain [95% CI] | Helpful precision | Delta DECAR [95% CI] | Delta task [95% CI] | Qualified |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0.5% | 120 / 64 | +0.000374 [+0.000058, +0.000768] | +0.000554 [+0.000219, +0.000980] | 31.27% | +0.000701 [+0.000320, +0.001186] | +0.000719 [+0.000327, +0.001196] | yes |
| 1% | 240 / 110 | +0.000794 [-0.000025, +0.001594] | +0.001126 [+0.000298, +0.001958] | 34.64% | +0.001196 [+0.000666, +0.001815] | +0.001142 [+0.000612, +0.001783] | no |
| 2% | 479 / 194 | +0.001279 [+0.000187, +0.002363] | +0.001980 [+0.000850, +0.003126] | 30.06% | +0.002524 [+0.001658, +0.003507] | +0.002710 [+0.001802, +0.003742] | yes |
| 5% | 1,198 / 390 | +0.004944 [+0.003051, +0.006990] | +0.006843 [+0.004861, +0.009021] | 32.57% | +0.007711 [+0.005891, +0.009820] | +0.007298 [+0.005687, +0.009157] | yes |
| 10% | 2,395 / 642 | +0.007561 [+0.004703, +0.010618] | +0.011572 [+0.008605, +0.014773] | 28.64% | +0.014276 [+0.011687, +0.017220] | +0.013790 [+0.011390, +0.016469] | yes |

The 10% point was selected by the frozen higher-oracle-utility, then lower-rate
rule. The 1% point failed only because its oracle-utility lower endpoint was
slightly below zero; both paired action-selector gaps were still strictly
positive. Every other registered point qualified.

## Scientific conclusion

Holding entropy-selected identities, one executed crop, and cost fixed, the
outcome-oracle crop choice changes the hybrid from negative to significantly
positive utility and significantly beats both frozen OOF crop selectors. The
where selector is therefore a material bottleneck, not merely a secondary
implementation detail.

This does **not** show that a realizable action model works. The oracle reads
the four observed crop outcomes. Its only valid consequence is to justify one
new official-train branch: build an outcome-free crop ranker that directly
learns within-state action contrasts, evaluate it with source-disjoint OOF
predictions, and require it to close a pre-frozen fraction of the oracle gap.
No validation/test opening is authorized by this result.

## Audit

- Exact population: 23,946 decisions, 4,406 images, 2,204 sources.
- Exact formal `int32 [20000, 2204]` bootstrap and source order reused.
- All raw prediction rows remained outcome-free; all 55 OOF source-overlap
  checks were zero.
- Hybrid identities, actions, costs, and all aggregate metrics reproduced
  exactly before the oracle comparison.
- Per-state and aggregate identities `oracle utility = learned utility +
  learned action-selection regret` passed at every point.
- The result explicitly records `outcome_oracle_used=true`,
  `deployable_method_evidence=false`, and
  `validation_or_test_inputs_used=false`.

## Artifact identities

```text
a44a66d10edc0396a25fd28f731dd9ec072163d9fff9b496608b40a4d20ef58c  slurm-infovqa-oracle-where-203078.out
6ef0869b453e1a70ad5f479e8a9604aa04ac0419a1dc92d1ce353c54f66f3025  entropy-oracle-where-factorization-v1/evaluation.json
022ed00d45e7bdb269f64bd2a0e3e4b7b1598f69e9206648664785f92ca550b5  entropy-oracle-where-factorization-v1/decision.json
b940258389e558d0d0bae277bd8d5b081923ce505ca2e6ad8520a79bd6411de7  entropy-oracle-where-factorization-v1/complete.json
8c11057f8f9cef77037a1feeee4c2fb6cf01f82d9f21fb980ffacf27a00b82b2  oracle-where-execution/job-203078.json
c22fe6fd51a60063c7765df88b2f0491070433dd905b3143ee01fa1539a9f82c  factorization freeze
c5b2b547460979634291a8508950221b9bc4c24239bc6421a2e6d3273f0718ee  resource amendment
17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6  bootstrap indices
5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0  bootstrap source order
```

No GitHub push is authorized by this result.
