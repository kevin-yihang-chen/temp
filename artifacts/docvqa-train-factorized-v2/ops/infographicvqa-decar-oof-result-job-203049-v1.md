# InfographicVQA DECAR official-train nested-OOF result

Status: completed, audited, and not advanced. Validation and test remain
sealed. This is a valid negative scientific result, not an engineering
failure.

## Execution

Slurm authoritatively records job `203049` as `COMPLETED`, `ExitCode=0:0`,
with zero restarts. It ran from 2026-09-01 20:13:16 to 20:30:59 HKT on one
NVIDIA H800 with 12 CPUs and 192 GiB. Queue wait was 25 seconds, fit time was
223 seconds, evaluation time was 830 seconds, and total worker time was 1,053
seconds. All supported state-change emails were bound to
`yihangc@connect.hku.hk`.

The job used code revision
`fd66185411f9a6187f91422706b5ae83d8e755d4`. It verified the frozen
scientific contract, resource amendment, startup-hash correction, generation
execution, and every task-input hash before fitting.

Bound evidence:

```text
a87736096b9c7763140eadcb710d94a4291ec52fbc218d25431ccb14c5d2cdd2  slurm job log
c8338b1960ca223c892c3f992b0ccc3027d543113f67c82a734eccd7e3699c4b  OOF predictions
dc3193dfc626a3df50321f4d92a336ce784aee34f1f2e91c57cf87d1f8085537  OOF audit
ebc936739e970fbfda25ebe02ef71d6b7f46674f9d00092011b5665c2daa9bf0  OOF report
8de073870fcade5ac111d59de81e9c70dc567c9900e0b223cdebca6a8318f31f  OOF completion
17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6  bootstrap indices
5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0  bootstrap source order
ee5f9972e1d897c7fb833208a5722ee3a0313a05f0217f921966b3e0e1978df9  evaluation
a66c587cdd1382914ef2da32918cdc20d23a7ce071709b58480c1e5a719039f8  advancement decision
d0443614c286349b7e360d646fef960816aba47a614bc20960c119d5e0ddeb79  evaluation completion
d8a5b26ff21475b832f1ff909db96da29737154594fcffffde9882eaff6ef673  OOF execution
```

## Integrity audit

The fit emitted exactly 23,946 predictions over 2,204 sources and 4,406
images. Every prediction row is outcome-free; an independent recursive field
scan found no outcome, answer, ANLS, accuracy, target, or label key. The five
outer source folds cover all 23,946 rows and 2,204 sources with zero overlap.
All 40 inner source-fit overlap checks are also zero.

The evaluator used one shared, deterministic `int32 [20000, 2204]` whole-source
bootstrap matrix with seed `20260917`. Independent loading reproduced its
shape, dtype, index range 0--2203, and a sorted unique 2,204-source order. All
registered fit, join, tie, cost, and bootstrap audits passed. No validation or
test input was read.

## Registered result

The decision is `decar_not_advanced`; no operating point qualified.

| Nominal rate | Calls | Called sources | Source utility (95% CI) | Helpful-call precision | Induced harm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5% | 120 | 68 | -0.000273 [-0.000460, -0.000103] | 6.71% | 0.000193 |
| 1% | 240 | 121 | -0.000529 [-0.000849, -0.000239] | 5.94% | 0.000389 |
| 2% | 479 | 194 | -0.001152 [-0.001831, -0.000619] | 5.19% | 0.000756 |
| 5% | 1,198 | 440 | -0.003300 [-0.004713, -0.001851] | 4.71% | 0.002450 |
| 10% | 2,395 | 734 | -0.007651 [-0.009668, -0.005751] | 3.27% | 0.004994 |

Every rate passed the minimum-call/source and general audit conditions. Every
rate failed strictly positive utility, dominance over every feasible
non-oracle baseline, and the harm constraint; `answer_now` was the strongest
feasible non-oracle baseline. DECAR strictly beat all three registered learned
ablations at 0.5%, 1%, and 2%, but not at 5% or 10%.

## Diagnosis and remaining headroom

At 0.5%, DECAR's source-balanced helpful-call precision was 6.71%, barely
above the 6.31% helpful-state base rate. Its source utility was already
significantly negative, so the failure is not caused by bootstrap variance or
insufficient calls. The dominant scientific failure is poor OOF enrichment in
the learned `when` ranking, followed by action-choice regret and induced harm.
Increasing the call rate makes every primary endpoint worse.

The outcome oracle proves that the task still has substantial selective-tool
headroom: oracle stopping attains source utility 0.03385 with zero induced
harm. Calling one crop everywhere or exhaustive four-crop UG is negative under
the registered cost, so the headroom specifically requires sparse, precise
selection.

The strongest observed train-only clue is the registered entropy-gated UG
baseline. It obtains positive ANLS gain at every rate and helpful-call
precision between 18.8% and 37.0%, but remains utility-negative because it
executes four crops per selected state. This motivates a new, explicitly
post-result hypothesis: use the label-free entropy signal only for `when` and
the outcome-free OOF action-value prediction only for `where`, executing one
crop rather than four. That hypothesis must be frozen and evaluated on
official train only before any validation access.

The registered decision is final for DECAR v1. Validation and test are not
opened, and no GitHub push is authorized.
