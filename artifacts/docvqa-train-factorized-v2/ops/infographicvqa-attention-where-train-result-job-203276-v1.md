# InfographicVQA raw-attention where train result v1

Date: 2026-09-02 (Asia/Hong_Kong)

## Bound execution and decision

- Slurm job: `203276`, state `COMPLETED`, exit `0:0`, runtime `00:07:08`.
- Evaluator revision: `5264d25e5cbd176dbd6597a74ba10e475e35b77a`.
- Feature job: `203257`, 23,946 decisions, 2,204 source groups, 4,406 images.
- Formal source bootstrap: 20,000 reused resamples over the frozen 2,204-source
  order.
- Decision: `attention_where_train_not_supported`; no operating point selected.
- Validation and test remained sealed, and the attention features contained no
  outcomes.

Artifact bindings:

```text
5c8bced0fdad0a4f7c3ad0dca8bf8cf31d40be4c9d2318c6b42ea72d065366ee  evaluation.json
ea38fb7adb024a1c96a6ec160d921687affb3ac0222aecba3f5d422728a4cbf5  complete.json
48099b9ffbed70882fdccbed6b9c5b704ea69d0c6d87698056883c7cd34a29de  job-203276.json
187c43167da777d42e96f5706bb80107769dcb4e9d58f20529c1161a6b8ab436  floating-point recovery protocol
```

## Registered gate results

No registered operating point had positive source-balanced cost-adjusted
utility or a strictly positive 95% lower endpoint.

| Nominal rate | Calls | Actual call rate | Utility | 95% interval | Helpful-call precision | Induced harm |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5% | 120 | 0.361% | -0.000097 | [-0.000334, 0.000140] | 9.46% | 0.000158 |
| 1% | 240 | 0.665% | -0.000041 | [-0.000772, 0.000629] | 15.53% | 0.000492 |
| 2% | 479 | 1.402% | -0.000585 | [-0.001573, 0.000311] | 14.96% | 0.001370 |
| 5% | 1,198 | 3.798% | -0.000410 | [-0.002438, 0.001681] | 19.35% | 0.003631 |
| 10% | 2,395 | 8.020% | -0.002685 | [-0.005536, 0.000195] | 17.82% | 0.008481 |

The 5% and 10% points were nevertheless paired-superior to every registered
deployable where comparator. At 5%, the 95% lower endpoints for utility minus
fixed, random, old-DECAR-where, and relative-where were respectively
`0.000787`, `0.000898`, `0.001000`, and `0.000452`. At 10%, they were
`0.001572`, `0.002190`, `0.001815`, and `0.001285`. These comparisons show a
real improvement in action localization, but not positive end-to-end value.

## Spatial signal and diagnosed bottleneck

Across all states, the raw-attention action matched the task-outcome oracle
crop 44.45% of the time and the best-NLL crop 31.48% of the time. It rescued
64.06% of helpful states. In the highest max-attention-score decile, those
figures rose to 50.65%, 41.59%, and 85.86%, respectively. Thus the attention
map and its confidence contain useful spatial information.

The frozen stopping policy, however, called solely on answer-now entropy. Even
with better crop selection, only 9.46%--19.35% of executed calls were helpful,
and every operating point had negative net utility at `lambda=0.05`. The
evidence therefore supports a factorized diagnosis: this candidate improves
**where**, while the remaining failure is dominated by **whether/when to
call** and by residual crop harm.

## Consequence

This branch does not authorize calibration, validation, or test. Continue the
already frozen ViCrop/LASER literature-attention extraction as the final strong
where-only check. If neither literature variant reaches positive utility, stop
adding pure where scorers under the entropy stop gate and move to a separately
registered joint stop/where construction or reposition the claim around the
identified factorization and strong negative evidence.
