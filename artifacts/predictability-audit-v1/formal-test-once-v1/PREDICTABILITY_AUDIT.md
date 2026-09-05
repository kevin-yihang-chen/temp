# Pre-action visual-tool utility predictability audit

**Final audit outcome: INCONCLUSIVE.**

This label was explicitly accepted by the user after the preregistered four-way rules returned their frozen inconclusive branch. It is not a post-hoc claim of GO, PIVOT, REPRESENTATION, or STOP.

> This rule outcome uses the complete preregistered 3 benchmark x 4 predictor levels x 3 target formulations matrix, three fixed seeds, and one source/RGB-disjoint held-out test transaction.

## Answer to the research question

Under the complete tested ladder, stable actionable pre-action prediction was not demonstrated. The fixed tool nevertheless has significant oracle headroom on all three benchmarks. Because the diagnostic post-action probe also lacked positive lower-confidence-bound utility, this audit does not establish that sequential acquisition will succeed; it only rules out claiming success for the tested static router.

Recommended next phase: stop tuning this consumed-test static gate. If work continues, preregister a new active/sequential evidence-acquisition study with a fresh held-out test; do not reuse this test for model or threshold selection.

## Oracle headroom

| Benchmark | Always-call utility | Binary-oracle utility | Oracle 95% CI | Rescue rate | Harm rate |
|---|---:|---:|---:|---:|---:|
| chartqa | -0.227000 | 0.023200 | [0.015200, 0.032000] | 0.029000 | 0.056000 |
| docvqa | -0.192239 | 0.019355 | [0.013229, 0.026421] | 0.068874 | 0.031176 |
| hrbench | -0.162500 | 0.050000 | [0.020000, 0.080000] | 0.062500 | 0.025000 |

## Primary deployable policy and strongest baseline

| Benchmark | Strongest baseline | Delta utility 95% CI | Accuracy (candidate/base) | Cost (candidate/base) | Pareto | Rescue precision higher | Harm no higher |
|---|---|---:|---:|---:|:---:|:---:|:---:|
| chartqa | `answer_now` | -0.001600 [-0.004400, 0.000000] | 0.781000/0.782000 | 0.012000/0.000000 | no | no | no |
| docvqa | `entropy_gate_fixed_visual_tool` | 0.000980 [-0.000918, 0.002973] | 0.898792/0.898098 | 0.065146/0.070876 | yes | yes | no |
| hrbench | `fixed_crop_with_matched_gate` | 0.006563 [-0.022812, 0.036250] | 0.537500/0.531250 | 0.125000/0.131250 | yes | no | no |

Undefined rescue/harm rates for a zero-call policy are compared as 0.0, exactly as frozen before test.

## Rescue and harm at the selected operating point

| Benchmark | Candidate rescue precision | Baseline rescue precision | Candidate harm/call | Baseline harm/call | Gain/call |
|---|---:|---:|---:|---:|---:|
| chartqa | 0.000000 | 0.000000 | 0.333333 | 0.000000 | -0.333333 |
| docvqa | 0.564103 | 0.404762 | 0.076923 | 0.071429 | 0.196684 |
| hrbench | 0.000000 | 0.047619 | 0.200000 | 0.142857 | -0.200000 |

## Predictor ladder

Each row is the mean across the three fixed seeds; model variant and threshold selection used validation only.

| Benchmark | Level | Target | Utility | AUROC | AUPRC | Brier | Calibration | Rescue AUPRC | Harm AUPRC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| chartqa | `l0_uncertainty` | `direct_gain` | -0.000600 | 0.745694 | 0.061188 | 0.028044 | 0.009568 | 0.061188 | 0.124034 |
| chartqa | `l0_uncertainty` | `rescue_harm` | -0.000600 | 0.745694 | 0.061188 | 0.028102 | 0.009348 | 0.061188 | 0.124034 |
| chartqa | `l0_uncertainty` | `factorized` | -0.000400 | 0.697006 | 0.049024 | 0.028236 | 0.008416 | 0.049024 | 0.205781 |
| chartqa | `l1_shallow` | `direct_gain` | -0.000867 | 0.648603 | 0.096742 | 0.027823 | 0.009163 | 0.096742 | 0.172282 |
| chartqa | `l1_shallow` | `rescue_harm` | -0.001600 | 0.612025 | 0.081917 | 0.027983 | 0.008548 | 0.081917 | 0.159893 |
| chartqa | `l1_shallow` | `factorized` | -0.000867 | 0.639630 | 0.156794 | 0.027395 | 0.008567 | 0.156794 | 0.185815 |
| chartqa | `l2_semantic` | `direct_gain` | -0.000200 | 0.500231 | 0.031096 | 0.028241 | 0.009019 | 0.031096 | 0.072680 |
| chartqa | `l2_semantic` | `rescue_harm` | -0.000067 | 0.502811 | 0.056243 | 0.028249 | 0.010313 | 0.056243 | 0.160938 |
| chartqa | `l2_semantic` | `factorized` | -0.000867 | 0.497508 | 0.053841 | 0.028681 | 0.012157 | 0.053841 | 0.144244 |
| chartqa | `l3_frozen_qwen` | `direct_gain` | 0.000000 | 0.613694 | 0.068698 | 0.028045 | 0.009469 | 0.068698 | 0.092787 |
| chartqa | `l3_frozen_qwen` | `rescue_harm` | 0.001000 | 0.633569 | 0.093585 | 0.028055 | 0.009373 | 0.093585 | 0.168433 |
| chartqa | `l3_frozen_qwen` | `factorized` | 0.000133 | 0.663104 | 0.095431 | 0.028200 | 0.009671 | 0.095431 | 0.177788 |
| docvqa | `l0_uncertainty` | `direct_gain` | -0.001738 | 0.758356 | 0.221173 | 0.060707 | 0.025089 | 0.221173 | 0.079966 |
| docvqa | `l0_uncertainty` | `rescue_harm` | -0.004217 | 0.769973 | 0.213571 | 0.061308 | 0.020823 | 0.213571 | 0.080516 |
| docvqa | `l0_uncertainty` | `factorized` | -0.001738 | 0.758356 | 0.221173 | 0.062305 | 0.024522 | 0.221173 | 0.079966 |
| docvqa | `l1_shallow` | `direct_gain` | -0.004364 | 0.752139 | 0.259028 | 0.057897 | 0.020553 | 0.259028 | 0.070048 |
| docvqa | `l1_shallow` | `rescue_harm` | -0.000969 | 0.774567 | 0.278076 | 0.055388 | 0.015414 | 0.278076 | 0.060877 |
| docvqa | `l1_shallow` | `factorized` | -0.000895 | 0.775104 | 0.297304 | 0.054173 | 0.016468 | 0.297304 | 0.056022 |
| docvqa | `l2_semantic` | `direct_gain` | -0.000253 | 0.557832 | 0.075433 | 0.059833 | 0.006718 | 0.075433 | 0.032398 |
| docvqa | `l2_semantic` | `rescue_harm` | -0.002850 | 0.558651 | 0.111582 | 0.062664 | 0.020076 | 0.111582 | 0.051203 |
| docvqa | `l2_semantic` | `factorized` | -0.001972 | 0.603268 | 0.131264 | 0.060313 | 0.014475 | 0.131264 | 0.053081 |
| docvqa | `l3_frozen_qwen` | `direct_gain` | -0.000310 | 0.510342 | 0.087536 | 0.060368 | 0.008712 | 0.087536 | 0.041239 |
| docvqa | `l3_frozen_qwen` | `rescue_harm` | -0.002091 | 0.594976 | 0.143255 | 0.062050 | 0.023317 | 0.143255 | 0.041095 |
| docvqa | `l3_frozen_qwen` | `factorized` | -0.000773 | 0.577160 | 0.143563 | 0.059097 | 0.014645 | 0.143563 | 0.042444 |
| hrbench | `l0_uncertainty` | `direct_gain` | -0.032500 | 0.669333 | 0.099885 | 0.068913 | 0.076697 | 0.099885 | 0.084728 |
| hrbench | `l0_uncertainty` | `rescue_harm` | -0.012500 | 0.742000 | 0.128042 | 0.065946 | 0.073020 | 0.128042 | 0.221976 |
| hrbench | `l0_uncertainty` | `factorized` | -0.032500 | 0.660000 | 0.098627 | 0.070073 | 0.090646 | 0.098627 | 0.084345 |
| hrbench | `l1_shallow` | `direct_gain` | -0.051250 | 0.648444 | 0.101335 | 0.071698 | 0.101970 | 0.101335 | 0.036729 |
| hrbench | `l1_shallow` | `rescue_harm` | -0.032500 | 0.607333 | 0.102643 | 0.069482 | 0.102211 | 0.102643 | 0.050067 |
| hrbench | `l1_shallow` | `factorized` | -0.034583 | 0.597111 | 0.151617 | 0.068092 | 0.097816 | 0.151617 | 0.035396 |
| hrbench | `l2_semantic` | `direct_gain` | 0.000000 | 0.511111 | 0.081571 | 0.074286 | 0.114084 | 0.081571 | 0.102687 |
| hrbench | `l2_semantic` | `rescue_harm` | -0.012500 | 0.557778 | 0.093257 | 0.070447 | 0.106932 | 0.093257 | 0.037262 |
| hrbench | `l2_semantic` | `factorized` | -0.001250 | 0.577556 | 0.106238 | 0.068858 | 0.102358 | 0.106238 | 0.033766 |
| hrbench | `l3_frozen_qwen` | `direct_gain` | -0.002917 | 0.444000 | 0.078795 | 0.073211 | 0.112726 | 0.078795 | 0.037285 |
| hrbench | `l3_frozen_qwen` | `rescue_harm` | -0.016667 | 0.504667 | 0.096482 | 0.071172 | 0.110756 | 0.096482 | 0.128258 |
| hrbench | `l3_frozen_qwen` | `factorized` | -0.013750 | 0.293333 | 0.058383 | 0.070596 | 0.106159 | 0.058383 | 0.021813 |

### Validation-selected seed score curves

These are test-time means of the three independently frozen seed curves. They expose accuracy versus call rate, accuracy versus visual cost, and utility versus call rate without selecting a test point.

| Benchmark | Requested call rate | Realized call rate | Accuracy | Visual cost | Utility |
|---|---:|---:|---:|---:|---:|
| chartqa | 0.000000 | 0.000000 | 0.782000 | 0.000000 | 0.000000 |
| chartqa | 0.005000 | 0.005000 | 0.782333 | 0.020000 | -0.000667 |
| chartqa | 0.010000 | 0.010000 | 0.783333 | 0.040000 | -0.000667 |
| chartqa | 0.020000 | 0.020000 | 0.784667 | 0.080000 | -0.001333 |
| chartqa | 0.050000 | 0.050000 | 0.789333 | 0.200000 | -0.002667 |
| chartqa | 0.100000 | 0.100000 | 0.789333 | 0.400000 | -0.012667 |
| chartqa | 0.200000 | 0.200000 | 0.787000 | 0.800000 | -0.035000 |
| chartqa | 0.500000 | 0.500000 | 0.789333 | 2.000000 | -0.092667 |
| chartqa | 1.000000 | 1.000000 | 0.755000 | 4.000000 | -0.227000 |
| docvqa | 0.000000 | 0.000000 | 0.896293 | 0.000000 | 0.000000 |
| docvqa | 0.005000 | 0.004010 | 0.896835 | 0.016041 | -0.000260 |
| docvqa | 0.010000 | 0.008735 | 0.897187 | 0.034940 | -0.000853 |
| docvqa | 0.020000 | 0.017679 | 0.898827 | 0.070715 | -0.001002 |
| docvqa | 0.050000 | 0.043273 | 0.901289 | 0.173093 | -0.003658 |
| docvqa | 0.100000 | 0.093834 | 0.903273 | 0.375338 | -0.011786 |
| docvqa | 0.200000 | 0.195602 | 0.907042 | 0.782407 | -0.028371 |
| docvqa | 0.500000 | 0.497480 | 0.910568 | 1.989919 | -0.085220 |
| docvqa | 1.000000 | 1.000000 | 0.904053 | 4.000000 | -0.192239 |
| hrbench | 0.000000 | 0.000000 | 0.543750 | 0.000000 | 0.000000 |
| hrbench | 0.005000 | 0.006250 | 0.543750 | 0.025000 | -0.001250 |
| hrbench | 0.010000 | 0.012500 | 0.537500 | 0.050000 | -0.008750 |
| hrbench | 0.020000 | 0.018750 | 0.537500 | 0.075000 | -0.010000 |
| hrbench | 0.050000 | 0.050000 | 0.543750 | 0.200000 | -0.010000 |
| hrbench | 0.100000 | 0.100000 | 0.537500 | 0.400000 | -0.026250 |
| hrbench | 0.200000 | 0.200000 | 0.550000 | 0.800000 | -0.033750 |
| hrbench | 0.500000 | 0.500000 | 0.581250 | 2.000000 | -0.062500 |
| hrbench | 1.000000 | 1.000000 | 0.581250 | 4.000000 | -0.162500 |

## Accuracy-cost frontier

The JSON report contains every per-seed curve point. The table below shows the actual frozen majority-vote operating point against its validation-selected strongest baseline.

| Benchmark | Policy | Accuracy | Visual cost | Tool-call rate | Utility |
|---|---|---:|---:|---:|---:|
| chartqa | `primary deployable` | 0.781000 | 0.012000 | 0.003000 | -0.001600 |
| chartqa | `answer_now` | 0.782000 | 0.000000 | 0.000000 | 0.000000 |
| docvqa | `primary deployable` | 0.898792 | 0.065146 | 0.016287 | -0.000758 |
| docvqa | `entropy_gate_fixed_visual_tool` | 0.898098 | 0.070876 | 0.017719 | -0.001738 |
| hrbench | `primary deployable` | 0.537500 | 0.125000 | 0.031250 | -0.012500 |
| hrbench | `fixed_crop_with_matched_gate` | 0.531250 | 0.131250 | 0.131250 | -0.019063 |

## Diagnostic upper bound and representation check

| Benchmark | Post-action vs answer-now 95% CI | L3 validation vs baseline 95% CI | L3 test vs baseline 95% CI | Max lower CI over all deployables |
|---|---:|---:|---:|---:|
| chartqa | [0.000000, 0.000000] | [0.000000, 0.004444] | [0.000000, 0.000000] | 0.000000 |
| docvqa | [-0.001548, -0.000126] | [-0.003164, 0.000588] | [-0.001384, 0.002069] | 0.000049 |
| hrbench | [-0.003750, 0.000000] | [-0.062812, 0.022813] | [-0.015312, 0.030313] | -0.004375 |

## Registered rule gap

| Frozen rule component | Observed benchmark count |
|---|---:|
| STOP oracle-small benchmarks | 0 |
| GO all-condition benchmarks | 0 |
| REPRESENTATION L3-validation-positive benchmarks | 0 |
| REPRESENTATION L3-test-nonpositive benchmarks | 1 |
| PIVOT oracle-positive benchmarks | 3 |
| PIVOT post-action-positive benchmarks | 0 |
| PIVOT all-deployable-nonpositive benchmarks | 2 |

The preregistered protocol explicitly maps this uncovered combination to `raise_and_do_not_emit_final_verdict`. The user explicitly chose to retain this frozen inconclusive outcome as the terminal report. This file does not relabel the result as GO, PIVOT, REPRESENTATION, or STOP.

## Bootstrap and integrity

All primary confidence intervals are paired whole-source bootstrap intervals with 20,000 resamples. The complete machine-readable report retains the deterministic seed schedule and all individual curves.

- Formal test report SHA-256: `adbd3f53ddb3d7d5dee04ff5b0ab553495cc74ff5c9464e42ebb5492ccd7d49f`
- Frozen matrix SHA-256: `be7f08f417653f20a15e49cee7f65bd893cbb3f2eeac935c64bea8c71c21ecbf`
- Frozen inventory SHA-256: `1005e507df87e02c6c6cb0af8a8569fa0b2b1af1961682b7fb9b5ff9d02b4572`
- Test access ledger SHA-256: `1bcb3c548dd3356eb3744ad00dc4690a186121136978f8834ff05bd3ea65e111`
- Protocol SHA-256: `699073b149c957022b203e71dc0ae9e7c7733515efb125f26a86713021a3c6e1`
- Untouched-test allocation report SHA-256: `4c072355b75dcd7b228267f30c4790efa3d9facbdae1a731ac903ec351efb468`
- Clean code revision: `2151b82e44bee0bcd48c30aebc7bc02e1da418a7`

The test transaction is consumed. This result must not be used to select a replacement predictor, threshold, feature, seed, or verdict rule.
