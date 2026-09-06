# Sequential Visual Acquisition: GO / NO-GO

Final verdict: **NO-GO — stop this formulation; do not run RL or scale the model.**

This is a development-stage stopping verdict under the preregistered rules. A one-shot
sequential test was not run because the validation stop conditions were already met.

## 1. Scope and data

The experiment asks only whether frozen Qwen2.5-VL-3B features at a shared partial visual
prefix can predict `STOP` versus one fixed additional acquisition. `sequential-opposite-ug-v1`
chooses both regions without outcomes: STOP sees the original image plus one crop, and
CONTINUE sees that identical prefix plus one geometrically fixed crop. Both branches share
the prompt, seed, decoder, scorer, and state. The critic never receives CONTINUE-side
answers, uncertainty, correctness, reward, gain, or crop-after-action features.

The bounded development screen used the following source/RGB-disjoint data. There are
1,152 paired decisions in the main screen; the earlier 144-decision engineering smoke is not
used for the verdict.

| Benchmark | Train decisions / sources | Validation decisions / sources | Train beneficial / harmful / neutral | Validation beneficial / harmful / neutral |
|---|---:|---:|---:|---:|
| ChartQA | 256 / 256 | 128 / 128 | 41 / 6 / 209 | 16 / 4 / 108 |
| DocVQA | 256 / 71 | 128 / 37 | 9 / 7 / 240 | 8 / 2 / 118 |
| HRBench | 256 / 256 | 128 / 128 | 22 / 14 / 220 | 9 / 7 / 112 |

## 2. Counterfactual headroom

Headroom exists in all three validation sets. The oracle result below is paired against
Always STOP with 10,000 whole-source bootstrap samples. At `lambda=0.05`, incremental cost
is charged exactly once only when CONTINUE is selected.

| Benchmark | Beneficial / harmful / neutral rate | Mean gain | Oracle gain, lambda=0 | Oracle utility, lambda=.05 (95% CI) |
|---|---:|---:|---:|---:|
| ChartQA | 12.50% / 3.13% / 84.38% | +0.09375 | +0.12500 | +0.11875 [0.06680, 0.17812] |
| DocVQA | 6.25% / 1.56% / 92.19% | +0.01631 | +0.01781 | +0.01523 [0.00000, 0.03917] |
| HRBench | 7.03% / 5.47% / 87.50% | +0.01563 | +0.07031 | +0.06680 [0.02969, 0.11133] |

Therefore STOP-1 is not the explanation: the fixed second observation has real causal
value. The failure is identifying the useful states before executing it.

Entropy reduction is not a reliable utility label. Validation entropy/gain Pearson is
`0.0292/0.1457/0.2250` for ChartQA/DocVQA/HRBench; sign mismatch is
`57.81%/62.50%/45.31%`; useful precision is only `12.82%/7.14%/13.43%`.

## 3. Critics

The initial raw frozen-state feature has 18,461 coordinates. It produced gain useful-action
AUROC `0.673/0.384/0.485` on ChartQA/DocVQA/HRBench and did not give a positive lower
confidence bound over the strongest matched baseline. ChartQA-to-HRBench transfer called
the tool on 90.63% of states for `lambda<=0.1`, outside the allowed nontrivial range, and
did not beat its matched baselines.

Following the protocol's PARTIAL-GO branch, exactly one representation correction was
frozen before its results: a 90-dimensional, label-free relational summary of norms,
cosines, pairwise distances, uncertainty, geometry, and history. The model, loss, hidden
size, seeds, split, lambda grid, and outcomes were unchanged.

| Benchmark | Gain AUROC | Gain Pearson | Risk AUROC | Entropy / confidence / margin risk AUROC | Risk Brier / ECE / AURC |
|---|---:|---:|---:|---:|---:|
| ChartQA | 0.7204 | 0.2916 | 0.9106 | 0.6915 / 0.6880 / 0.6696 | 0.1158 / 0.0824 / 0.2762 |
| DocVQA | 0.6396 | 0.2172 | 0.7293 | 0.6536 / 0.6950 / 0.6918 | 0.1735 / 0.1505 / 0.1342 |
| HRBench | 0.4528 | -0.0748 | 0.7256 | 0.8177 / 0.7913 / 0.7766 | 0.2112 / 0.1030 / 0.3341 |

Remaining STOP risk is predictable on ChartQA and weakly on DocVQA, but this does not imply
that another crop will repair the answer. HRBench's entropy baseline is stronger than the
learned risk critic.

## 4. Matched-rate policy result

The table deliberately reports the *most favorable* nontrivial-rate cell for each focal
policy over the already frozen lambda grid. This is an optimistic screen, not a test-set
selection. “Strongest” is the best of deployable entropy, confidence, and margin at the
same acquisition count. Every interval uses 10,000 paired whole-source resamples.

| Benchmark / policy | Lambda | Acquisition rate | Utility minus strongest baseline (95% CI) | Accuracy difference (95% CI) |
|---|---:|---:|---:|---:|
| ChartQA gain | .20 | 25.78% | +0.03125 [-0.01719, 0.08125] | +0.03125 [-0.01563, 0.07813] |
| ChartQA risk+gain | .10 | 34.38% | +0.03125 [-0.01484, 0.07969] | +0.03125 [-0.01563, 0.07813] |
| DocVQA gain | .025 | 27.34% | +0.01348 [-0.00269, 0.03770] | +0.01348 [-0.00327, 0.03944] |
| DocVQA risk+gain | all | 0.00% | trivial Always STOP | trivial Always STOP |
| HRBench gain | .05 | 36.72% | +0.00000 [-0.03164, 0.03164] | +0.00000 [-0.03125, 0.03125] |
| HRBench risk+gain | .025 | 32.81% | -0.00781 [-0.03730, 0.01621] | -0.00781 [-0.03906, 0.01563] |

At `lambda=0`, the gain policy's ChartQA/DocVQA/HRBench acquisition rates are
`61.72%/49.22%/71.09%`; accuracies are `0.5391/0.9307/0.5078`, versus Always STOP
`0.4609/0.9150/0.5156`. HRBench therefore loses accuracy while spending more visual cost.
No learned or risk-aware policy has a strictly positive lower confidence bound on even one
benchmark, let alone two. The accuracy-cost figures are stored beside each final evaluation:

- `artifacts/sequential-acquisition-v1/critic-chartqa-relational-audit-256x128-v1/job-209065/evaluation/accuracy_cost_frontier.png`
- `artifacts/sequential-acquisition-v1/critic-docvqa-relational-audit-256x128-v1/job-209066/evaluation/accuracy_cost_frontier.png`
- `artifacts/sequential-acquisition-v1/critic-hrbench-relational-audit-256x128-v1/job-209064/evaluation/accuracy_cost_frontier.png`

## 5. GO conditions and stop rules

1. **GO-1 headroom:** pass. Beneficial rate exceeds 3% in all domains; oracle utility is
   positive, with a strictly positive lower bound in ChartQA and HRBench.
2. **GO-2 predictable gain:** fail. Learned-minus-strongest lower bound is not positive in
   any domain.
3. **GO-3 matched-rate action quality:** fail. All matched-rate accuracy/utility intervals
   include zero; HRBench often has the wrong sign.
4. **GO-4 nontrivial policy:** fail for DocVQA risk+gain, which is Always STOP. The gain-only
   policies are nontrivial but not statistically better.
5. **Cross-domain stability:** fail. ChartQA-to-HRBench transfer is trivial/high-call and
   non-superior.

This triggers STOP-2 (headroom exists but current pre-action representation does not identify
gain), STOP-3 (DocVQA risk+gain collapses), STOP-4 (matched-rate intervals cross zero), and
STOP-5 (no cross-benchmark stability). The sole allowed representation correction has been
used, so no further feature, seed, threshold, architecture, backbone-size, or RL search is
authorized.

## 6. Reproducibility and test-access note

Rollout jobs `209048/209052`, `209051/209049`, and `209050/209053` and final audit jobs
`209065/209066/209064` all completed with exit code 0 on RTX 4090 GPUs and Slurm email
notifications set to `ALL`. Final evaluation report SHA-256 values are:

- ChartQA: `00f95dae7f0dd7c7015e3cf025af69ce6f78b1ddb3520a1b92d514871b114faa`
- DocVQA: `bd025f78b0348af02747438a31490412aae4d5f4f9708a9c52d5a2b4853749da`
- HRBench: `9ba6846a76d333044509a5bdb6073aa66be292b5777b22537b572ea751b89792`

The sequential test transaction was never created, and no sequential test rollout, feature,
reward, or model evaluation was generated or consumed. During a later inventory audit,
however, metadata and one example row per benchmark from the pre-existing predictability test manifests
were read outside the sequential ledger. Those manifests must therefore not be described as
untouched or reused for any future positive sequential claim. This operational breach does
not create the negative validation result, but it independently disqualifies those identities
from a future formal test; any future, newly formulated study would need a fresh allocation.

## Final answer

**NO-GO: acquisition value exists, but it is not reliably identifiable from the current
pre-action frozen representation. Stop this formulation; do not run RL, 7B, multi-step
acquisition, or another representation search.**
