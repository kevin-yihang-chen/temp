# High-dimensional diagonal-bilinear union result v1

Status: completed on 2026-09-01 under the frozen opened-DocVQA development
protocol. The mechanical decision is
**`highdim_diagonal_bilinear_union_not_advanced`**. ScreenQA and every
protected role remain sealed.

## Bound execution

- Slurm job `199871`, one NVIDIA H800, 12 CPUs, 192 GiB, completed in
  `00:06:54`, exit `0:0`, zero restarts, with all-state email enabled.
- Code revision:
  `3b2fa775c9917b72735eaa32b223b014ffb8487d`.
- 3,500 sources, 13,580 decisions, 4,875 equal proposer pairs, 8,705 unequal
  pairs, and 22,285 unique union rows.
- Five whole-source OOF folds, 20,000 whole-source bootstrap resamples, seed
  `20260909`.
- Report SHA-256:
  `23b02f44df007f26cb86ccca3d6f555d3d4953be43ff4616caaf88f539e45cca`.
- Score-report SHA-256:
  `958a7e10ca37ac933a10ceba7732d2ebdb7f0c698ddc905c7d5699dff7464edf`.
- Outcome-free score rows SHA-256:
  `fe6a03ef7feda4f25ca9cb74f2e38f93cd4dd3083c59fffadff0c33796fa663d`.
- Model SHA-256:
  `4aee308fff108adcf29903c341544ec4e57d30707b205dd085e09a08ea96c49d`.
- Completion SHA-256:
  `3d198b1ff973cbb13659c2fc0096687eaca2a67862fc3f68517dbdb920504c0f`.

The first two actual submissions were rejected before job creation because the
original 12-CPU/256-GiB request exceeded the partition's 20-GiB-per-CPU rule,
while a 16-CPU request exceeded the node's 12 restricted cores per H800. Slurm
`--test-only` did not expose the second-stage filter. Reducing the operational
request to 12 CPUs and 192 GiB changed no code, input, feature, model, fold,
threshold, or statistical contract. Peak measured RSS was about 6.2 GiB.

## Registered result

Both methods execute exactly 225 pooled calls. Source-balanced metrics are:

| Metric | Incumbent | High-dimensional union |
|---|---:|---:|
| Utility | 0.00317324 | 0.00166351 |
| Raw gain | 0.00397661 | 0.00236734 |
| Gain per call | 0.247496 | 0.168175 |
| Helpful-call precision | 0.394407 | 0.373706 |
| Induced harm | 0.000341563 | 0.000453850 |
| Negative-value call mass | 0.00977110 | 0.00888757 |
| Helpful-state proposal recovery | 0.516410 | 0.628212 |

The registered highdim-minus-incumbent source-balanced utility difference is
`-0.00150973`, with 95% interval `[-0.00277221, -0.000362087]`. The interval
is entirely negative. The utility margin, paired lower endpoint, gain per call,
joint harm/negative-call constraint, and helpful-call precision clauses all
fail. Every data hash, feature dimension, embedding/action alignment,
source-exclusion, weighting, matched-call, incumbent-reproduction, finite-score,
and no-leakage audit passes.

## Frozen-result failure decomposition

A post-hoc decomposition changed no model or threshold and used only the
already opened DocVQA outcomes. It independently reproduced the complete
frozen result, then crossed the two call sets and two action selectors. Report
SHA-256:
`a0d9fc7dfd8684cfc03d093a5a481edc6640a3e67813f9e6278ba8d8fd189887`.
Its 20,000 source bootstraps use seed `20260910` and are descriptive rather
than a new registered claim.

| Composition | Source-balanced utility | Difference from incumbent (95% CI) |
|---|---:|---:|
| Highdim call set + incumbent action | 0.00175501 | -0.00141823 [-0.00264143, -0.000302857] |
| Incumbent call set + highdim action | 0.00291724 | -0.000255998 [-0.000613309, 0.0000254350] |
| Incumbent call set + union-oracle action | 0.00340155 | +0.000228312 [0.0000718998, 0.000438805] |

The stopping score is the dominant realized failure: applying the highdim call
set even to the incumbent action remains significantly worse. The highdim
action selector improves global helpful-state coverage but is slightly worse
inside the incumbent's high-value call states. Even a non-deployable oracle
choosing between the two union actions on the incumbent call set improves by
only `0.000228312`, below the registered `0.00025` advancement margin.

There is still learnable headroom outside that fixed call set. An outcome
oracle stopping on the frozen highdim action obtains utility `0.0138949` at a
source-balanced 4.04% call rate; an oracle over both union stopping and action
obtains `0.0160439` at 4.74%. These are ceilings, not deployable evidence.

## Consequence

Do not tune diagonal-bilinear regularization or add another independent
rescue/harm classifier on this result. The high-dimensional rescue folds have
only roughly 500 union rows for 4,142 features, while the factorized score must
multiply three rare-event probability estimates. The observed failure is
therefore consistent with an unstable stopping tail rather than lack of any
useful action signal.

The next candidate should change the learning target and decomposition, not
only the representation: learn a full-four-action pairwise crop ranker and a
separate continuous signed-gain head, using whole-source nested OOF actions.
Freeze its regularization from the already completed external TextVQA branch
before any DocVQA fit, compare at the same 225-call budget, and retain the same
utility/harm advancement rule. No ScreenQA outcome may be opened unless that
new candidate passes and receives a separate deployment/calibration freeze.

No GitHub push is authorized by this result.
