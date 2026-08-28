# Fresh TextVQA attention action-value formal result

Date: 2026-08-28

## Decision

The frozen policy **failed** its preregistered confirmation rule. Mean utility
and the lower endpoint of the two-sided 97.5% source-bootstrap interval were
both required to be strictly positive. The point estimate was approximately
zero and the interval crossed zero.

| Frozen metric | Formal result |
| --- | ---: |
| Decisions / image sources | 3,166 / 2,000 |
| Answer-now TextVQA score | 0.768572 |
| Policy TextVQA score | 0.773247 |
| Raw score gain | **+0.004675** |
| 97.5% source-bootstrap gain CI | **[+0.001112, +0.008549]** |
| Tool-use rate | 9.2862% (294 calls) |
| Mean utility, `gain - 0.05 * calls` | **+0.000032** |
| 97.5% source-bootstrap utility CI | **[-0.003541, +0.003852]** |
| Gain per call | 0.050340 |
| Unnecessary-call rate | 87.7551% |
| Correct-stopping rate | 87.6816% |
| Oracle action-and-stopping utility | +0.051516 |

The raw gain interval is strictly positive: the fixed attention policy did
improve TextVQA score on unseen sources. However, its gain per call was only
`0.05034`, almost exactly the registered per-call cost of `0.05`. Therefore the
cost-adjusted effect is indistinguishable from zero and the confirmation fails.
No threshold, feature, regularizer, loss, action set, or cost may be revised and
retested on this bank as a formal claim.

## Prospective isolation

This evaluation used 2,000 whole validation image sources at frozen hash-rank
offsets 600--2599, containing 3,166 questions. Before rollout, state, source,
and decoded-RGB audits showed zero overlap with both the original 200-source
development bank and the earlier 400-source formal bank.

The serialized model was fit only on the original development bank and existed
before target rollout. The rollout bank was never inspected for partial
outcomes. Job `190332` completed all 3,166 questions and 15,830 sibling records
with `ExitCode=0:0` in 2:11:16 on one RTX 4090. Slurm email notification was
enabled for all state changes.

Feature job `190619` ran the frozen three-stage original-image, multimodal
question, and final-four-layer question-region attention chain. Slurm's live
job record had expired before the final audit was collected, and cluster
accounting is disabled, so a historical exit-code query was unavailable. The
job nevertheless reached its final mandatory audit, which covered all 3,166
decisions and reported `outcome_fields_present=[]` and
`outcomes_included_metadata=false`; no temporary output remained.

The frozen evaluator was then executed exactly once with 20,000 bootstrap
resamples of whole `source_id` groups, two-sided confidence 97.5%, and seed
`20260828`.

## Development-to-formal comparison

| Metric | Development OOF | Frozen fresh formal |
| --- | ---: | ---: |
| Raw gain | +0.009119 | +0.004675 |
| Tool rate | 9.4340% | 9.2862% |
| Utility | +0.004403 | +0.000032 |

Unlike the earlier context-policy failure, the call rate transferred almost
exactly and the formal raw gain remained positive. The missing quantity was
effect size per call: the gain shrank by roughly half, leaving no margin over
the deployment cost. This narrows the failure from "the policy never helps" to
"the policy does not select a sufficiently high-value subset to pay for its
calls."

## Bound artifacts

- manifest SHA-256:
  `56973583dcd1aa8367d8a4e72f1c84f130864536dac128741a3914ae69ed901d`
- formal rollouts SHA-256:
  `78285c00f88f24e5530027802b0f29d48d2f39b3642d2cb793fba00e5609b2d0`
- frozen model SHA-256:
  `f9b5dc897c5e8499ea5a245b0c512684579a5c6756da9196b628148ccf2c9a76`
- label-free attention features SHA-256:
  `60598bf8009454f8b2bc20f6633baa968c5fbdb6da321cb049814a7c517f4466`
- label-free audit SHA-256:
  `6d553ec8fd4f6351ab24f015adb2eda6df988accc4b0bc92a221d08ba295c46e`
- frozen evaluation SHA-256:
  `bb7814fe7b3d62bf3d458768e80c0f174afb2b5d0c2e2bcba855c3bb7b213e1c`
- rollout code revision:
  `1c3caae033adc34d934e88b08554f7471c0ea511`
- evaluation code revision:
  `31d2617d234886c8edf596a61905a68c006e1fc3`

## Consequence

This result is stronger than the previous failures as evidence for selective
visual acquisition: a source-disjoint fixed policy produced a statistically
positive raw score gain. It is still negative evidence for the registered
cost-sensitive method claim because the utility interval crosses zero and most
calls were non-positive after cost.

The 200-source attention policy family is now closed on all opened TextVQA and
DocVQA formal banks. Subsequent analysis on this target is diagnostic only. A
new positive claim requires the already separated risk-calibration method and
larger TextVQA train source bank, followed by a policy frozen before a new
untouched formal split is opened.
