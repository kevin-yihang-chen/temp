# DocVQA attention action-value secondary formal result

Date: 2026-08-28

## Decision

The frozen secondary policy **failed** its preregistered confirmation rule.
Mean utility and the 97.5% source-bootstrap lower bound were required to be
strictly positive. Instead, both the point estimate and the entire interval
were negative.

| Frozen metric | Formal result |
|---|---:|
| Decisions / source documents | 1,608 / 400 |
| Answer-now ANLS | 0.900441 |
| Policy ANLS | 0.897603 |
| ANLS gain | -0.002837 |
| Tool-use rate | 5.7836% (93 calls) |
| Mean utility, `gain - 0.05 * calls` | **-0.005729** |
| 97.5% source-bootstrap utility CI | **[-0.010852, -0.001225]** |
| Gain per call | -0.049059 |
| Unnecessary-call rate | 92.4731% |
| Correct-stopping rate | 89.4279% |

The policy therefore provides negative confirmatory evidence. No threshold,
regularizer, feature, action set, or cost may be tuned and retested on this
partition.

## Isolation and execution audit

The formal rollout bank was completed before feature evaluation. Feature job
`190328` then completed with `ExitCode=0:0` in 24 minutes 16 seconds on one RTX
4090 and had Slurm email notifications enabled for all state changes.

The final feature bank contains 1,608 decisions and was audited both before and
inside the exact evaluation wrapper. Both audits report
`outcome_fields_present=[]` and `outcomes_included_metadata=false`. The model
was evaluated exactly once with 10,000 source-cluster bootstrap resamples,
confidence 97.5%, and seed `20260828`, after the original context primary result
had already been locked.

## Development-to-formal shift

| Metric | Development OOF | Frozen formal |
|---|---:|---:|
| Gain | +0.006551 | -0.002837 |
| Tool rate | 3.8835% | 5.7836% |
| Utility | +0.004609 | -0.005729 |
| Unnecessary calls | 81.25% | 92.47% |

The failure is not explained by a lack of counterfactual headroom. Oracle
action and stopping utility on the same formal bank remains `+0.033935`.
Post-hoc decomposition shows failures in both components:

- call precision for any helpful crop is 19.35%, and realized positive-utility
  call precision is only 7.53%;
- frozen stopping with an oracle action would yield `+0.003299` utility;
- oracle stopping with the frozen learned action would yield `+0.019148`;
- the frozen learned top action rescues 46.77% of helpful states;
- raw question-region attention alone rescues 53.23%, versus 40.32% expected
  for a random crop, but its all-state mean gain remains negative.

Thus the action bank contains useful crops and the pre-action attention signal
retains some localization information, but the learned rescue/harm calibration
does not transfer. Stopping is the larger practical failure because it calls
mostly harmful or unhelpful states, while the learned action head also weakens
the zero-shot attention ranking.

## Bound artifacts

- formal rollouts SHA-256:
  `a7f44c267b11c12f6cbf8f1e714350174c4dfd7e4ab3866fde0dbd84fe0b5aa3`
- frozen model SHA-256:
  `1f8b6cf5d026bcd9921434c1c6ef0c753259d36504dedc040b8145c76bd06ff3`
- label-free attention features SHA-256:
  `bc58c1694261d8e5da91e8d3006a25c29447c0633f8e378796262439f9448ba7`
- label-free audit SHA-256:
  `c605952e150b8de6486ffad2f5f13b08d36ee72578f677a71d402085c806ec2e`
- frozen evaluation SHA-256:
  `a87b99af4f4fe9d20cff2f816bbf6baa8313f0607e0f182e17f58f5e5e7c2c62`
- post-hoc decomposition SHA-256:
  `0d6ebd7ea7bbc22b3d6edfb705e198f01793105cd392d0044d85174d870bb6c6`
- post-hoc attention-ranking report SHA-256:
  `67d9b30b84869ac16f5f6116c964dd97e2857da0135b1b9f25ed3337a29cc438`
- evaluation code revision:
  `455e83e33cb0d10b732a62afa0e30996fd98f7b2`

## Consequence

The current evidence validates the problem formulation and oracle opportunity,
but not the proposed learned policy. The next confirmatory experiment must use
an untouched source partition. The formal DocVQA bank is now analysis-only and
may be used to diagnose invariance failures, never to choose a new positive
claim on itself.

