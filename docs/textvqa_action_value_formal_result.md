# TextVQA frozen action-value formal result

Status: completed failed confirmation under
`textvqa_action_value_formal_preregistration.md`. The frozen primary is not
modified after observing this result.

## Integrity anchors

| Artifact | SHA-256 |
| --- | --- |
| Formal manifest (633 states / 400 sources) | `847899f91147633186b61a802004c49cfe8ef3258427cb92ea390c891ec5ef2c` |
| Formal sibling rollouts (3,165 records) | `07af632e8beb15b7784e4c16c7fc8fc1b6ae6734da6ecfd54726e77c01615c2d` |
| Frozen development model | `ca224964aeb429478aeffaa3f084750cab05daf2c56be0b3f70fda68dceadc33` |
| Frozen evaluation report | `12e6f2e1f47c85ff3074c60d726ae31012a88c865c98b21309566c97d9063141` |
| Formal action-bank report | `a2cb5c7b1a217fb314352b812ef7812f605eac56f260e8e7e2e9087f3769d5cc` |
| Post-hoc failure decomposition | `40bdce70fa97449d63cd940e8c2efc3c86376e33c78d1da3269dae1eb7e9bfc0` |

The Slurm collector finished 633/633 examples with exit code zero. Provenance
records the frozen manifest/model settings and collection code revision
`fcb7ad2a4e45d359921d0dde34fe75039b53beae`.

## Preregistered primary decision

| Metric | Formal result |
| --- | ---: |
| Answer-now TextVQA score | 0.75877 |
| Frozen policy score | 0.76319 |
| Task-score gain | +0.00442 |
| Tool rate | 0.26856 |
| Gain per call | 0.01647 |
| Mean utility at `lambda=0.05` | **-0.00900** |
| Utility 95% source-cluster CI | **[-0.02022, 0.00252]** |
| Unnecessary-call rate | 0.92941 |

The preregistered pass rule required positive mean utility and a confidence
interval lower bound above zero. Both conditions fail. This is a failed
confirmation, even though the raw task-score point estimate increases.

## Fixed secondary baselines

| Policy | Score gain | Calls | Utility |
| --- | ---: | ---: | ---: |
| Answer now | 0 | 0 | 0 |
| Uniform random crop expectation | -0.00407 | 1 | -0.05407 |
| Fixed center crop | -0.00790 | 1 | -0.05790 |
| Exhaustive lowest-entropy crop | +0.01912 | 4 | -0.18088 |
| Action-and-stopping oracle | +0.06066 | 0.08057 | +0.05664 |

Oracle utility remains strongly positive, with 95% CI `[0.04068, 0.07328]`.
Thus this failure does not arise because the four crops contain no useful
counterfactual information. Exhaustive entropy selection also improves score,
but only after paying for all four observations; its utility CI is entirely
negative. This preserves the problem motivation while rejecting the current
learned solution.

## Post-hoc diagnosis

This section is explicitly diagnostic and cannot change the formal decision.

- The frozen gate called 170 times. Only 13.5% of those states had any
  positive-gain crop, and only 7.1% of calls realized gain above the 0.05 cost.
- Its top-ranked crop rescued 40.4% of helpful states, versus 38.5% for a
  uniform-random crop: action ranking transferred only weakly.
- Keeping frozen stopping but substituting the oracle crop yields utility
  `+0.00964`.
- Keeping frozen crop ranking but substituting oracle stopping yields utility
  `+0.02275`.
- Oracle stopping and action together yield `+0.05664`.

Both components lose value under shift, with false-positive stopping calls the
clearest failure. No alternate threshold or feature is selected on this formal
bank. Any replacement must be developed elsewhere and confirmed on a new
outcome-unseen dataset or benchmark split.
