# Proposal-conditioned rescue/harm gate result v1

Status: completed on 2026-09-01. Registered decision:
**`proposal_conditioned_gate_not_advanced`**. ScreenQA and every protected role
remain sealed.

## Bound execution

- Slurm job `199858`: `COMPLETED`, `ExitCode=0:0`, zero restarts, runtime
  `00:00:39`, one NVIDIA H800, eight CPUs, and all-state email enabled.
- Population: 3,500 opened DocVQA development sources and 13,580 decisions.
- Both policies call exactly 225 questions using outcome-blind, tie-preserving
  score thresholds.
- Report SHA-256:
  `6b1ea97b1de0cd1ebb61036aa6e026151afa8c2e9999d8ed3d7f5cdb1624c160`.
- Score report SHA-256:
  `e04e883eaaa0550cd488d370f97cd25960dcb2e64da09d49a4efdac7391edb5d`.
- Model SHA-256:
  `d088e5c24c3dbb5202af492374d72992d9908cfdb2bddc390bd1ed431561dad9`.
- Outcome-free score rows SHA-256:
  `dd25a5262be1290d32cc7ef9c589f89bafed99d4e54ea79250b4e996e08da365`.
- Completion SHA-256:
  `f09d26b2b0588357cde2f3a620ddd4caf25216dc711d4ac346d5b5e4ea347132`.

All input hashes, source exclusion, 46-dimensional feature schema, exact
positive/negative half-mass weighting, OOF coverage, matched calls, incumbent
call set and pooled metrics, and score-leakage audits passed.

## Mechanical result

| Source-balanced metric | Incumbent | Proposal-conditioned |
|---|---:|---:|
| Utility | 0.00317324 | -0.00014389 |
| Gain | 0.00397661 | 0.00060497 |
| Gain per call | 0.24750 | 0.04039 |
| Induced harm | 0.00034156 | 0.00024372 |
| Negative-value call mass | 0.00977110 | 0.01222383 |
| Helpful-call precision | 0.39441 | 0.18861 |
| Proposal recovery | 0.51641 | 0.64336 |

The candidate-minus-incumbent source-balanced utility difference is
`-0.00331712`, with 95% whole-source interval
`[-0.00481969, -0.00198051]`. The entire interval is negative. Only the audit
clause passes; all five performance clauses fail.

## Post-result mechanism diagnosis

The rescue head still ranks its rare positive label with AUC `0.85235` and
average precision `0.27788`, versus prevalence `0.04728`. The harm head has AUC
`0.74255` and average precision `0.07382`, versus prevalence `0.02585`. The
failure is therefore not absence of all predictive signal.

The registered class balancing deliberately maps each head's positive and
negative training mass to one half. Those independently reweighted
probabilities are not estimates under the natural task prior, yet the
registered score subtracts them directly. At the selected tail, only 12 of 225
candidate calls have baseline correctness below `0.5`; 213 have baseline
correctness at least `0.5`. The incumbent split is 94 / 131. The candidate thus
uses most of its scarce budget where rescue is structurally less likely,
despite choosing ten helpful actions among the twelve low-correctness calls.

This diagnosis was computed only after the registered result was frozen. It is
descriptive opened-development evidence, not a revised endpoint or permission
to tune the failed score.

## Next registered direction

Keep the loss-only proposal fixed, but replace the unconditional probability
difference with the original factorized decision structure trained on that
proposal distribution:

`P(error) * P(rescue | error, proposed action) * rescue_magnitude`

minus

`P(correct) * P(harm | correct, proposed action) * harm_magnitude + cost`.

This is a structural correction to the diagnosed prior mismatch, not a
threshold change. It requires a separate frozen protocol and fresh OOF fit on
opened DocVQA development. No GitHub push is authorized.
