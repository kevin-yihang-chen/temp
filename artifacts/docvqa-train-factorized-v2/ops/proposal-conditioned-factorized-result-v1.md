# Proposal-conditioned factorized gate result v1

Status: completed on 2026-09-01. Registered decision:
**`proposal_conditioned_factorized_gate_not_advanced`**. ScreenQA and all
protected roles remain sealed.

## Execution and recovery

Job `199860` failed after 25 seconds on one H800 because the full-refit
source-weight sum differed from its analytic row count by
`1.0950316209346056e-09`, just beyond the engineering tolerance `1e-9`.
It produced no output directory, candidate scores, thresholds, metrics, or
decision. The bound recovery changed only that assertion to tolerance `1e-8`,
added a regression that still rejects an error of `1e-4`, and preserved all
scientific computations.

Recovery job `199863` completed on one H800 in `00:00:41`, exit `0:0`, zero
restarts, with all-state email enabled.

- Report SHA-256:
  `32c4181a0149c9a245c676974e7c0792b79287e5e8f8b16b55070950532be539`.
- Score report SHA-256:
  `c9777479eb5cca8d4b4aa127e456acaea95eba719bea156cd68d0bfc0950fea7`.
- Model SHA-256:
  `8beecd917dec7b75d91ceff4aae8aa37bc9b5cef6a7e7bf243647f74630b266d`.
- Outcome-free score rows SHA-256:
  `5153f863078eecc7b48f8c9455351cf02057911937c0dd5038042c9b148493a1`.
- Completion SHA-256:
  `c529e7766a809ec88568ece79ec684323b82c957b496735d3cb57c1211722cb9`.

All registered audits passed.

## Mechanical result

| Source-balanced metric | Incumbent | Factorized-conditioned |
|---|---:|---:|
| Utility | 0.00317324 | 0.00206747 |
| Gain | 0.00397661 | 0.00291129 |
| Gain per call | 0.24750 | 0.17251 |
| Induced harm | 0.00034156 | 0.00052332 |
| Negative-value call mass | 0.00977110 | 0.01131600 |
| Helpful-call precision | 0.39441 | 0.33401 |
| Proposal recovery | 0.51641 | 0.64336 |

The candidate-minus-incumbent source-balanced utility difference is
`-0.00110577`, with 95% whole-source interval
`[-0.00184517, -0.00042278]`. The whole interval remains negative. Only the
audit clause passes; all performance clauses fail.

The registered factorization is nevertheless a real recovery from the
unconditional candidate utility `-0.00014389`. Explicit error conditioning
fixes most of the prior mismatch but does not beat the incumbent. Candidate
and incumbent call indicators disagree on only `0.9278%` of decisions, while
their proposed actions disagree on `64.10%`. The residual failure is therefore
predominantly candidate-action selection in the high-score tail rather than a
gross stop/call-rate error.

## Closed route and remaining opportunity

At the incumbent's fixed 225-call set, a perfect per-state oracle choosing
between incumbent and loss-only actions would improve source-balanced utility
by only `0.00022831`, below the registered `0.00025` margin. A fixed-call-set
arbitrator is therefore closed without fitting.

Across all opened-development states, however, the union of the two OOF
proposals contains a helpful action in `73.59%` of helpful-state source mass,
compared with `64.34%` for loss-only and `51.64%` for incumbent. The proposals
differ on 8,705 of 13,580 decisions. The next admissible route is one candidate
scorer trained on this deduplicated two-proposer union, allowed to choose both
the action and the 225-call tail. It must be frozen separately before fitting.

No GitHub push is authorized.
