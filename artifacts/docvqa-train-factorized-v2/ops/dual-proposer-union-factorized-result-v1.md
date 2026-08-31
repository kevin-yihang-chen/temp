# Dual-proposer union factorized gate result v1

Status: completed on 2026-09-01. Registered decision:
**`dual_proposer_union_factorized_gate_not_advanced`**. ScreenQA and all
protected roles remain sealed.

## Bound execution

- Slurm job `199865`: `COMPLETED`, `ExitCode=0:0`, zero restarts, runtime
  `00:00:53`, one NVIDIA H800, eight CPUs, all-state email enabled.
- Population: 3,500 sources, 13,580 decisions, 4,875 equal proposal pairs,
  8,705 unequal pairs, and 22,285 deduplicated union candidates.
- Both policies call exactly 225 questions from outcome-blind score thresholds.
- Report SHA-256:
  `268323a18f8d826f302629a3b80a65131710182fdb41275be072bb0d247e1797`.
- Score report SHA-256:
  `1acfafa8f50429dc091b91431fca47f71ac28a482b0c2dcf06404a732a78ec75`.
- Model SHA-256:
  `8820ac4528245db3745582882c62cb9aa2cae0fefa7a2f398120e590e18ee3cb`.
- Outcome-free scores SHA-256:
  `f6c78cf7bdfd425c7693aaccd816ebf7192c86ce2741a36785238c42d61dd6d5`.
- Completion SHA-256:
  `383f9f729da7dc9f0664512a66fc4d3288b97c0d4d3deb2286c0dda52a82f688`.

Every registered input, cardinality, weighting, feature, source-exclusion, OOF,
matched-call, incumbent-reproduction, finite-score, and leakage audit passed.

## Mechanical result

| Source-balanced metric | Incumbent | Dual union |
|---|---:|---:|
| Utility | 0.00317324 | 0.00264814 |
| Gain | 0.00397661 | 0.00347065 |
| Gain per call | 0.24750 | 0.21098 |
| Induced harm | 0.00034156 | 0.00054709 |
| Negative-value call mass | 0.00977110 | 0.01054842 |
| Helpful-call precision | 0.39441 | 0.36124 |
| Proposal recovery | 0.51641 | 0.54146 |

The candidate-minus-incumbent source-balanced utility difference is
`-0.00052510`, with 95% whole-source interval
`[-0.00107299, -0.00001050]`. The upper endpoint is close to zero but remains
strictly negative. Only the audit clause passes.

The selected dual-union action differs from the incumbent on `12.09%` of all
decisions, and its call indicator differs on `0.7658%`. The candidate does
recover more helpful states than the incumbent, but realizes only `54.15%`
source-balanced recovery despite the bound union containing `73.59%`. The
remaining gap is candidate discrimination inside the union, not candidate-set
coverage.

## Representation conclusion

The 46-dimensional hybrid feature exposes only compact cosine, attention,
geometry, and text/confidence summaries. The frozen feature artifact also
contains complete 2,048-dimensional question, global-image, and per-region
embeddings. Continuing to change folds, thresholds, or minor head structure in
the same compressed feature space is closed by this result.

The next admissible development route retains the exact union and factorized
risk structure but replaces compact semantic summaries with strongly
regularized diagonal bilinear interactions from the frozen embeddings. This is
a representation change and must be separately frozen before fitting.

No GitHub push is authorized.
