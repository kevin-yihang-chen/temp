# InfographicVQA raw-attention evaluation floating-point recovery v1

Date: 2026-09-02 (Asia/Hong_Kong)

## Incident

Slurm job `203262` stopped before producing a candidate outcome or decision.
The frozen comparator check recomputed
`$.question_balanced.action_selection_regret` as
`0.0007172290756453434`, while the frozen JSON contains
`0.0007172290756453436`. The absolute difference is approximately `2e-19`.

## Frozen recovery rule

- Mapping keys, sequence lengths, strings, booleans, integers, and types remain
  exact.
- Finite floating-point comparator fields use `rel_tol=1e-12` and
  `abs_tol=1e-15` solely to absorb machine-scale reduction-order noise.
- Non-finite values remain invalid.

This does not change the candidate, operating points, thresholds, actions,
population, bootstrap indices, confidence level, qualification rules, or
selection rule. Validation and test data remain sealed. The rerun uses the
same completed feature job `203257` and frozen protocol/resource amendment.

## Recovery outcome contract

The recovery document and its SHA-256 are mandatory evaluation inputs. The
evaluator may write `evaluation-v1` only after all original bindings pass and
all frozen comparator discrepancies are either exactly equal or within the
numeric rule above.
