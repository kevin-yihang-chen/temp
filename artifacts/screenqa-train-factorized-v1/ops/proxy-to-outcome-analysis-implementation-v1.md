# Proxy-to-outcome analysis implementation contract v1

Status: frozen on 2026-08-31 after a one-decision engineering smoke and before
answer likelihood was measured for more than one ScreenQA decision.  The smoke
was used only to validate model loading, answer-span scoring, atomic resume, and
the absence of raw target text.  Its proxy/outcome values are not used to choose
the definitions below.

This document resolves implementation details left implicit by
`proxy-to-outcome-audit-protocol-v1.md`.  It does not alter that protocol's
population, endpoints, or decision boundary.

## Merge and population checks

- Merge exactly four state-aligned shards with indices 0 through 3.
- Require 14,511 complete decisions, 72,555 score rows, 1,510 whole-source
  groups, one ANSWER row and four ZOOM rows per decision.
- Verify every shard's manifest, rollout, model, model revision, code revision,
  target rule, configuration hash, output hash, and raw-target prohibition.
- Sort decisions by `(state_id, replicate_id)` and actions with ANSWER first,
  then lexicographic `action_id`.

## Point estimands

- Compute proxy correlations across all 58,044 ZOOM actions.  Pearson uses raw
  values.  Spearman uses average ranks for ties.
- For answer-loss, entropy, and oracle top-one selection, maximize the relevant
  value within each decision and break ties by lexicographically first
  `action_id`.
- The random top-one comparator is the exact uniform expectation over all four
  crops in each decision.  It has no Monte Carlo selection seed.
- A helpful decision has at least one crop with positive signed task gain.
  Helpful-state rescue is the fraction/probability of helpful decisions for
  which the selected crop has positive task gain.
- Induced harm means negative signed task gain.  Task-gain regret is oracle task
  gain minus selected task gain.  Utility regret is defined analogously at
  `lambda=0.05`.

## Descriptive call-rate grid

- Rank decisions by their selected top-one proxy score, descending, with
  `(state_id, replicate_id, action_id)` as the deterministic tie-break.
- Convert every target call rate to the nearest integer number of decisions,
  with exact halves rounded upward and a minimum of one call.
- Report the achieved call rate and the inclusive boundary score.
- Policy utility and task gain use all decisions in the denominator; gain per
  call, induced harm, and unnecessary calls use executed calls in the
  denominator.  A call is unnecessary when its realized utility is nonpositive.
- Helpful-state rescue uses every helpful decision in the denominator.
- Every grid threshold is descriptive on the opened development bank and is
  forbidden for deployment, calibration, or formal evaluation.

## Uncertainty

- Use 2,000 iid whole-source bootstrap resamples, seed `20260831`, and two-sided
  95% percentile intervals.
- Retain every decision and action of each sampled source.  Repeated sources are
  represented by integer multiplicities.
- Recompute Pearson from multiplicity-weighted sufficient statistics and
  Spearman from exact multiplicity-weighted midranks.
- For the descriptive grid, condition outcome intervals on the full-bank
  ranking and threshold rather than reselecting a different top-k set inside
  each resample.  This must be stated in the report.

## Disagreement and reporting

- Count disagreements at the ZOOM-action level:
  1. positive answer-loss gap with negative task gain;
  2. positive task gain with nonpositive answer-loss gap.
- Report counts, rates over all ZOOM actions, distinct source counts, and at
  most 25 identifier-only examples.  Never write raw target text.
- Emit machine-readable JSON, a Markdown rendering, and a completion record
  binding their hashes.
- The output remains a retrospective development-only measurement audit.  It
  cannot reopen ScreenQA ranker search or any sealed calibration, formal,
  reserve, validation, or test role.
