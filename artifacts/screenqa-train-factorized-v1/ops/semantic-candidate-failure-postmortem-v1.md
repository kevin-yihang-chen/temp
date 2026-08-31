# ScreenQA semantic candidate failure postmortem v1

Date: 2026-08-31

Scientific status: immutable interpretation of the sole preregistered ScreenQA
v2 semantic candidate.  The fit job completed successfully; the method gate did
not pass.  Calibration, formal, reserve, untouched, and official validation/test
outcomes remain unopened.

## Terminal result

- Feature job `197065`: `COMPLETED`, exit `0:0`, 14,511 decisions and 1,510
  sources, final feature SHA-256
  `4de090c6b03c56aa68d6d2cab990a0055d539967834f82459ad78a1fdb254b1b`.
- Fit job `197082`: `COMPLETED`, exit `0:0`, selected alpha `100`.
- OOF call rate: `0.0018606574`.
- OOF mean task gain: `0.0002067397`.
- OOF mean cost-adjusted utility at lambda `0.05`: `0.0001137068`.
- Unnecessary-call rate: `0.8888889`.
- Candidate status: `eligible=false`, `candidate_frozen=false`,
  `tail_selection_status=no_non_degenerate_safe_threshold`.
- Candidate audit SHA-256:
  `b555e47623a78a24d5e0514e61799570e5ab47b7effcc0d71682b1548e26aeef`.
- Semantic fit audit SHA-256:
  `65dc141d3bd93199382ddcfabca795066210c0e6ba3464e3b67aade0b270986a`.

## Why it failed

The failure is not explained only by a conservative tail threshold.

1. At the registered 0.5% pooled-call tail, both risk tests pass, but the
   source-balanced call rate is only `0.0061608`, below the `0.01` non-degeneracy
   floor, and source-balanced utility is negative (`-0.00034165`).
2. At the 1% tail, source call rate becomes non-degenerate (`0.0114265`), but
   utility remains negative (`-0.00062442`) and the net-negative-call-mass test
   fails.
3. Every more permissive registered tail has negative source-balanced utility;
   induced harm and negative-call risk failures accumulate.
4. Among the 466 decisions with at least one helpful crop, the selected
   alpha-100 model rescues `42.49%`; matched random crop rescues `45.76%`.
   Thus the central deficit is action ranking, not merely stopping.
5. Oracle cost-adjusted value remains large (`0.0305079`), so the bank contains
   useful visual actions; the learned pre-action representation does not
   identify them reliably.

## Binding decision

Protocol v2 requires ranker development to stop.  No additional representation,
relaxed bound, or threshold may be selected on this ScreenQA development bank,
and calibration may not be opened.  Follow-up work may use the already opened
bank only for clearly labeled retrospective diagnostics that cannot become a
replacement ScreenQA candidate.
