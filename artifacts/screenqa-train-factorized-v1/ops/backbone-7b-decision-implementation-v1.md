# Qwen2.5-VL-7B backbone decision implementation v1

Status: frozen on 2026-09-01 before the full 512-state Qwen2.5-VL-7B report
or any full-bank endpoint was available. It mechanically implements the four
conditions already registered in `backbone-7b-diagnostic-protocol-v1.md`.

## Bound code

- implementation revision:
  `40d84d7a29ade63c37952245bbee7b05369bc11a`;
- decision module SHA-256:
  `a69f4b098a2e3a7879728085b5efd1e2d68e90c941f690240a616cd9b0a48486`;
- decision CLI SHA-256:
  `08111d528284bb18cc422d5f6113e11bcd869b741445271d807b010c87abd6fd`;
- targeted test SHA-256:
  `707dc5c99aea26fd692eb55e9d45da743fe3359dc2d6cff119e4d3477eb9bf1b`;
- focused decision/runtime/likelihood/smoke tests: `12 passed`;
- full repository test suite: passed, with only the existing optional-runtime
  skips.

## Input guards

The evaluator requires the frozen audit schema and study label, independently
supplied report/protocol hashes, report-bound protocol agreement, population
dimensions `512 / 512 / 2,048 / 2,560`, and exactly 5,000 valid whole-source
resamples with seed `20260903` and confidence `0.95`. Opened development must
be declared; candidate search and every protected outcome-use flag must be
false.

## Mechanical decision

1. answer-loss Spearman 95% lower endpoint greater than zero;
2. answer-loss top-one mean-task-gain 95% lower endpoint greater than zero;
3. answer-loss top-one task-gain point estimate greater than both entropy and
   exact-uniform-random estimates;
4. answer-loss top-one induced-harm point estimate below both entropy and
   exact-uniform-random estimates.

All four passing yields `strong_backbone_replication`. Conditions 1 and 2
passing while condition 3 or 4 fails yields `partial_backbone_replication`.
Any failure of condition 1 or 2 yields `backbone_non_replication`.

The evaluator writes JSON, Markdown, and a completion record without
overwriting prior output. It explicitly records that no score threshold, call
rate, or protected outcome was selected.
