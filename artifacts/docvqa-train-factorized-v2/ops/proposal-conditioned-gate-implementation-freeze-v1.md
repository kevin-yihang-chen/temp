# Proposal-conditioned rescue/harm gate implementation freeze v1

Status: frozen on 2026-09-01 after implementation and tests, before any formal
proposal-conditioned OOF fit or candidate score existed.

## Bound implementation

- gate/evaluation module SHA-256:
  `e2c9e2340fbbf4485590532881cc4c0785cbbfd67563689cb2f1891d242486e4`;
- fit/export runner SHA-256:
  `a8c576a60ec8fc08e68f71845e4210c321b6eca1d6d297101c27e1d37aad565f`;
- focused test SHA-256:
  `15fa97331c869c01ee5424488daa3be285baf36a33bdc75e00c1719475d1ba7c`;
- frozen protocol SHA-256:
  `043fe88c8115e350a2b292f0bec19b8690769ec94efb9e6a8b898d882f2d0203`;
- preceding decoupled result record SHA-256:
  `4d93d75a17fbbcfc00cc700984bc381b6555a81c57826aa3103348c89b649f9b`.

The focused proposal-conditioned plus decoupled tests passed (`8 passed`). The
complete repository suite passed with only existing optional-runtime skips.
Python compilation, runner argument parsing, `git diff --check`, and a read-only
13,580-row identity/schema/outcome-free audit of both bound score inputs passed.

## Fail-closed implementation properties

The runner verifies every frozen input and protocol hash before reading data,
refuses an existing output directory, and writes JSON with non-finite values
forbidden. The evaluator requires exactly 3,500 sources, 13,580 decisions,
four crop actions per decision, 46 finite pre-action features, five whole-source
folds, the registered seed, 20,000 resamples, and the audited loss-proposer
report contract.

Each fold proves zero source overlap. Both binary heads require both classes,
give positive and negative rows exactly one half of normalized training mass,
and fail if liblinear reaches its maximum iteration count. Candidate and
incumbent thresholds consume only scores and identities. The incumbent 225-call
set and its frozen pooled gain, utility, and call rate must reproduce exactly
before any candidate outcome comparison is returned.

Serialized OOF rows contain identities, actions, scores, call indicators, and
rescue/harm probabilities only. Raw or derived task outcomes are forbidden.
The deployment artifact contains the five OOF audit models and one full refit;
it composes with the already frozen full loss-only proposer only after a passing
decision.

ScreenQA and every protected role remain sealed. Every submitted Slurm task
must use all-state email for `yihangc@connect.hku.hk`. No GitHub push is
authorized.
