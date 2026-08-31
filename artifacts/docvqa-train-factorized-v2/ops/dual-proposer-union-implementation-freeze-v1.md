# Dual-proposer union factorized implementation freeze v1

Status: frozen on 2026-09-01 after implementation and tests, before any formal
dual-union OOF fit or candidate score existed.

## Bound artifacts

- model/evaluation module SHA-256:
  `632863f40a254c4071a2c965ce769539d47bd307e0490a32dc4e370f61c4cc11`;
- fit/export runner SHA-256:
  `dfb6322f12630a7a86fa2b82b719a2d2f0f4c332986cac4d6efcd2d9a760152f`;
- focused test SHA-256:
  `2f6c6c6de66a2fdb92406a31df0be9569fc619a170d4d63eae16c713d9943d1a`;
- frozen protocol SHA-256:
  `b08fab22a4e1d13457186537fc658efeaa20bc6a5b2bf4dd42e870fa7625b625`;
- preceding factorized-conditioned result record SHA-256:
  `e1495b68498053dd66fb610a13852c415b8ff952a0799ba702fc78234d0f5eab`.

All union/factorized/conditioned/decoupled focused tests passed (`16 passed`).
The complete repository suite passed with only existing optional-runtime skips.
Compilation, runner argument parsing, `git diff --check`, and a real-input
identity-only cardinality preflight passed.

## Fail-closed properties

The implementation requires exact union counts `4,875` equal proposal pairs,
`8,705` unequal pairs, and `22,285` unique candidate rows. It constructs these
only from the hash-bound OOF loss and incumbent proposal identities, validates
every selected action against the four rollout siblings, and encodes only the
registered 27 state and 46 action features.

Every gate fold excludes whole test sources. Error-head weights equalize domain,
source, and decision. Rescue/harm weights additionally equalize unique candidates
within each decision, so a two-proposal state has the same total training mass
as a one-proposal state. All weights are row-normalized without class balancing.
Every head requires both classes and fails on nonconvergence.

At test time, candidate scores are finite, proposal-only, and action ties use
lexicographic IDs. Thresholds consume only scores and identities. The candidate
and incumbent must each call exactly 225 states, and incumbent call identities
and frozen pooled metrics must reproduce before any outcome comparison.
Serialized score rows forbid task outcomes.

ScreenQA and every protected role remain sealed. All Slurm state changes email
`yihangc@connect.hku.hk`. No GitHub push is authorized.
