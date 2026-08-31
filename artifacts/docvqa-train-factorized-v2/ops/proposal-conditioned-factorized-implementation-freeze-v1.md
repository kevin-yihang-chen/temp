# Proposal-conditioned factorized gate implementation freeze v1

Status: frozen on 2026-09-01 after implementation and tests, before any formal
factorized-conditioned OOF fit or candidate score existed.

## Bound artifacts

- model/evaluation module SHA-256:
  `71bb48049718ffdb48408829ec03a38011cd440807310098160438391d9c2eba`;
- fit/export runner SHA-256:
  `656b33fbaed84ebdac4f1c76ae06eb258c368007781301ae417f03b6aeab0f02`;
- focused test SHA-256:
  `d12721546935a3b7de3c85a59b0f5c5b7337418b64b3e4c91876a5396a6f53d8`;
- frozen protocol SHA-256:
  `bcf0d50e914682206a706731f177b8242bfb775c09f8fb147fc630fd402571ed`;
- preceding unconditional result record SHA-256:
  `f22ff1640dd7f26f161acbe6673e85293672652d29b35a050649d8e375de95d2`.

The factorized, unconditional-conditioned, and decoupled focused tests passed
(`11 passed`). The complete repository suite passed with only existing optional
runtime skips. Compilation, runner argument parsing, and `git diff --check`
passed.

## Fail-closed properties

The implementation requires the exact 3,500-source / 13,580-decision /
four-crop population, the bound OOF loss proposal, 27 state features, 46 action
features, five whole-source folds, seed `20260907`, 20,000 source resamples,
and preverified input hashes. Each gate fold proves zero train/test source
overlap.

The error head consumes every train decision. Rescue consumes only baseline
correctness below `0.5`; harm consumes only the complementary rows. All three
heads require both classes, use independent fold-local standardization, use
equal domain/source/row mass normalized to their row count, explicitly forbid
class balancing, and fail on nonconvergence. Positive rescue/harm magnitudes
are computed only with the registered head weights.

The evaluator refuses to continue unless both policies produce exactly 225
calls, the incumbent call set is byte-for-byte reproducible from its bound
scores, and frozen pooled gain, utility, and call rate reproduce to absolute
tolerance `1e-15`. Threshold selection receives scores and identities only.
Serialized scores contain no task outcomes.

ScreenQA and all protected roles remain sealed. All submitted Slurm tasks use
all-state email for `yihangc@connect.hku.hk`. No GitHub push is authorized.
