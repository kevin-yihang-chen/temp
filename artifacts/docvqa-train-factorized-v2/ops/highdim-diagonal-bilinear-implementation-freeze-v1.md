# High-dimensional diagonal-bilinear implementation freeze v1

Status: frozen on 2026-09-01 after implementation and tests, before any formal
high-dimensional OOF fit or candidate score existed.

## Bound artifacts

- model/evaluation module SHA-256:
  `6d04f376a05383ad045b3c855e5c7b98f7cdcbb44a55abe198833a93735823f9`;
- fit/export runner SHA-256:
  `a095216078d25e372377d7d3a7aef28f11480345c94ac59673f8db72599b0882`;
- focused test SHA-256:
  `25552f0eb5529827a0a7efe61a6b50eadc650daa9dbd42e2e8b0f994e166b66c`;
- frozen protocol SHA-256:
  `01159b71ff7aaad02ba1cdc23827a8e4a0003c47a0f1defc4d4ec85e2f658d9f`;
- preceding compact-union result record SHA-256:
  `24fc17a15d18d3b5cab78a14635601add6a955c2626f4e1156532b323d61ed20`.

All highdim/union/factorized/conditioned/decoupled focused tests passed
(`19 passed`). The complete repository suite passed with only existing
optional-runtime skips. Compilation, runner argument parsing, `git diff
--check`, and a real-input 13,580-decision embedding shape/action-alignment
preflight passed.

## Fail-closed properties

The implementation requires finite `(2048,)` question and global embeddings,
finite `(4,2048)` region embeddings, four unique action IDs, and exact selected
action alignment. It L2-normalizes each embedding and verifies the 2,075 state
and 4,142 action dimensions after constructing only registered elementwise
products. Tests prove invariance to independent positive rescaling of each raw
embedding family.

Every head uses `C=0.01`, L2, liblinear, at most 4,000 iterations, its own
fold-local scaler, and the registered unbalanced source/decision/candidate
weights. Whole-source fold exclusion, exact union cardinality, finite OOF
coverage, matched 225 calls, incumbent reproduction, and outcome-free score
serialization all fail closed.

No Qwen parameter is loaded or trained by this experiment; the H800 allocation
provides the cluster execution/CPU-memory environment for sklearn. ScreenQA and
every protected role remain sealed. All Slurm state changes email
`yihangc@connect.hku.hk`. No GitHub push is authorized.
