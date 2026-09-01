# InfographicVQA DECAR OOF full-shape runtime benchmark freeze

Status: implementation frozen on 2026-09-01 while full generation job
`200130` was still running and before any official-train OOF prediction or
scientific endpoint existed.  The benchmark is synthetic and may affect only
the Slurm wall-time request for the already frozen OOF computation.

## Purpose and exact schedule

The prior H800 smoke proved execution but used a tiny synthetic population; it
does not establish that the four-hour OOF request is sufficient for all 65
registered 200-epoch fits.  This benchmark allocates float32 tensors at the
real full shape: 23,946 decisions, 2,204 sources, 3,584-dimensional question,
global, and four-region embeddings, four candidates, and 16 scalar features.
No image, question, answer, task score, likelihood, prediction, validation
input, or test input is read.

It measures five epochs of one 60%-population `where` fit, one 80%-population
`where` fit, one 80%-population three-class `when` fit, and one 80%-population
binary `when` fit on one H800.  The registered nested schedule is represented
exactly as:

```text
40 inner where fits
10 outer where fits
10 outer ternary when fits
 5 outer binary when fits
200 registered epochs per fit
```

The raw projection scales each measured fit by `200/5` and the exact fit
count.  A second projection adds a fixed 25% reserve for prediction, hashing,
serialization, evaluation, and benchmark-to-real-data variance.  The result
may only increase or retain the OOF Slurm wall-time.  It cannot change epochs,
architecture, folds, features, targets, action set, losses, operating points,
baselines, bootstrap, or advancement criteria.

## Frozen implementation

```text
2351914fe75d56301a60a753dc2a9152e7dd274808c6cff18fc7f22658a1b77d  scripts/benchmark_infographicvqa_decar_fit_runtime.py
424d5c0a47d151111ecaebb34503cd5533d0606b0691e234bebe35906c881643  scripts/slurm_infographicvqa_decar_fit_benchmark_h800.sh
64047e6e315637a80c537aa8f9e24894089354591d69fa88cd911a458ced32d2  scripts/submit_infographicvqa_decar_fit_benchmark_h800.sh
```

The worker requests one H800, eight CPUs, 128 GiB, and 45 minutes; enables
deterministic CUDA behavior; removes credentials and network proxy variables;
refuses overwrite; and emails `yihangc@connect.hku.hk` for all Slurm state
changes.  The output contract explicitly records that only synthetic inputs
were used and that no scientific endpoint or validation/test input was read.

## Pre-submission verification and scheduler state

- CPU smoke completed all four fit types and wrote the expected contract.
- Python compilation, Black, focused mypy, shell syntax, whitespace checks,
  and the complete repository regression passed.
- Local commit `226267a` contains the three benchmark files; it was not pushed.
- The first H800 submission attempt created no job.  Slurm rejected it with
  `QOSMaxSubmitJobPerUserLimit`.
- The live `normal` QOS record for `yihangc` reported
  `MaxSubmitJobsPU=1(1)`: the sole submitted-job slot was occupied by healthy
  generation job `200130`.

Do not cancel or requeue `200130` for this benchmark and do not retry while it
is submitted.  After `200130` reaches a verified terminal state and releases
the slot, submit this benchmark first.  If its 25%-reserve projection fits
within four hours with operational margin, retain the frozen OOF request.  If
not, increase only the OOF Slurm time limit under a dated compute-only
amendment before OOF submission.
