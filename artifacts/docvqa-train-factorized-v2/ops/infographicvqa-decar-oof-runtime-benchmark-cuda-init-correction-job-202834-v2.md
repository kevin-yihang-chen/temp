# InfographicVQA DECAR runtime benchmark CUDA-initialization correction

Status: frozen on 2026-09-01 after replacement job `202834` failed and before
any successful full-shape timing report or official-train OOF prediction
existed.  This supersedes only the CUDA instrumentation diagnosis in the
job-`201711` engineering correction; every benchmark and scientific setting
remains fixed.

## Second failed execution

Slurm authoritatively recorded job `202834` as `FAILED`, reason
`NonZeroExitCode`, `ExitCode=1:0`, with zero restarts.  It ran for five seconds
on one NVIDIA H800 and again stopped at
`torch.cuda.reset_peak_memory_stats` before the first synthetic fit.  No
runtime report was written.

The first correction correctly replaced the CUDA string with a canonical
`torch.device`, but that was insufficient.  In the pinned PyTorch 2.4 runtime,
`reset_peak_memory_stats` resolves the device index and calls the allocator C
API without performing CUDA lazy initialization.  The benchmark had allocated
only CPU synthetic tensors at that point, so no prior GPU operation had
created a CUDA context.  `torch.cuda.is_available()` proves availability but
does not establish the allocator context required by the stats API.

Bound evidence:

```text
b00dfc10f3822841b877d31977e3080553c1b8142afae2e0794d0216735d3b4b  slurm-infovqa-decar-fitbench-202834.out
5efe034ccca2914df6f1fedd6d5470c7532c3aa109d88b2d3e256badacd76baf  job-201711 correction v1
2c6a05ef460aab86cdd4eeca32404a70900ea490  first-correction code revision
```

Neither failed job read a task image, question, answer, likelihood, model
prediction, validation input, test input, or scientific endpoint.  Both failed
before any model fit and produced no report.

## Minimal correction

Before clearing CUDA caches or resetting peak stats, `_timed_fit` now calls
the public idempotent `torch.cuda.init()` and binds the already canonical
device with `torch.cuda.set_device()`.  The fake-CUDA regression rejects every
stats, synchronization, or device-name call before initialization and verifies
that initialization occurs exactly once for the measured fit.

Corrected assets:

```text
c88bddebd701ff0fbed7b6fed03d3b094f5de489ba4d38bb305b30f90829bfcb  scripts/benchmark_infographicvqa_decar_fit_runtime.py
84f8cbc9737bc06b999b4fcf6e19ac6423a11880f0ac77053d7935b271e89666  tests/test_infographicvqa_decar_fit_runtime.py
424d5c0a47d151111ecaebb34503cd5533d0606b0691e234bebe35906c881643  scripts/slurm_infographicvqa_decar_fit_benchmark_h800.sh
64047e6e315637a80c537aa8f9e24894089354591d69fa88cd911a458ced32d2  scripts/submit_infographicvqa_decar_fit_benchmark_h800.sh
```

This correction may not alter the synthetic population, tensor dimensions,
fit counts, epoch counts, seed, DECAR models or losses, projection formula,
25-percent reserve, H800 requirement, time limit, OOF schedule, features,
targets, policies, baselines, bootstrap, or advancement rule.

## Replacement rule

One replacement may be submitted only after the updated fake-CUDA test,
focused DECAR tests, complete repository regression, mypy, compilation,
formatting, shell, and whitespace checks pass; the tracked worktree is clean;
and the report remains absent.  It retains one H800, 45 minutes, offline
execution, and all supported state-change emails to
`yihangc@connect.hku.hk`.

Formal OOF remains forbidden until a replacement report passes the frozen
runtime-only contract and its 25-percent-reserve projection is used only to
retain or increase the OOF wall-time request.
