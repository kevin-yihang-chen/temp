# InfographicVQA DECAR OOF runtime benchmark engineering correction

Status: frozen on 2026-09-01 after synthetic benchmark job `201711` failed
and before any successful full-shape timing report or official-train OOF
prediction existed.  This correction changes only CUDA device normalization
for runtime instrumentation.

## Failed execution

Slurm authoritatively recorded job `201711` as `FAILED`, reason
`NonZeroExitCode`, `ExitCode=1:0`, with zero restarts.  It ran for seven
seconds on one NVIDIA H800.  The failure log ended with:

```text
RuntimeError: Invalid device argument
```

The runner passed the string `cuda:0` to
`torch.cuda.reset_peak_memory_stats`.  PyTorch 2.4 rejected that argument
before the first synthetic fit.  No report was written.  The failed job read
no task image, question, answer, likelihood, model prediction, validation
input, test input, or scientific endpoint.

Bound failure evidence:

```text
ead4f31550977ee17eafad7801ae3eccb481d3e4db5d046a1ea879926c8052d6  slurm-infovqa-decar-fitbench-201711.out
58392547b2aa288847dee56b894cf53ba5fa907647541ef7541f1eff38b3aab3  generation execution job-200130.json
e8d00817171c96d27410db0b6069839e8417740c865dc1b87c3be9ad920bf30f  runtime benchmark freeze v1
```

## Permitted correction

The runner now converts CUDA strings with `torch.device(device)` before
calling CUDA synchronization, peak-memory reset/query, and accelerator-name
APIs.  The original string is still passed to the registered DECAR fitters and
recorded in the report.  A fake-CUDA regression requires every affected
runtime API to receive a canonical device object and retains a CPU control.

Corrected assets:

```text
98a961404ad086ab6142d680a741fd2da8469854621d7441f92e94841d0ce812  scripts/benchmark_infographicvqa_decar_fit_runtime.py
424d5c0a47d151111ecaebb34503cd5533d0606b0691e234bebe35906c881643  scripts/slurm_infographicvqa_decar_fit_benchmark_h800.sh
64047e6e315637a80c537aa8f9e24894089354591d69fa88cd911a458ced32d2  scripts/submit_infographicvqa_decar_fit_benchmark_h800.sh
a820f706311ba43c570d15e8f9d71b56620010efd7db2270f9885acf3d1efa52  tests/test_infographicvqa_decar_fit_runtime.py
```

The correction may not alter the synthetic population, tensor dimensions,
fit counts, epoch counts, seeds, DECAR models or losses, projection formula,
25-percent reserve, H800 requirement, 45-minute limit, OOF schedule, features,
targets, policies, baselines, bootstrap, or advancement rule.

## Recovery rule

A single replacement benchmark may be submitted only after the corrected
assets pass formatting, regression, type, shell, and whitespace checks; the
tracked worktree is clean; the failed output remains absent; and at least 45
GPU-minutes remain.  The replacement retains one H800 and all supported Slurm
state-change emails to `yihangc@connect.hku.hk`.

The replacement report is runtime-only.  Formal OOF fitting remains forbidden
until its output contract passes and the frozen 25-percent-reserve projection
has been used only to retain or increase the OOF wall-time request.

## Verification

- Fake-CUDA regression covers every corrected runtime API and a CPU control.
- Focused DECAR runtime, fit, and evaluation regression passed seven tests.
- The complete 453-test collection exited zero with registered skips retained.
- Focused mypy, Python compilation, Black formatting, shell syntax, and
  whitespace checks passed.
- The failed benchmark report remains absent before replacement submission.
