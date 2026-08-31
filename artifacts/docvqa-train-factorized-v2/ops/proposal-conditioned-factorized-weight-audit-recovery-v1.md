# Proposal-conditioned factorized weight-audit recovery v1

Status: prepared on 2026-09-01 after failed job `199860` and before any
factorized-conditioned candidate output or result existed.

Job `199860` started immediately on one H800, passed all revision, hash,
accelerator, compilation, and focused-test checks, and failed after 25 seconds
with `ExitCode=1:0`. Slurm recorded `FAILED`, `NonZeroExitCode`, and zero
restarts. All-state email was enabled.

The failure occurred during the full refit after OOF fitting but before the
evaluator returned, before threshold selection, and before the output directory
was created. No candidate score, metric, or decision was available or used.

## Root cause

The source-weight helper analytically normalizes weight mass to the head row
count. On the exact population, IEEE-754 summation produced:

- error head: `13579.999999998905` for 13,580 rows, difference
  `-1.0950316209346056e-09`;
- rescue head: `439.0000000000025` for 439 rows;
- harm head: `13141.000000000507` for 13,141 rows.

The implementation used absolute tolerance `1e-9`, so only the full error
head crossed the engineering guard by approximately `9.5e-11`.

## Recovery

Use a named absolute tolerance `1e-8` for the two weight-mass assertions. This
accepts the observed floating summation noise but still rejects a synthetic
mass error of `1e-4`. The exact weight vector, model inputs, labels, folds,
scalers, solver, seed, magnitudes, score formula, thresholds, bootstrap,
comparators, and advancement conditions are unchanged.

- recovered model/evaluation module SHA-256:
  `54468e2434e22580187a5cc9c182fd818068f2fc7de471983c9455ad3606fef8`;
- recovered focused test SHA-256:
  `cca935e3b778c2e16d88d58c6765df4180458a46679fad33ec2308356299f565`.

The new full-scale tolerance regression, all focused tests (`12 passed`), the
complete repository suite, compilation, and `git diff --check` passed. The
recovery may resume the original frozen experiment in a new output directory
only if the old directory remains absent. ScreenQA remains sealed. No GitHub
push is authorized.
