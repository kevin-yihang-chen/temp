# InfographicVQA DECAR OOF runtime CUDA smoke result

Status: passed on 2026-09-01 before full-shape benchmark recovery and before
any official-train OOF prediction or scientific endpoint existed.

## Execution

Slurm authoritatively recorded job `202898` as `COMPLETED`, `ExitCode=0:0`,
with zero restarts.  It ran for five seconds on one NVIDIA H800 with four CPUs,
64 GiB requested memory, and a ten-minute limit.  State-change email was bound
to `yihangc@connect.hku.hk`.

The smoke used the corrected runner at code revision
`46f37780fd742b36ff9b38cb9a3123254b0798e8` with exactly 20 synthetic
decisions, five synthetic sources, embedding dimension eight, and one epoch.
It executed one instance of every measured fit path:

```text
where_inner         1.1806928378064185 s   peak 72816128 bytes
where_outer         0.0107204290106893 s   peak 72817664 bytes
when_ternary_outer  0.0267730210907757 s   peak 72833024 bytes
when_binary_outer   0.0099052020814270 s   peak 72830976 bytes
```

All four peak-memory values are positive, proving that the real PyTorch 2.4
H800 allocator instrumentation passed the corrected initialization path.

## Contract audit

The persisted report and the complete JSON line in the Slurm log are
semantically identical.  The report asserts:

- synthetic inputs only;
- no task outcome read;
- no scientific endpoint computed;
- no validation or test input used; and
- no Hugging Face credential present.

Bound evidence:

```text
895f7e865adbf18fe3c616a4f2ee8ca504c94d2916036c14249cbb42a5a4ca9c  smoke report.json
925297a99f98fccb83408c79e0865c5e4aab05bb3c4ecd8306d40df390f723d2  slurm-infovqa-decar-cudasmoke-202898.out
c88bddebd701ff0fbed7b6fed03d3b094f5de489ba4d38bb305b30f90829bfcb  corrected benchmark runner
635b31b897c85907c3602204bea4dcad60344e16d891eb2aff30f3675fbff837  CUDA-init correction v2
```

Immediately after job completion, one login-node lookup briefly did not see
the report while the Slurm log already contained its JSON.  A subsequent
shared-filesystem lookup found the original 1,571-byte file with the job-end
timestamp.  No recovery, reconstruction, or rewrite occurred.

## Decision

The real-H800 smoke authorizes one replacement full-shape runtime benchmark
with the otherwise unchanged frozen configuration.  Its small-shape runtime
projection is intentionally ignored and may not set the OOF time limit.  Only
the successful full-shape report and its frozen 25-percent reserve can retain
or increase the registered OOF wall-time request.
