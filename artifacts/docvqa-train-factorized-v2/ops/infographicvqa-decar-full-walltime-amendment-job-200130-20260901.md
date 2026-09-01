# InfographicVQA DECAR full compute-only wall-time amendment

Decision time: 2026-09-01T15:37:04+08:00

Scope: Slurm job `200130` only.  This amendment was triggered exclusively by
wall-clock state, atomic checkpoint row counts, and file timestamps.  No task
score, answer, teacher likelihood, feature value, policy prediction,
validation input, or test input was read.

## Original allocation

- four NVIDIA H800 GPUs on `q-h800`;
- original time limit `08:15:00`;
- start `2026-09-01T11:10:31+08:00`;
- original hard end `2026-09-01T19:25:31+08:00`.

The merged rollout file was written at `2026-09-01T15:28:41+08:00`, after all
four source shards reached exactly `119,730` five-action rows.  Thus rollout,
resume audit, and merge consumed approximately 4 hours 18 minutes before NLL
model loading.

## NLL throughput evidence

The four NLL shard row counts were:

```text
time                       shard 0  shard 1  shard 2  shard 3  total
2026-09-01T15:34:13+08:00     1120     1120      960      800   4000
2026-09-01T15:35:32+08:00     1440     1440     1240     1000   5120
2026-09-01T15:37:04+08:00     1600     1720     1560     1240   6120
```

There are five NLL rows per decision.  Across the first fixed window, shard 3
advanced by 40 decisions in 79 seconds, about 30.4 decisions/minute.  Across
the second, it advanced by 48 decisions in 92 seconds, about 31.3
decisions/minute.  Completion is determined by the slowest shard, not aggregate
throughput.  At approximately 30--31 decisions/minute, its 5,986 decisions
require about 3.2--3.3 hours plus model loading and the required no-op resume
audit.

The corrected pilot projects another approximately 0.44 hours for label-free
feature extraction and strict join.  Merge, model reload, resume, provenance,
and final audit overheads leave negligible or negative margin under the
original hard end.

## Amendment

Request `TimeLimit=09:30:00`, moving the hard end to approximately
`2026-09-01T20:40:31+08:00`.  The maximum additional reservation is 75 minutes
times four GPUs, or 300 GPU-minutes.  The authoritative quota immediately
before this decision reported 181,297 GPU-minutes remaining, so the added
reservation is approximately 0.17% of remaining quota.

This is strictly an operational continuation margin.  It does not change the
population, source shards, candidate actions, prompts, model or revision,
seeds, dtype, attention implementation, pixel bounds, NLL target, feature
definition, cost, OOF evaluator, bootstrap, advancement rule, or sealed-data
status.  Existing Slurm state-change email remains enabled.

If Slurm rejects the extension, do not cancel or restart the healthy job.
Continue until terminal state and use the separately frozen exact-prefix
recovery path only after an unsuccessful terminal record.
