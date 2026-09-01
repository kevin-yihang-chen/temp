# InfographicVQA hybrid Slurm resource amendment

Status: operational correction before any hybrid endpoint computation.
Validation and test remain sealed.

The first admission attempt requested `debug`, four CPUs, 64 GiB, 45 minutes,
and no GPU. Slurm rejected it before creating a job with `QOSMinGRES`; this
cluster's compute QOS requires a GPU GRES even for CPU-only statistics.

Request one RTX 4090 solely to satisfy admission, while keeping four CPUs, 64
GiB, and 45 minutes. The `debug` partition binds four CPUs per GPU and permits
24 GiB per CPU, so the resulting 16 GiB per CPU is valid. Export an empty
`CUDA_VISIBLE_DEVICES` inside the worker so the evaluator remains CPU-only and
cannot change numerical behavior based on the reserved GPU. The 45-minute
upper bound consumes at most 45 GPU-minutes and 180 CPU-minutes; current quota
is sufficient.

This amendment changes no input, entropy identity, OOF action, call budget,
metric, bootstrap draw, comparator, cost, support rule, or validation/test
boundary. Its SHA-256 is passed by the submitter and verified by the worker.
All supported state-change emails remain enabled. No GitHub push is
authorized.
