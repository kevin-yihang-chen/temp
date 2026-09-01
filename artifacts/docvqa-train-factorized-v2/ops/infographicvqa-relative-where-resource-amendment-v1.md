# InfographicVQA relative-where resource amendment

Status: frozen after the scientific protocol and before submission. This
amendment changes only Slurm resources and runtime checks. It does not change
the model family, folds, targets, optimizer, endpoints, or train gate.

At 2026-09-01 HKT the live account report showed 222,000 GPU-minutes with
41,598 used and 2,664,000 CPU-minutes with 194,372 used. H800 mixed-capacity
nodes were visible. The bound feature payload is 1.4 GiB, each full outer fit
uses 3,584-dimensional question/global/four-ROI tensors, and the primary
constructs full-batch candidate-relative activations. The previous 65-fit DECAR
run completed on one H800, while a 24 GiB RTX 4090 would add avoidable OOM and
retry risk.

Request:

```text
partition: q-h800
GRES:      gpu:h800:1
CPUs:      12
memory:    192 GiB
limit:     1 hour
```

The submitter must retain at least 60 GPU-minutes and 720 CPU-minutes at
admission. The worker requires exactly one NVIDIA H800, validates every code
and input hash before fitting, removes credentials/proxies, and refuses to
overwrite any fit/evaluation output. It runs the 20 OOF fits and only then the
frozen CPU-compatible evaluator on the same allocation.

Use Slurm `--mail-type=ALL` for `yihangc@connect.hku.hk`. No GitHub push is
authorized.
