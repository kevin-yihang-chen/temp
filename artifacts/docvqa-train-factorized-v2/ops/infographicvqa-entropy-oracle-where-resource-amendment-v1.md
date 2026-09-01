# InfographicVQA entropy oracle-where resource amendment

Status: frozen after the scientific factorization protocol and before
submission. This amendment changes only Slurm admission resources; it does not
change data, state identities, actions, cost, metrics, bootstrap, support
rules, or the validation/test seal.

The cluster's debug QOS rejects CPU-only jobs with `QOSMinGRES`. The immediately
preceding evaluator with the same 23,946-decision population and exact
`[20000, 2204]` bootstrap completed in 783 seconds on four CPU cores. Therefore
the factorization diagnostic will request:

```text
partition: debug
GRES:      gpu:rtx_4090:1
CPUs:      4
memory:    64 GiB
limit:     45 minutes
```

The evaluator remains CPU-only. The reserved GPU is hidden by exporting an
empty `CUDA_VISIBLE_DEVICES`; its sole role is satisfying the cluster admission
contract. A submitted run may consume at most 45 GPU-minutes and 180
CPU-minutes. The submitter must verify those reserves against the live account
quota before admission.

The worker must use `--mail-type=ALL` and notify
`yihangc@connect.hku.hk` on every Slurm state transition. Credentials and proxy
variables are removed before evaluation. No GitHub push is authorized.
