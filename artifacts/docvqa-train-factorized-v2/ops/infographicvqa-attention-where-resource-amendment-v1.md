# InfographicVQA attention-where resource amendment v1

Date: 2026-09-02

This amendment was frozen after the scientific protocol and implementation
were committed, but before any attention feature was extracted or any
scientific outcome was scored.

The first `sbatch --test-only` request for four H800 GPUs on
`q-hgpu-small` was rejected with `QOSMaxGRESPerUser`. It created no job and
consumed no compute. Read-only association state still permits four concurrent
GPUs globally, but scheduling probes established that this partition's QOS
rejects three H800s and accepts two. The idle H800 node itself has eight free
devices; the constraint is account/QOS-specific.

The operational allocation is therefore amended from four H800 GPUs in one
four-shard wave to two H800 GPUs in two deterministic waves:

- wave 1: frozen source shards 0 and 1;
- wave 2: frozen source shards 2 and 3;
- one shard per GPU, identical model, revision, dtype, eager attention,
  checkpoint interval, input hash, output schema, and audit;
- `q-hgpu-small`, two H800 GPUs, 16 CPUs, 192 GiB, six-hour limit;
- all-state email remains enabled.

Shard extraction is decision-independent and no batch statistic crosses
shards. The canonical merger sorts by `(state_id, replicate_id)`, so wave
ordering cannot change any feature or scientific calculation. This amendment
changes only parallelism and wall-clock exposure. The protocol's population,
attention construction, baselines, bootstrap, advancement rule, protected
data policy, and no-GitHub-push rule remain unchanged.
