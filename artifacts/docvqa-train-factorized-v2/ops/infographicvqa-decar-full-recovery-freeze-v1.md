# InfographicVQA DECAR full-generation recovery freeze v1

Status: frozen on 2026-09-01 while original Slurm job `200130` was healthy and
running.  This is a contingency asset only.  It does not authorize a second
job while `200130` is active, alter the scientific experiment, or inspect any
scientific endpoint.

## Activation rule

The recovery submitter may run only after an authoritative Slurm record shows
that the referenced predecessor is in a terminal unsuccessful state:
`TIMEOUT`, `NODE_FAIL`, `PREEMPTED`, `FAILED`, `OUT_OF_MEMORY`, `BOOT_FAIL`,
`DEADLINE`, `REVOKED`, or `CANCELLED`.  It refuses active, pending, or completed
jobs and refuses any predecessor that already has a completed scientific
execution record.

No recovery is needed if job `200130` completes normally.  Preserve its output
and proceed to the separately frozen fit-runtime benchmark and OOF evaluation.

## Scientific identity

The scientific and output-provenance revision remains exactly:

```text
5b1b0211372ccb96ec21fc55fa954d427a5504b5
```

The recovery worker creates a detached temporary Git worktree at this revision
on the allocated compute node.  It links only the already frozen live manifest
directory and checkpoint/output root into that isolated worktree, then invokes
an ephemeral byte-audited copy of the original worker with `resume=1`.  The
only textual relocation changes its single hard-coded repository root to the
isolated worktree; the relocated worker hash is frozen as
`8e2bb53dc067e0f81ee3372c3f468e3b56ef76ccbea0a6e4ffc6eda3a642f388`.
Consequently Python modules, generation scripts, protocol documents, and the
original worker logic are loaded from the scientific revision, not from the
later launcher revision.  Resolved manifest and output paths still point to
the original live paths through the two links.

The later launcher revision is passed and recorded separately.  It is not used
as `BE_CODE_REVISION` and is not written as the scientific implementation
revision.

## Frozen execution assets

```text
ebee8cd95422681772a864367ba91cc5d50a3c22163416ffadb7c6779c978bc8  scripts/slurm_infographicvqa_decar_full_recovery_h800.sh
bd0feffa5f7ce821abde9ed6da1a1c7b89724950c00c27cdc86585cd86bdc18a  scripts/submit_infographicvqa_decar_full_recovery_h800.sh
4bb26a8977de2f1838b9cd2838cedbe6d11d6b3c9157df3c70deb17dc94acc86  scripts/slurm_infographicvqa_decar_full_h800.sh
f9ee68799c17c0f5864fac61ae6ea52268017623bb833e2e0e0faf1f4c3f9a0b  infographicvqa-decar-full-generation-freeze-v1.md
```

The submitter binds the current launcher revision and hashes of the wrapper,
this freeze, original worker, and original generation freeze into the Slurm
arguments.  The wrapper fails closed on any mismatch.  The isolated original
worker then repeats all original manifest, model, hardware, shard, resume,
byte-identity, leakage, and joined-input checks.

## Hardware and notification contract

- exactly four NVIDIA H800 GPUs;
- 32 CPUs and 384 GiB memory;
- at most 8 hours 15 minutes;
- the same conservative 1,980 GPU-minute admission reserve;
- offline model loading with credentials and network proxies removed by the
  original frozen worker; and
- Slurm email for every supported state change to
  `yihangc@connect.hku.hk`.

The recovery job writes the original scientific execution record unchanged and
adds a separate `job-<id>.recovery.json` sidecar containing the predecessor
state, predecessor exit code, scientific revision, launcher revision, original
and relocated worker hashes, all recovery hashes, queue/recovery time, and the
scientific execution hash.

## Forbidden changes

Recovery may not change the population, source shards, action bank, prompts,
model snapshot, seeds, dtype, attention implementation, pixel bounds, cost,
targets, features, OOF evaluator, bootstrap, advancement rule, or sealed-data
status.  It may not delete or rewrite a complete predecessor execution record.
It may not be described as an independent replication.

The only permitted effect is exact-prefix continuation of incomplete rollout,
teacher-NLL, or label-free feature checkpoints followed by the original
byte-identity and completeness audits.
