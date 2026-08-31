# DocVQA proxy-to-outcome cross-domain activation v1

Status: activated on 2026-08-31 after the frozen protocol, implementation
contract, and one-decision real-model smoke, but before any additional DocVQA
ranker-development answer likelihood was computed.

## Completed smoke

- Slurm job `197940`: `COMPLETED`, exit `0:0`, zero restarts, runtime
  `00:00:29`, one RTX 4090.
- Code revision:
  `2b9c0c522b1eedc7831a45e9b54dba3d98e7c3e1`.
- Output: one complete decision and five sibling records.
- Output SHA-256:
  `9aa6e35023a5246b4433f12eb2ee1a3149ce62b066e2e2ec61171d6b2471bc9d`.
- Provenance SHA-256:
  `6e036a454f261a348c40dbcdb6342a1d3567559a600cf3d4036b4359254574e3`.
- The second invocation resumed from one complete decision and preserved the
  exact output bytes.
- Structural validation found one ANSWER and four ZOOM records, finite
  non-negative NLL, positive token counts, one configuration hash, no raw target
  field, and an RTX 4090/bfloat16/SDPA measurement contract.

The smoke is engineering evidence only. Its proxy values are not inspected or
used to change the frozen endpoints or replication decision.

## Full-run resource decision

Activate four RTX 4090 GPUs, 16 CPUs, 128 GiB memory, and a four-hour limit for
the `13,580`-decision bank. At activation, the live account reported 39,041 of
42,000 GPU-minutes used, leaving 2,959 GPU-minutes. The full request reserves at
most 960 GPU-minutes and therefore fits the live quota with more than 3x reserve
margin.

The prior matched ScreenQA engineering audit showed that H800 and RTX 4090
could change small answer-loss differences enough to fail preregistered
loss-gap sign/ranking stability gates. The DocVQA sibling pipeline used the same
Qwen revision and 4090 execution family. Keeping all four full shards on RTX
4090 therefore takes precedence over the modest H800 throughput advantage.

Every full shard must preserve the scorer and answer-likelihood hashes in the
frozen implementation contract, record an accelerator name containing `4090`,
and share the same non-shard measurement configuration. Exact-prefix resume is
allowed only on RTX 4090 with the identical configuration. Mixed hardware is
forbidden.

No DocVQA calibration/formal/reserve-comparator input and no ScreenQA protected
role is authorized by this activation.
