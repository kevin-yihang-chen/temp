# ScreenQA full proxy-NLL hardware activation v1

Status: frozen on 2026-08-31 after the matched engineering hardware audit and
before any full-bank answer-NLL shard was submitted or computed.

## Completed matched benchmarks

- H800 job `197351`: `COMPLETED`, exit `0:0`, 64 decisions / 320 records,
  85 measured seconds, score SHA-256
  `362df54951811ffd24dbc9c25ff16f38fb4da38c0de92078220db7c8872102d7`.
- RTX 4090 job `197352`: `COMPLETED`, exit `0:0`, 64 decisions / 320 records,
  100 measured seconds, score SHA-256
  `4c9a5e93c45410792bab3bc3f06d3dea94bc2918f602bc33bce31295f9bdff15`.
- Both used scoring code revision
  `e4de14c00bcc162edd9701b7e9c1fddbfd09a2a1` and the same non-hardware
  measurement configuration.

## Frozen consistency result

- Audit report SHA-256:
  `bb1ba6d1e066086bcaebd1713f6ccee796656892087986bb3a8adae6ffc371a8`.
- Completion record SHA-256:
  `ea26bf4898f41baa30e90886f372d594a1b3bdc84770d62647c42b1b1ff6e981`.
- Loss-gap Pearson: `0.9907346323922269`.
- Loss-gap Spearman: `0.9267530661715111`.
- Positive/nonpositive sign agreement: `0.8984375`.
- Top-one crop agreement: `0.734375`.
- Median / p95 / maximum absolute loss-gap difference:
  `0.006744901649653884 / 0.10904854536056519 / 0.39057445526123047`.
- Every preregistered H800 stability gate failed.

## Activated full-run hardware

The frozen decision rule selects **4 x RTX 4090**.  The 4090 benchmark projects
`5668.359375` seconds of four-GPU wall time and `377.890625` GPU-minutes for the
14,511-decision bank.  At activation, the live account had 3,296 GPU-minutes
remaining.  The full job therefore requests four hours, providing more than
2.5x wall-time margin while reserving at most 960 GPU-minutes.

Every full shard must:

- run on an accelerator whose recorded name contains `4090`;
- use scorer SHA-256
  `d278b8cd50a58133d6f512467dce8b53a38a690ade3e874b9721c61adabe523d`;
- use answer-likelihood module SHA-256
  `10c2b647b6ebbc036d6ce06b046521476b4f3d26e73e66b63b7d3f32382b51e4`;
- preserve the frozen model, prompt, bfloat16, SDPA, pixel, target, manifest,
  rollout, sharding, and raw-target contracts;
- share one identical measurement configuration across all four shards.

The orchestration commit may differ from the benchmark commit only because it
binds this completed hardware decision and changes the Slurm resource request;
the scorer and answer-likelihood component hashes above may not change.

ScreenQA calibration, formal, reserve, validation, test, and untouched roles
remain sealed.  This activation authorizes only the retrospective opened-bank
proxy audit defined by the frozen protocol and implementation contract.
