# InfographicVQA DECAR corrected engineering-pilot result v2

Status: passed on 2026-09-01. This result validates implementation and
throughput only. No pilot task/NLL/policy endpoint was inspected or used for
selection, and official validation/test remain sealed.

## Execution

- Slurm job `200074`, four NVIDIA H800 GPUs, tracked revision
  `714c68d6175894a61e390199047a9578dc2aff6b`.
- Queue wait: 2 seconds; runtime: 528 seconds; restarts: 0; exit code: 0.
- Rollout plus byte-identical resume: 267 seconds.
- Teacher NLL plus byte-identical resume: 224 seconds.
- Label-free semantic features, merge, and strict DECAR join: 34 seconds.
- Population: 512 questions, 512 source components, 512 images, one ANSWER
  and four registered UG crops per question.
- All-state email was configured for `yihangc@connect.hku.hk`.

## Corrected input contract

The strict DECAR join passed for every decision:

- contextual question embedding: 3,584 dimensions;
- full-image embedding: 3,584 dimensions;
- ROI tensor: `4 x 3584`;
- scalar/geometry vector: all 16 registered fields;
- generated-token count, mean/max normalized entropy, and aligned mean
  generated-token log probability: complete and finite;
- label-free semantic storage: outcome-free; and
- scientific endpoints reported by the audit: false.

Applying the same join to pilot v1 fails closed at the missing generated-token
statistics, so v1 cannot silently enter a DECAR fit.

## Immutable outputs

```text
fea12e9c5f327a2d586ec0a3e08be1f9b4235a0a1e29c2690b217936d0c9d577  merged rollouts.jsonl
8121700c95f384c48b2c151130acad7181f28ae1a175c9d4a2f75792539853a4  merged answer-nll.jsonl
3ca280e53579d93020a7d23d631dfa04e8c18b1a673902ecf3fbc28433047a22  merged features-label-free.pt
f4e556f5c347bad5218b7636449c65caa5a4ba75c4202d0377959b44f052c288  decar-input-audit.json
5b35f92d16bbf59a84b01c1ccff78528ec520ce1d10a2a14d29fce7b771ca853  execution job-200074.json
```

Independent login-node hashing reproduced the rollout, NLL, feature, and
strict-audit hashes stored by the worker.

## Full-run projection and decision

Scaling 528 seconds by `23946/512` projects the full four-H800 bank at about
24,699 seconds (6 hours 52 minutes) and 1,647 GPU-minutes. A 20% reserve is
about 8 hours 14 minutes and 1,976 GPU-minutes. At the post-pilot live quota
snapshot, 2,264 GPU-minutes remained.

The corrected pilot therefore authorizes the registered full official-train
rollout/NLL/feature generation on four H800s, with an 8-hour-15-minute limit
and a 1,980-GPU-minute admission reserve. It does not establish scientific
success and does not authorize opening validation/test or pushing GitHub.
