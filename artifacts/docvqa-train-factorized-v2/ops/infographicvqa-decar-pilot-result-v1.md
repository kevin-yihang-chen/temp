# InfographicVQA DECAR four-H800 pilot result v1

Status: engineering pilot passed on 2026-09-01. This result exposes only
coverage, leakage audits, immutable hashes, hardware, and runtime. No task
endpoint or diagnostic value was opened or used for a scientific or hardware
decision.

## Slurm execution

- Job `200031` ran on four NVIDIA H800 GPUs in partition `q-h800`.
- Submitted and started at `2026-09-01T09:30:18+08:00`; queue wait was three
  seconds.
- The job completed at `2026-09-01T09:38:57+08:00` in `00:08:39`, with exit
  code `0:0` and zero restarts.
- The exact tracked revision was
  `166b1f008adf36351c2ca74f4ebcf018cd038ff9`.
- Slurm email was configured to `yihangc@connect.hku.hk` for all supported
  state changes.

## Contract results

- The source-disjoint population contained 512 questions, 512 sources, and
  five actions per question.
- All four rollout shards completed. Their first-pass outputs survived a
  complete resume byte-for-byte, covering all 2,560 action records.
- The canonical rollout merge contains 512 decisions and 2,560 actions.
- All four teacher-NLL shards completed and survived a complete resume. The
  canonical merge contains 512 decisions, 512 sources, and 2,560 records, and
  reports `raw_targets_written=false`.
- All four original-image feature shards completed. The canonical label-free
  merge contains 512 source-disjoint decisions, reports
  `outcomes_included=false`, and has no outcome fields.
- The actor/scorer was the pinned
  `Qwen/Qwen2.5-VL-7B-Instruct` revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, using bfloat16 and SDPA with the
  frozen pixel and prompt contract.
- No validation or test input was used. The execution record explicitly sets
  `task_endpoints_used_for_selection=false` and
  `validation_or_test_inputs_used=false`.

## Immutable outputs

```text
7d5e2500fdd7ca6580b2241871209aa979a02d30a3064a3cb20e68fcba5a9b1b  execution/job-200031.json
2061fbd49c4c297a9ffc6b50a763dd11a008625393e7bfff701ef4be1a78cc56  merged-rollouts/rollouts.jsonl
f2a3d829c02532d28ef543b8ca83017b640219ab479a2f8106619312b8bdca94  merged-rollouts/rollouts.merge.json
8f11bff364e49aaffde0c46e6f53e2f78ca60fb033047b4043705e2a315cd9c8  merged-nll/answer-nll.jsonl
18b4b0bb8430f4325161d3301ac7add9401a9c00c74e6ae3477a3f89e8c1e77d  merged-nll/answer-nll.provenance.json
1b150299c744a98ff1b59d3b4e60dafa1ef8cef4b365536461de1386131a33a3  merged-features/features-label-free.pt
33d16ed7943a78064505130a003f036ffd6daddac1e5080c6aefff5e8826e3c2  merged-features/merge-report.json
f548f64f61b97207c911c709ce7c8a44a6fd0c4d9cffa972ba817529f4d0b0c4  merged-features/label-free-audit.json
```

The artifact root is
`artifacts/infographicvqa-train-v1/decar-v1/pilot-qwen7b-v1`.

## Runtime projection and accelerator decision

The recorded four-H800 stage times were 264 seconds for rollout plus complete
resume, 217 seconds for teacher NLL plus complete resume, 34 seconds for
label-free features, and 516 seconds total. Scaling conservatively by
`23,946 / 512 = 46.76953125` projects:

| Stage | Projected wall time |
| --- | ---: |
| rollout and complete resume | 3 h 25 min 47 s |
| teacher NLL and complete resume | 2 h 49 min 9 s |
| label-free features | 26 min 30 s |
| total | 6 h 42 min 13 s |

The unbuffered four-H800 projection is about 1,609 GPU-minutes. A 20% runtime
reserve projects about 8 h 3 min and 1,931 GPU-minutes. At the terminal audit,
the live account had 2,307 of 42,000 GPU-minutes remaining; `q-h800` allowed
up to 18 hours and one eight-H800 node was idle. H100 capacity existed but had
no matched InfographicVQA timing measurement, while switching accelerator can
introduce numerical drift. RTX 4090 capacity was available but offers less
memory headroom and is not expected to improve the validated H800 runtime.

Therefore the registered full-bank implementation should use four H800s, one
accelerator class across all shards, and an 8 h 15 min wall-time request. The
submitter must recheck live quota and require at least 1,980 remaining GPU
minutes immediately before submission. This decision uses engineering runtime
and resource availability only, never pilot endpoints.

## Scientific interpretation

This is not evidence that DECAR improves ANLS or utility. It proves that the
frozen 7B rollout, answer-likelihood, and pre-action feature path is executable,
resumable, source-disjoint, and leakage-audited at the pilot scale. Scientific
success or failure remains bound to the registered 23,946-question nested-OOF
comparison and its source bootstrap advancement gates.
