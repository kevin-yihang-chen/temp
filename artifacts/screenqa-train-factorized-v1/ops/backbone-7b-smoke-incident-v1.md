# Qwen2.5-VL-7B smoke incident v1

Status: resolved on 2026-09-01 without changing the frozen model, population,
pixels, actions, endpoints, or hardware-selection rule.

## Failed attempt

Slurm job `199110` ran on one NVIDIA H800 and failed after `00:01:42` with
exit code `1:0`, zero restarts, and all-state email enabled. It successfully:

- loaded the pinned Qwen2.5-VL-7B revision;
- generated all 32 smoke decisions and 160 sibling rollout records;
- reproduced the completed rollout bytes under exact-prefix resume.

The answer-likelihood stage then raised:

`ValueError: manifest and rollout state coverage differ`

The scorer correctly required exact coverage but had no way to declare that a
32-state rollout was the frozen prefix of the 512-state manifest.

## Outcome-blind correction

The scorer gained an explicit optional `manifest_limit`. When absent, its
full-manifest behavior and configuration object are unchanged. When present,
the scorer:

1. verifies the SHA-256 of the complete manifest;
2. validates every manifest row, then selects the declared prefix;
3. requires exact state coverage between that prefix and the rollout bank;
4. binds `manifest_limit` and `manifest_examples_before_sharding` into the
   score configuration and provenance.

The smoke verifier now requires both fields. The failed directory was retained
and the corrected attempt used a fresh `smoke-h800-v2` root. Focused tests and
the full repository test suite passed before resubmission. No task endpoint was
reported, compared, or used to make this correction.

## Successful rerun

Slurm job `199116` completed on one NVIDIA H800 in `00:02:07`, exit code
`0:0`, zero restarts. The verifier passed all of the following:

- exactly 32 unique sources/states and 160 rollout plus 160 NLL records;
- one ANSWER and four ZOOM siblings per state;
- pinned Qwen2.5-VL-7B model/revision, bfloat16, SDPA, and H800 runtime;
- exact-prefix rollout and NLL resume with unchanged bytes;
- finite nonnegative NLL, positive token counts, and no raw target field;
- no protected input and no task endpoint computed for hardware selection.

Smoke completion SHA-256:
`e944437165523b4dab5261822abbeb002872f068d0ecd70b7af023688ae64e11`.

First-pass timings were `67` seconds for rollout generation and `47` seconds
for answer likelihood, or `114` seconds total. Resume validation required
another `9` seconds. These endpoint-blind timings, not the 32-state task
outcomes, are the only smoke quantities eligible for hardware activation.
