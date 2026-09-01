# InfographicVQA DECAR full runtime and resume risk audit

Snapshot: 2026-09-01T11:43:53+08:00, while Slurm job `200130` was
authoritatively `RUNNING` on four NVIDIA H800 GPUs with zero restarts and
`ExitCode=0:0`.  This audit used only Slurm state, checkpoint row counts,
runtime metadata, frozen file hashes, and pilot timing.  It did not read any
task score, teacher likelihood, policy prediction, validation input, or test
input.

## Live throughput and wall-time risk

At `00:31:56` runtime, the four rollout shards contained respectively
`4,080`, `4,480`, `4,320`, and `4,080` five-action rows.  These correspond to
`816`, `896`, `864`, and `816` complete questions.  Against the frozen shard
populations `6,014`, `6,036`, `5,910`, and `5,986`, the observed question
rates were approximately `25.6`, `28.1`, `27.1`, and `25.6` per minute.

Holding the slowest observed rate fixed projects completion of the rollout
first pass after about `3.9` hours from job start.  Scaling the corrected
pilot's 224-second teacher-NLL stage and 34-second feature/join stage by the
full population adds approximately `2.9` hours and `0.44` hours.  The resulting
point projection is about `7.3` hours.  Relative to the `8:15` Slurm limit,
the nominal margin is about 57 minutes; merge, resume-audit, and rate-drift
overheads make the usable margin smaller.

Operational thresholds are frozen for monitoring only:

- green: slowest rollout shard remains at or above 25 questions/minute;
- yellow: slowest shard falls below 23 questions/minute or the rollout stage
  remains incomplete after 4 hours 25 minutes;
- red: projected total exceeds 8 hours or Slurm reports a terminal nonzero
  state.

These thresholds cannot change any model, action, target, feature, split,
policy, or scientific decision.  Do not cancel a healthy job merely because a
point projection crosses yellow: all shard outputs are atomic checkpoints and
continued observation is cheaper than speculative restart.

## Revision and content audit

The running job is bound to code revision
`5b1b0211372ccb96ec21fc55fa954d427a5504b5`.  The login worktree is now at
`70a3decc8ef1025fddc64f287efef9dfb031e92c` because OOF evaluation, literature
positioning, and supplemental audit files were added after job start.  The
diff between those revisions adds or changes only OOF evaluator/launcher,
test, and operations-document files.  It does not change a full-generation
dependency.

Independent SHA-256 verification at this snapshot reproduced every frozen
generation implementation hash, including `sharding.py`, `cli.py`,
`rollout_shards.py`, `answer_likelihood.py`, `proxy_outcome_audit.py`, the
rollout merger, NLL scorer, full worker, full submitter, and the corrected
`qwen_backend.py`.  The running process therefore remains content-identical to
its generation freeze even though repository HEAD has advanced.

## Failure or timeout continuation rule

No second generation job may be submitted while `200130` is running.  If it
ends successfully, verify the final execution record and proceed normally.

If Slurm instead records `TIMEOUT`, `NODE_FAIL`, `PREEMPTED`, or another
nonzero terminal state, preserve every checkpoint and first establish the
terminal state, last complete five-row prefixes, and absence of a successful
execution record.  The stock command
`scripts/submit_infographicvqa_decar_full_h800.sh --resume` must **not** be used
unchanged from the advanced HEAD: it would bind the continuation to the new
revision while existing provenance is correctly bound to `5b1b021...`.

Before a continuation, freeze a resume-only launcher that:

1. preserves `5b1b0211372ccb96ec21fc55fa954d427a5504b5` as the scientific code
   revision and output-provenance revision;
2. verifies every generation dependency against the hashes in
   `infographicvqa-decar-full-generation-freeze-v1.md` and the corrected
   `qwen_backend.py` hash before loading any checkpoint;
3. records the later launcher revision separately without treating it as a
   scientific implementation change;
4. uses the existing exact-prefix `--resume` paths on the same four H800
   source shards and reruns every byte-identity audit; and
5. retains Slurm email notification for all state changes.

This continuation is an execution recovery, not a new experiment.  It may not
alter the population, action bank, prompts, model snapshot, seeds, dtype,
attention implementation, thresholds, or downstream advancement rule.
