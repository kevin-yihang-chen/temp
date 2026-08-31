# DocVQA reserve feature-resume recovery audit

Scientific state remained frozen at code revision
`491be10aa298a04fa337878be96b489e9818c114` and freeze SHA-256
`eebc6ee4c3b1affcfc97b1c953094722001838166c7c90c4bcccc88821c7315e`.

## Interrupted jobs

- Job `199826` completed and merged all 2,585 reserve rollouts, then failed
  before feature extraction because the frozen worker pre-created the semantic
  plan directory without passing `--resume` to the plan preparer.
- Job `199838` verified the reserve gate and all completed shard rollouts, then
  failed before feature extraction because the no-op rollout resume rewrote
  shard `resumed_from_records` metadata and the frozen merger correctly
  refused to replace its first-run provenance.
- Both jobs used Slurm `--mail-type=ALL`. Neither job created policy scores or
  the one-shot evaluation result.

## Preserved first-run rollout evidence

- merged rollouts SHA-256:
  `60ac67d8154a7941f271b4034fdd8681027cd8b702074556c009f11d6aee3925`
- merged provenance SHA-256:
  `0c3520b5d6ae9b5193da741f1974953073e6ab14bb6a83b33b9501a97f73ac35`
- merged diagnostic SHA-256:
  `17a385c3c6b4694dd7fede616849436707056b8eaaa22a100cefbf37da811ec9`
- complete bank: 2,585 states, 12,925 records, 688 sources, 688 images

The no-op resume left every shard rollout SHA unchanged. Before restoring the
four runtime counters to their truthful first-run value of zero, the rewritten
shard-provenance SHA-256 values were:

- shard 0: `2418de31c92dbf7c5a1f99fc6cede2f67fb1300c0f49d9d3b371af703d89f9df`
- shard 1: `918b75a0f01653eeb5fdf70519c1072e67bf7babe96d696f7d8341509e9d829f`
- shard 2: `c68d202b4b9f5487dcd13875fbc88bab04f8d713509633a12660d2f00a210488`
- shard 3: `2c04bc1cfa77cb4c1cee57e22b185d457b9e963c523c762ceeee3c30af45d835`

After restoring only `resumed_from_records`, the frozen merger accepted the
existing merged rollout, diagnostic, and provenance byte-for-byte and reported
`passed=true`.

## Outcome-blind feature plan and recovery worker

- feature plan SHA-256:
  `7b94e4619088ab7c6ccae53b15001f6778a4378917752cb115458034effae86b`
- assignment fields: `state_id`, `replicate_id`, `source_id`
- `assignment_outcome_fields_used=false`
- frozen original pipeline SHA-256:
  `835ae4c25c6f6aeb055d5cbce71050c4e6c3efaa4b7d7a9098ac862e4bb8fe41`
- feature-resume wrapper SHA-256:
  `98caeff8a6ec7f2f7b92cfd2f4b8af5d1fd288d55a27952b05fbdc9283b34ca4`

The recovery wrapper verifies all hashes above, sources the frozen pipeline
initialization through the pre-collector boundary, skips only the completed
collector and merge, and sources the exact frozen feature-to-evaluation suffix.
Its local validation-only execution passed before submission.

## Completed recovery and one-shot result

Recovery job `199840` ran the frozen feature-to-evaluation suffix on four
NVIDIA H800 GPUs and completed successfully. It used Slurm `--mail-type=ALL`.
The completed artifacts are bound as follows:

- label-free attention-semantic features SHA-256:
  `7c9f6eab329dfff22ed47ad4ba4a30d5e5c9f799cc0b7b41fedac749e9a7cd5a`;
- label-free feature audit SHA-256:
  `fc8ea57bec26c6a4a1088a5e2c15652e2e30ea61ce197590e084db194b74d4b2`;
- outcome-free policy scores SHA-256:
  `0cbd5ed1cbeef52316abcff422e849365ca944c3b206c0692756525947069f93`;
- score report SHA-256:
  `d68405c95dc75fbb62750ff8110abd74f05c3f62397dfbd4eb8145de20194e5e`;
- one-shot evaluation report SHA-256:
  `c3c5103db85246c85da3fd6740194726fb7871279256acd2942369e6c7812315`.

The primary source-balanced utility difference, Policy A minus frozen Policy
B, is `0.0003826055243206405` with paired whole-source 95% percentile interval
`[-0.0012353519784479077, 0.001968124913482835]`. The registered lower-endpoint
condition therefore fails and `supports_policy_a_over_policy_b=false`.

## Independent deterministic replay and score audit

After job completion, the evaluator was invoked independently from the login
node with `PYTHONPATH=src`, the frozen Qwen environment, every registered input
hash, 20,000 resamples, and seed `20260829`. Its temporary report was
byte-identical to the formal result and reproduced SHA-256
`c3c5103db85246c85da3fd6740194726fb7871279256acd2942369e6c7812315`.

An independent schema and threshold audit verified all 2,585 score rows:

- exactly nine allowed identifier, action, score, and call fields per row;
- 2,585 unique, deterministically sorted decision identities and 688 sources;
- finite Policy-A values and Policy-B probabilities in `[0, 1]`;
- exact Boolean types and only the four frozen UG-grid action identifiers;
- exact threshold reproduction: 42 Policy-A calls, 43 frozen Policy-B calls,
  and 50 secondary test-feature-matched Policy-B calls;
- `selection_uses_outcomes=false` and no serialized outcome fields.

Two preliminary replay attempts produced no report: the first named a
nonexistent project-local virtual environment and the second omitted
`PYTHONPATH=src`. They changed no tracked or frozen artifact and are disclosed
here to distinguish environment setup failures from scientific evaluation.
