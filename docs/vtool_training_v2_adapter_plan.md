# VTool-R1 training-v2 adapter plan

## Frozen upstream target

The adapter targets `VTool-R1/training-v2` main commit
`d2aa28353ec10c7f91b39f502925003a81d6982d`, inspected on 2026-08-28. The
repository reports verl commit `498c988ab7af49aa36c157c9214ebbc780013d61`
and VLLM 0.17. Do not implement against an unfrozen branch name.

The public `VTOOL/Refocus_Chart` dataset target is revision
`00f10ecc5b25d94fd66e14c3671af9fb0f088989` (15,170 total rows; train and test
parquet files). Its row IDs are not the state IDs used by the confirmation
manifest.

The relevant implementation is `recipe/vtool/vtool.py`. Its one-tool agent
loop generates a first assistant segment, parses it as refocus code, executes
one image-editing round, appends an observation, and generates the final answer.
When a tool is attempted, every first-turn assistant token is set to
`response_mask=0`; observation tokens are also masked. Only the final assistant
segment remains trainable and the reward manager scores that final response.

This means the current upstream loop has no trainable visual-action-token
surface. A binary stopping advantage cannot be added only in the reward
function: the corresponding first-turn tokens are already excluded by the
response mask.

## Stage A: controlled when-to-call evaluation

Precompute the frozen gate with the base Qwen revision used by the confirmation
and store its decision under
`extra_info.tools_kwargs.metadata.beyond_entropy_gate`. The project-side
`VToolGateControl` schema records action, score, threshold, registered cost, and
model hash while enforcing `spatial_action_id=None`.

`scripts/export_vtool_gate_manifest.py` produces a label-free JSONL join table
and a provenance JSON containing byte-level rollout, model, and output hashes.
The join key is `(state_id, replicate_id)`; the exported row contains no answer,
correctness, post-action entropy, or candidate outcome.

The first export over the 4,500-state confirmation target is complete at
`artifacts/gate3-vtool-chartqa-train-4500/frozen-when-to-call-v1/gate_manifest.jsonl`.
It contains 4,500 rows and 288 calls (6.4%). Its SHA-256 is
`e76ca67cd98edd015c4180e81a17fb91808e828984d7b335b2873fec3ed5e0b6`;
the adjacent provenance file binds it to the confirmed rollout and model hashes.
This is an integration artifact, not a new evaluation result.

The exported confirmation state IDs cannot be joined directly to the public
VTool parquet. Before reuse, prove identity using the decoded-RGB SHA-256 plus
the normalized question hash produced by `vtool_identity_join_key`. The
normalizer only folds Unicode/whitespace/case and removes the manifest's trailing
`Answer:` instruction. Require one-to-one keys and report duplicates/unmatched
rows; index-based, fuzzy-text, or filename-only joins are forbidden. If coverage
is incomplete, run the frozen base-model gate directly on the unmatched VTool
rows instead of guessing a mapping.

The pinned public VTool test parquet was audited with
`scripts/audit_vtool_identity_join.py`. All 826 rows have unique valid identity
keys, and all 826 match the 2,500-state ChartQA development manifest; none match
the validation or train-replication manifests. The audit report SHA-256 is
`015a2bc18a9175bc121370fdfa082de314c2dcd339aca84624ce6cb97b5e803a`.
Consequently, VTool test may be used only for adapter parity/debugging: it is
not an untouched Gate 3 evaluation set and cannot support a paper claim.

Route each sample as follows:

- `ANSWER`: run one no-tool final-answer turn;
- `CALL_VISUAL_TOOL`: append exactly one controlled visual observation and then
  run the final-answer turn.

Use a fixed or seed-frozen random crop for the controlled call branch. Do not
let a learned refocus program choose the region in this experiment, because
where-to-look has not passed confirmation. Include the base-model gate pass in
token, latency, and GPU accounting. Compare answer accuracy, call rate, and
utility at `lambda=0.05` against answer-only, unconditional one-crop, and the
unmodified VTool loop.

## Stage B: binary call-token credit ablation

Only after Stage A parity succeeds, add a separate `call_token_mask` and
`call_advantage` to the pinned agent-loop output. Keep the existing
`response_mask` behavior for final-answer optimization. The new advantage may
apply only to the binary call/stop decision; observation tokens, Python tool
arguments, bbox values, and other spatial-action tokens remain excluded.

Run matched-compute branches:

1. upstream outcome-only reward;
2. upstream reward plus the frozen binary stopping advantage; and
3. a budget-matched random binary advantage negative control.

Freeze optimizer steps, rollout count, model/data revisions, judge, sampling,
and total permitted tool calls before training. Report both task reward and
registered accuracy-minus-cost utility. This stage tests credit assignment; it
does not test spatial proposal quality.

## No-go boundary

Do not train or claim localized visual-action advantages until a spatial
selector beats matched random/fixed candidates on a new untouched target. The
existing validation action contrasts and chart-layout confirmation do not meet
that condition.

Do not report VTool Refocus_Chart test as independent generalization because it
is a strict subset of the ChartQA development identities already used to build
the stopping gate. Pre-register a new benchmark/dataset split with zero RGB and
question identity overlap before any Gate 3 outcome is inspected.
