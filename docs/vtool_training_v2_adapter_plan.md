# VTool-R1 training-v2 adapter plan

## Frozen upstream target

The adapter targets `VTool-R1/training-v2` main commit
`d2aa28353ec10c7f91b39f502925003a81d6982d`, inspected on 2026-08-28. The
repository reports verl commit `498c988ab7af49aa36c157c9214ebbc780013d61`
and VLLM 0.17. Do not implement against an unfrozen branch name.

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
