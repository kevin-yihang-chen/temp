# Conditional ChartQA train chart-layout treatment protocol

## Activation

This protocol is frozen before the 2,137-state image-disjoint chart-layout
confirmation outcome is inspected. It is activated only if the go/no-go rule in
`docs/chart_layout_followup_decision.md` passes. Otherwise the scripts remain
unused.

## Frozen target and treatment

The target is the same 4,500-state, 4,500-image balanced ChartQA train manifest
used by the UG stopping replication. Its SHA-256 is
`72db6feaa4bc042e98741a48dd55421c5246c1b48c84b1fd75740d1d072ca621`.

The treatment changes only the candidate proposer:

- baseline: four spatially balanced UG grid crops;
- treatment: four chart-layout crops at left-top, left-middle, center-middle,
  and right-middle.

Both use Qwen2.5-VL-3B-Instruct revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`, deterministic seed 0, the concise
answer-only prompt, additive original-plus-crop inputs, SDPA, the same pixel
budget, and unit cost per crop. Answer-now generations must match exactly.

The analysis and success conditions are frozen in
`docs/chart_layout_followup_decision.md` and implemented by
`scripts/analyze_gated_candidate_confirmation.py`. No stopping threshold,
candidate count, cost, prompt, or target-derived calibration may change.
