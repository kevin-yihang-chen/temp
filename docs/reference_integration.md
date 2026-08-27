# Reference repository integration notes

Snapshot reviewed: 2026-08-27.

## UG framework

Source: <https://github.com/ExplainableML/ug-framework>

Pinned commit: `13050ee49865e4330519108f42d1ccfccff1aee1`.

The reference implementation is built on its bundled `lmms-eval` fork and
supports Qwen2.5-VL. Its visual-search interface exposes `ug_search`,
`num_visual_crops`, and `visual_crop_ratio`; listed search benchmarks include
V*Bench, HRBench, TextVQA, POPE, DocVQA, and GQA.

Implemented project-side adapter boundary:

1. Reuse or faithfully reimplement its crop proposal geometry in an external
   adapter.
2. Generate proposals from `AgentState`, which cannot access ground truth.
3. Capture every proposed bbox before candidate execution.
4. Save its post-crop response entropy as `entropy_after` only.
5. Run the same candidate set through the task scorer to populate
   `correct_after`.
6. Do not expose post-crop entropy to the learned pre-action model.

`UGGridProposer` now reproduces the reference geometry: a square crop with side
`min(width, height) / visual_crop_ratio`, a half-crop stride, and explicit final
edge positions. It can deterministically select a spatially balanced four- or
nine-candidate budget from the full grid. `Qwen25VLBackend` resizes each crop to
the original dimensions, keeps the original image as the first observation,
and computes the mean generated-token entropy normalized by `log(vocab_size)`.

The reference's KV-cache-sharing optimization is not copied; this adapter runs
each sibling independently so that correctness is easier to audit first. That
is a compute difference, not a scoring or observation-semantics difference.

## VTool-R1

Sources:

- <https://github.com/VTOOL-R1/vtool-r1>
- <https://github.com/VTool-R1/training-v2>

The main repository describes outcome-reward training on ChartQA/TableQA. It now
points to `training-v2`, based on asynchronous agent loops in `verl`, as the
recommended training path; its README explicitly labels the original path as old.

Planned adapter boundary after the Stage-1 value model succeeds:

1. Map `[ORIGINAL, ZOOM]` observation histories to its agent-loop messages.
2. Keep VTool-R1's final-answer reward for reasoning/final tokens.
3. At each emitted visual action, look up or predict sibling action VOI.
4. Normalize sibling VOI into a visual-action advantage.
5. Apply that advantage only to visual-action tokens in an ablation branch.
6. Compare against unmodified outcome-only training with matched compute.

No large-scale RL result is claimed in the current repository. Published resource
estimates in the reference README make the diagnostic and value-head stages the
appropriate first gate.

## Reproducibility rule

Record upstream commit hashes before any real run. The upstream repositories are
active, so branch names alone are not stable experimental identifiers.
