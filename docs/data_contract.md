# Counterfactual sibling data contract

One JSONL row represents one action rolled out from a shared state. Rows with the
same `state_id` form a sibling group and must contain exactly one `ANSWER` row and
at least one `ZOOM` row.

Required fields:

| Field | Meaning | Model input? |
|---|---|---:|
| `state_id` | Shared pre-action state identifier | grouping only |
| `question` | User/task query | adapter-defined |
| `original_image` | Image path or stable URI | adapter-defined |
| `action_id` | Unique action within a state | no |
| `action_type` | `ANSWER` or `ZOOM` | yes |
| `candidate_bbox` | Normalized xyxy box; null for `ANSWER` | yes |
| `entropy_before` | Baseline predictive entropy | yes |
| `entropy_after` | Entropy after executing this action | **no** |
| `answer_before`, `answer_after` | Baseline and counterfactual answers | **no** |
| `correct_before`, `correct_after` | Success scores in `[0,1]` | **no** |
| `tool_cost` | Relative visual-action cost | yes |
| `pre_action_features` | Numeric signals computed before action execution | yes |
| `metadata` | Provenance/debug information | no |

The target is:

```text
(correct_after - correct_before) - lambda_cost * tool_cost
```

`FeatureEncoder` rejects common post-action/label names inside
`pre_action_features`. This is a guardrail, not a substitute for auditing the
actual feature computation graph.

## Real rollout collection

Implement `VisualBackend.infer`, a candidate proposal callback, and a benchmark
correctness scorer, then call `collect_sibling_rollouts`. Use the same decoding
settings and paired seeds for all siblings. Record model revision, dataset
revision, prompt hash, decoding parameters, and software commit in `metadata`.
