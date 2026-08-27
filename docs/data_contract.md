# Counterfactual sibling data contract

One JSONL row represents one action outcome from a shared agent state. Rows with
the same `(state_id, replicate_id)` form a paired sibling decision and must
contain exactly one `ANSWER` row and at least one `ZOOM` row.

## Three information domains

The runtime API separates information by construction:

```text
AgentState + ActionSpec  -> proposal/value model may read
GroundTruth             -> scorer may read
ActionOutcome           -> labels and diagnostics only
```

`proposals` receives `AgentState`, never `TaskExample` or `GroundTruth`. A feature
name blacklist remains as defense in depth, but it is no longer the primary
ground-truth isolation mechanism.

## Flat JSONL representation

| Field | Meaning | Gain-model input? |
|---|---|---:|
| `state_id` | Question/trajectory state identifier | grouping only |
| `image_id` | Stable image grouping identifier | split only |
| `source_id` | Stable source-example grouping identifier | split only |
| `replicate_id` | Paired stochastic replicate | grouping only |
| `generation_seed` | Seed shared by baseline/action siblings | no |
| `question` | User/task query | semantic adapter |
| `original_image` | Image path or stable URI | semantic adapter |
| `action_id` | Unique action within a decision | no |
| `action_type` | `ANSWER` or `ZOOM` | policy |
| `candidate_bbox` | Normalized xyxy box; null for `ANSWER` | yes |
| `entropy_before` | Baseline predictive entropy | yes |
| `entropy_after` | Entropy after executing this action | **no** |
| `answer_before`, `answer_after` | Paired answers | **no** |
| `correct_before`, `correct_after` | Success scores in `[0,1]` | **no** |
| `tool_cost` | Relative visual-action cost | policy only |
| `pre_action_features` | Numeric signals computed before action execution | yes |
| `metadata` | Provenance/debug information | no |

The learned target is cost-independent:

```text
gain_target = correct_after - correct_before
```

The policy applies deployment preference later:

```text
predicted_utility = predicted_gain - lambda_cost * tool_cost
```

This permits a single trained model to sweep multiple cost preferences.

## Additive ZOOM semantics

The backend receives an observation history. Baseline inference receives:

```text
[ORIGINAL]
```

A zoom sibling receives:

```text
[ORIGINAL, ZOOM(bbox)]
```

The crop supplements rather than replaces the original image. `AgentState` also
contains the pre-action trajectory so an adapter can map the request to a
message-style agent loop.

## Real rollout collection

Implement `VisualBackend.infer` or `BatchVisualBackend.infer_batch`, a proposal
callback, and a benchmark correctness scorer. `CachedVisualBackend` supplies a
safe in-memory cache keyed by state, observations, and generation seed.

Use the same candidate set and paired seeds for all siblings. Record model
revision, dataset revision, prompt hash, decoding parameters, and software commit
in `metadata`. Split formal experiments by `image_id` or `source_id`, never by
action row.
