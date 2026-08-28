# Multi-domain action-value development protocol

Status: development-only protocol fixed before DocVQA or TextVQA sibling
rollout outcomes are generated.

## Purpose and isolation

The failed ChartQAPro formal split is excluded from every operation below.
Allowed labeled development inputs are ChartQA source data, ChartQAPro pilot,
and the source-disjoint DocVQA/TextVQA development manifests registered in
`cross_benchmark_split_preregistration.md`. DocVQA/TextVQA formal and HRBench
remain outcome-unseen until a complete replacement model and confirmation rule
are frozen.

## Frozen rollout protocol

Both new development banks use:

- model `Qwen/Qwen2.5-VL-3B-Instruct` at revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`;
- released UG four-candidate grid geometry at pinned UG revision
  `13050ee49865e4330519108f42d1ccfccff1aee1`;
- one deterministic generation replicate at seed 0;
- additive original-image plus crop observations;
- one unit of visual cost per selected crop and `lambda=0.05`;
- `max_new_tokens=32`, `min_pixels=200704`, `max_pixels=602112`, SDPA,
  bfloat16, and system prompt `You are a helpful assistant.`;
- pinned official-compatible DocVQA ANLS or TextVQA soft-accuracy scoring; and
- a second frozen-Qwen original-image pass that extracts global and four ROI
  representations without consuming action outcomes as features.

The frozen development inputs are:

| Domain | States | Sources | Manifest SHA-256 |
| --- | ---: | ---: | --- |
| DocVQA development | 824 | 200 | `873df25b9df1bcff1aa12ad99a352bc7d7cc89ade4a0db02caf1510a3163f862` |
| TextVQA development | 318 | 200 | `bfe1105df2b9f37ed352207a46d519c0a3468a677759ec8039dbbbdec1fd54fa` |

Every state produces one answer-now record and four concrete crop siblings.
Checkpoint resume is permitted only when the manifest hash, model revision,
prompt hash, pixel bounds, candidate count, and generation settings match.

## Development selection boundary

Development may compare low-capacity direct-gain, factorized risk/rescue/harm,
and frozen semantic-ROI action-value heads. Every candidate must:

1. use only question, answer-now confidence/output surface, original-image
   features, crop geometry, and cost at decision time;
2. split by source group within each domain;
3. weight domains equally during fitting and model selection;
4. include answer-now as an explicit zero-cost option;
5. report learned-top-crop versus random, fixed, and oracle crop diagnostics
   separately from the stopping result; and
6. select on the worst development-domain utility before domain-balanced mean
   utility, with no-call as a safe candidate.

After this development cycle, the exact feature mode, estimator, regularizer,
calibration margin, cost, crop set, and pass criterion are frozen in a new
formal protocol. No formal outcome may be used to choose among them.
