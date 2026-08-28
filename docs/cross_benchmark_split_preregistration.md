# Cross-benchmark split pre-registration

Status: frozen before downloading or scoring any DocVQA, TextVQA, or HRBench
rollout outcomes.

## Selection rule

- Namespace: `beyond-entropy-cross-benchmark-v1`
- Seed: `20260828`
- Rank each public source ID by
  `SHA256(namespace + NUL + seed + NUL + source_id)`.
- Retain every question belonging to a selected source.
- Development uses source-group offset 0. Formal confirmation starts at offset
  200, so no image/document source can occur in both partitions.
- The manifest sidecar must record the selected source IDs, source indices,
  dataset revision, split, prompt/scorer protocol, and manifest SHA-256.

## Frozen roles and budgets

| Benchmark | Source key | Development | Formal confirmation |
| --- | --- | ---: | ---: |
| DocVQA validation | `docId` | 200 source groups | next 400 source groups |
| TextVQA validation | `image_id` | 200 source groups | next 400 source groups |
| HRBench | `index` | none | all 800 paired 4K/8K indices |

The group counts are not question counts: all questions from a selected source
are retained, and final question totals are recorded only after manifest freeze.
HRBench 4K and 8K are paired resolution conditions over the same 800 indices;
they are not independent datasets. Neither resolution may be used for method
selection if the other is reported as untouched confirmation.

## Outcome-dependent branch discipline

- The current ChartQAPro formal result cannot alter these splits.
- If the existing frozen gate confirms, these tasks measure broader transfer;
  no threshold or feature is changed from their formal outcomes.
- If it fails, only ChartQAPro pilot plus the DocVQA/TextVQA development sources
  may be used to build a multi-domain action-value model.
- Model class, features, action candidates, cost, threshold selection, and all
  ablations are frozen before any DocVQA/TextVQA formal or HRBench rollout
  outcome is scored.
- A failure on a formal partition is retained as a negative result; it is never
  converted into a development set for a second claim on the same partition.

## Planned export commands

Development uses `--source-group-count 200 --source-group-offset 0`; formal
DocVQA/TextVQA uses `--source-group-count 400 --source-group-offset 200`. Both
use `--selection-namespace beyond-entropy-cross-benchmark-v1 --seed 20260828`.
HRBench exports all 800 rows in each resolution split and binds them by the
shared `hrbench:<index>` source ID.
