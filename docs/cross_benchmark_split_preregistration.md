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
- After decoding, audit RGB image digests across partitions. If distinct public
  source IDs contain identical RGB content, exclude the colliding formal source
  and take the next source in the same hash ranking. Record every exclusion and
  repeat until state, source, and image overlap are all zero. This collision
  rule uses no question target or model outcome.
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

## Collision audit amendment

The first DocVQA formal export exposed one RGB-content collision before any
model rollout: development `docId=4386` and formal `docId=4331` share image
digest `c0facff643f69f9505a5b5e5d15c4b57d51c1d25796c114c4ae2d82c2fd8b98f`.
The failed formal export is retained as infrastructure evidence. DocVQA formal
v2 excludes source group `4331` after offset 200 and deterministically backfills
the next ranked source; it is acceptable only if the repeated audit reports
zero state, source, and image overlap.

## Frozen manifest products (2026-08-28)

The following products were frozen and audited before any rollout outcome was
generated on these sources:

| Benchmark/role | Sources | Questions | Manifest SHA-256 | Image-bundle SHA-256 |
| --- | ---: | ---: | --- | --- |
| DocVQA development | 200 | 824 | `873df25b9df1bcff1aa12ad99a352bc7d7cc89ade4a0db02caf1510a3163f862` | `770080820b2f638555bb302a0ed2c85e97e09775ced9b5212928267614ea9dd8` |
| DocVQA formal v2 | 400 | 1,608 | `9ceb28d05df5feecedf6cf61fbbb27ce281b94dd027e5d6d6da43ddc091081ac` | `4645bb9b1e796fac5388cf8818f0fb3a110b367655cb39659064dc7b44c18e02` |
| TextVQA development | 200 | 318 | `bfe1105df2b9f37ed352207a46d519c0a3468a677759ec8039dbbbdec1fd54fa` | `d17e652d8307e69eaa411ba1c450165dcf9f34ff7c7324ec7fc927397fac70b8` |
| TextVQA formal | 400 | 633 | `847899f91147633186b61a802004c49cfe8ef3258427cb92ea390c891ec5ef2c` | `b5265fc39b64931f3cbedc6f30462ab8e75fee5c611b13915955984f3d928e80` |

Both pair audits report `states=0`, `sources=0`, and `images=0`. Formal target
answers are present only to support a future frozen evaluation; they must not be
read for model development or threshold selection. Dataset revisions remain
`539088ef8a8ada01ac8e2e6d4e372586748a265e` for DocVQA and
`9c0699cd19768ac5ab97568f6b3cbac4c0062884` for TextVQA.
