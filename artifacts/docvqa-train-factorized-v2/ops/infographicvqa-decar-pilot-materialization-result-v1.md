# InfographicVQA DECAR pilot-materialization result v1

Status: accepted and frozen on 2026-09-01 after the train-only method and
identity allocation were frozen. This is engineering input preparation, not a
task result or a policy-selection event.

## Bound implementation

- Code revision: `b41de0898d6026bc82fa0c8fd25fd4c4e7c4e5d3`.
- Materialization module SHA-256:
  `f4d998494bd8e8ffa6de7e3c2c31b999b9bd802dd3867e52f3af4daad7264660`.
- Runner SHA-256:
  `4281e85aaa42d84573ec49a40382f71dfc34523ec0a3bef4f9dfebd7f974c109`.
- Test SHA-256:
  `8add39a022ce636a721ada3f29959fa94b2ce2acbcfc54f9c3146e0ba92d3605`.
- Method protocol SHA-256:
  `d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342`.
- Allocation-result SHA-256:
  `3d0948cc6840b008cd4b19408ff002ed0756bb0d9f7f5e6b8cdb6d0af5a4da60`.
- Pilot-question identity SHA-256:
  `fd5de6036f685b4b03d560fc59959196b9e9b96aba2fce70a80353b5adfc2388`.

Targeted materializer/allocation tests passed (`4 passed`), focused mypy
passed, and the complete repository suite passed (`408 passed, 19 skipped`)
before materialization.

## Frozen outputs

- Task manifest SHA-256:
  `80067cc1446782f458665d8ddfa98745bda73b03b9eb96da3528f82f22158d29`.
- Image manifest SHA-256:
  `fefdb22d762249f85026dd5312c6a6d1ba00bc799fcc0002f0679e511a66ca7d`.
- Report SHA-256:
  `3d721310f274acb67d2331eae695dbfd8f9b355056d55804cfb22618e4df4314`.
- Completion SHA-256:
  `9b28285892d43290b898eefa9bca3abef79f40a248323c84c4bce0df5b52562a`.

The task manifest contains exactly 512 unique states from 512 unique source
components and 512 unique registered images. It carries 542 accepted answer
references. Exactly 512 encoded images were materialized, totaling 167,374,442
bytes. Independent recomputation found zero encoded-image hash mismatches.

Before opening selected fields, the runner reverified all 24 train parquet
hashes against the pinned download manifest. It read only `questionId`,
`question`, `answers`, `image`, and `data_split` from required train row groups.
It did not read `answer_type`, `image_url`, `operation/reasoning`, or `ocr`.
Every selected row retained `data_split=train`; no validation/test file or row
was present.

The output report correctly records question and answer access as true. No
Qwen inference, generated answer, entropy, ANLS, correctness, task gain, task
utility, teacher likelihood, or policy score was computed. The engineering
pilot may now exercise the exact frozen 7B actor, sibling rollout, answer-NLL,
and label-free pre-action feature contracts. Its endpoints remain forbidden
for model, method, population, or hardware selection. No GitHub push is
authorized.
