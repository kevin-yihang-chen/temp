# InfographicVQA fresh-development activation v1

Status: frozen on 2026-09-01 after public repository metadata inspection but
before downloading any InfographicVQA parquet, reading any question or answer,
or computing any Qwen outcome. This branch cannot revise any DocVQA, TextVQA,
or ScreenQA conclusion.

## Scientific rationale

The minimum-rank consensus failure closes candidate selection on the opened
DocVQA bank. InfographicVQA is a newly introduced project population with dense,
high-resolution, text-and-layout-heavy images for which selective crop
acquisition is technically relevant. The official dataset page reports roughly
5,000 images and 30,000 manually annotated questions, primarily extractive with
a smaller non-extractive subset:
https://site.docvqa.org/datasets/infographicvqa . The WACV 2022 paper is
https://arxiv.org/abs/2104.12756 . Neither source nor any InfographicVQA row has
previously participated in this project's candidate selection.

## Pinned transport mirror

The official challenge download requires registration. For reproducible
engineering, use only the public Hugging Face transport mirror below, while
preserving original-dataset attribution and disclosing that the mirror is not
the official challenge host.

- Repository: `lmms-lab-encoder/DocVQA`.
- Repository type: `dataset`.
- Exact revision:
  `539088ef8a8ada01ac8e2e6d4e372586748a265e`.
- Eligible download pattern: `InfographicVQA/train-*.parquet` only.
- Exact expected train file count: 24.
- Exact aggregate remote train size: `1,981,251,656` bytes.
- Validation metadata: four files, `266,457,648` bytes; do not download.
- Test metadata: four files, `297,182,775` bytes; do not download.

The download job must write a local file-size and SHA-256 manifest, verify the
exact 24-file set and aggregate size, and use all-state email notification.
The current filesystem had 61 GiB available before activation; no disk cleanup
is authorized or required.

## Frozen role semantics

1. **Official train: development only.** After a label-free integrity audit and
   a separately frozen method protocol, train questions/answers and complete
   sibling outcomes may be opened for bounded model development.
2. **Official validation: sealed calibration.** Do not download or inspect any
   validation row until the actor, scorer, feature schema, primary candidate,
   baseline set, cost, call-rate grid, and advancement rule are frozen from
   train alone.
3. **Official test: sealed one-shot formal.** Do not download or inspect any test
   row unless validation passes its frozen gate. Test may be evaluated once and
   may never select a revision.
4. Existing ScreenQA calibration, formal, reserve, untouched, validation, and
   test roles remain sealed and are not fallbacks for this branch.

## Outcome-blind train audit

Before reading question text or answer fields:

1. require the exact pinned revision, 24 train filenames, aggregate byte size,
   and downloaded SHA-256 manifest;
2. inspect only the parquet schema, split marker, question identity, encoded
   image bytes, image dimensions, and row-to-image membership;
3. require every image to decode, reject corrupt images, and report exact and
   decoded-RGB duplicate groups;
4. define `source_id` as the decoded-RGB SHA-256 so all questions sharing an
   image or byte-equivalent image remain in one OOF fold;
5. report row count, unique source count, questions per source, resolutions,
   and exact schema without printing question or answer contents; and
6. stop if train/validation/test markers mix, identities collide, images are
   absent, or an outcome field is required for grouping.

The audit report and deterministic source manifest must be frozen before any
task label is opened.

## Strong-backbone and compute rule

The new acting and scoring backbone is
`Qwen/Qwen2.5-VL-7B-Instruct`, using the already pinned local revision,
bfloat16, SDPA, no quantization, and no offload. A deterministic 512-source
train subset selected only by source hash may be used for runtime, memory,
checkpoint, and output-contract validation. Its task endpoints cannot select
hardware or population size.

For work projected above one hour, compare live queue plus runtime and quota for
four H800/H100 GPUs versus four RTX 4090 GPUs without reading endpoints. Prefer
four H800/H100 GPUs when the full 7B model loads uncompromised and the complete
reserve fits quota. Every shard must use one hardware class. Do not silently
downgrade the model, quantize, offload, shrink the source population, or mix 3B
outcomes with 7B likelihoods.

## Method-protocol boundary

No task outcome may be computed until a separate train-only method protocol
freezes a bounded primary family. Its center is a joint four-action/stop model
with explicit rescue, harm, and neutral targets, signed utility and
teacher-forced answer-loss auxiliary supervision at training time, and only
pre-action features at inference. It must include matched-call entropy,
random/fixed crop, charged exhaustive UG, task-value-only, loss-only, and
no-harm-head ablations. It must prohibit post-result hyperparameter expansion
and predeclare what happens if train OOF evidence is negative.

Every submitted compute task must email `yihangc@connect.hku.hk` for all state
changes. No GitHub push is authorized.
