# InfographicVQA outcome-blind source-audit result v1

Status: accepted and frozen on 2026-09-01 before any InfographicVQA question
text, answer, answer type, operation/reasoning annotation, OCR annotation, task
outcome, validation row, or test row was read.

## Bound execution

- Slurm job: `200004`; `q-h800`; one NVIDIA H800, 12 CPUs, and 96 GiB RAM.
- Start/end: `2026-09-01T09:01:31+08:00` / `2026-09-01T09:04:09+08:00`.
- State: `COMPLETED`, exit `0:0`, zero restarts, 158 seconds wall time.
- Mail: `yihangc@connect.hku.hk` with all supported execution-state events.
- Tracked code revision: `9cd698c249a9a24c89620ffa9492fd1e99d153ae`.
- Pinned train download manifest SHA-256:
  `ecc46c6a073ebd89fc114cba6fee5c711c8600e596b5a785bec981d98b168f13`.
- Audit report SHA-256:
  `1c801a641c13747a1be2abbcc3c4a8b2d0a32e33599caa36d86208853c866547`.
- Full source manifest SHA-256:
  `fc577513dd8f9993f40d14454c7ec4ecf48897ff0d1660479fb5c49d3ae9512a`.
- Pilot-source manifest SHA-256:
  `75f20c141ccc273dcc36a4527ec7697826e3fea4b2bfc110754027ad9bb9ffe3`.
- Completion SHA-256:
  `7eb0b0faca65da84e00a39464c538462dbe186e819210d59f6855e5aabee1fe4`.

The first two download submissions, jobs `199978` and `199982`, failed before
data transfer because of a missing test dependency and an inherited loopback
proxy respectively. The corrected public unauthenticated download job
`199985` completed. These incidents changed no scientific input or selection
rule and are retained in the earlier download/audit freeze.

## Verified population

The exact 24 pinned train parquets contain 23,946 rows, 4,406 encoded images,
4,406 decoded-RGB images, 23,946 unique non-empty question IDs, and 2,204
normalized hostnames. All images decoded without truncated-image recovery.
Every split marker was `train`; validation and test files were absent.

The later, stricter source freeze intentionally refines the initial
RGB-identity grouping in the activation note. A source is a connected
component under either equal normalized hostname or equal decoded-RGB hash.
This prevents images from one publishing site or exact visual duplicates from
crossing OOF folds. The resulting 2,204 source components exactly cover all
rows. No cross-host decoded-RGB merge occurred in this transport snapshot, so
the component and normalized-host counts are equal.

The component distribution is long-tailed:

| Statistic | Questions/source | Images/source |
|---|---:|---:|
| Minimum | 1 | 1 |
| Median | 5 | 1 |
| 95th percentile | 23 | 4 |
| Maximum | 1,680 | 308 |

The three largest components contain 1,680, 971, and 620 questions. They must
remain indivisible. Ordinary row-random folds are forbidden; all subsequent
folding must be source-disjoint and balance question counts without consulting
task outcomes.

The deterministic pilot contains 512 unique source components, 4,315 questions,
and 798 images. Its largest component has 333 questions and 55 images. All
selection ranks `0..511` occur exactly once and every pilot source is present in
the full manifest.

## Independent output checks

The published source manifest has exactly 23,946 JSONL rows and exactly 2,204
unique source IDs. Its field inventory is limited to:

`decoded_rgb_sha256`, `encoded_sha256`, `height`, `image_path`,
`normalized_hostname`, `question_id`, `source_id`, `transport_file`,
`transport_row`, and `width`.

It contains no question text, answer, answer type, operation/reasoning, OCR,
correctness, reward, entropy, likelihood, or model output. The report records
`question_text_read=false`, `answers_read=false`, `task_outcomes_read=false`,
and `validation_or_test_rows_read=false`. Recomputed file hashes match the
completion record.

## Access decision

The train transport is accepted as a fresh development population. Train
question and answer fields may be opened only after the separate method
protocol is frozen. Official validation remains sealed calibration and official
test remains sealed one-shot formal evaluation. No ScreenQA protected role is
opened or repurposed. This result authorizes no GitHub push.
