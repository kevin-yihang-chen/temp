# InfographicVQA download result and source-audit freeze v1

Status: frozen on 2026-09-01 after the train-only transport download and
parquet schema/footer inspection, but before reading any parquet row, question
text, answer, reasoning label, OCR text, image URL, or image byte payload.

## Download result and incidents

- Attempt `199978` stopped before network access because the qwen environment
  did not contain pytest. No partial directory was created.
- Attempt `199982` stopped before any shard transfer because the compute node
  inherited an unreachable login-node proxy at `127.0.0.1`. It created only an
  empty partial directory.
- Worker-only recovery unset the proxy variables using the already proven RICO
  download pattern. Job `199985` completed in `00:00:18`, exit `0:0`, zero
  restarts, on one H800 with all-state email enabled.
- Final transport manifest SHA-256:
  `ecc46c6a073ebd89fc114cba6fee5c711c8600e596b5a785bec981d98b168f13`.
- Exact revision:
  `539088ef8a8ada01ac8e2e6d4e372586748a265e`.
- Exact files/bytes: 24 train parquets / `1,981,251,656` bytes.
- Validation/test files downloaded: zero / zero.
- The partial directory is absent after atomic promotion.

The manifest contains a SHA-256 for every parquet. Its first/last file hashes
are `11bdf0477b3f6b4f8e6369225769d68b1ed31fb4be2cdffdca510a2797533cd0`
and `976b84e517bbb0ba66db37b25a1149dfdde72428c2e50b620a90f9267929952b`.

## Outcome-blind footer evidence

Reading only Arrow schemas and parquet footers establishes exactly 23,946 train
rows across the 24 files. The exact ordered field names are:

```text
questionId, question, answers, answer_type, image, image_url,
operation/reasoning, ocr, data_split
```

`image` is a struct with binary `bytes` and string `path`. Footer inspection
did not read a data row or print any field value.

## Source-grouping amendment

The initial activation proposed decoded-RGB hashes as sources. Because the
schema exposes a label-free original `image_url`, exact-image grouping alone
would permit same-site infographic templates to cross OOF folds. Before any
row is read, replace that rule with the stricter deterministic connected
components below.

The audit may scan only `questionId`, `image`, `image_url`, and `data_split`.
It must never scan `question`, `answers`, `answer_type`,
`operation/reasoning`, or `ocr`.

1. Normalize an image URL by parsing its hostname, lowercasing it, removing a
   terminal dot, and removing exactly one leading `www.`. Ports, paths,
   queries, fragments, schemes, and credentials are excluded. Missing or
   malformed hostnames are reported and create no hostname edge.
2. Decode every image without truncated-image recovery, convert it to RGB, and
   hash width, height, and contiguous RGB bytes. Also hash the encoded bytes.
3. Create one graph node per decoded-RGB hash. Union nodes when they have the
   same nonempty normalized hostname. Identical decoded images are already one
   node even if encoded bytes or hostnames differ.
4. Define a source component as a connected component. Its stable source ID is
   SHA-256 over namespace `beyond-entropy-infographicvqa-train-source-v1`, a
   NUL separator, and the sorted full decoded-RGB hashes joined with NUL.
5. All questions attached to any image in a component remain indivisible in
   source-held-out OOF folds. Exact/decoded duplicates and same-host template
   families therefore cannot cross folds.

This conservative grouping may reduce the effective source count; that is a
scientific property, not a reason to split a large component. Near-duplicate
images across different hostnames remain an acknowledged limitation rather
than an outcome-selected heuristic.

## Frozen audit outputs

The audit must verify all 24 file hashes and exact schema before scanning the
four allowed columns, then write:

- a report with row/source/host counts, questions per source, image resolution,
  encoded/decoded duplicate counts, missing/malformed URL counts, largest
  components, and explicit columns-read/columns-forbidden records;
- one outcome-free row manifest containing question identity, image identity,
  normalized hostname, encoded/decoded hashes, and source ID, but no text or
  answer;
- one deterministic 512-source engineering-pilot manifest selected by
  SHA-256 rank of `(namespace, source_id)` without using any endpoint.

The audit fails closed on a non-train split marker, duplicate/empty question
identity, corrupt image, missing image bytes, inconsistent repeated encoded
image, schema/hash mismatch, nonexact row coverage, or forbidden-column scan.
No train task label may be opened and no Qwen call may run until the audit and a
separate method protocol are frozen. Validation/test remain absent and sealed.
Every submitted task must use all-state email. No GitHub push is authorized.
