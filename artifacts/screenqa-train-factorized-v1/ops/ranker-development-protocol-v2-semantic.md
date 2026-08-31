# ScreenQA conditional semantic ranker-development protocol v2

Status: conditionally frozen on 2026-08-31 while recovery Job `197011` was
still fitting the registered `spatial-context-geometry` candidate and before
its report, model, or candidate-selection audit existed.  No ScreenQA
calibration, formal, reserve, untouched, or official validation/test outcome
was opened while writing this protocol.

## Activation gate

This protocol is dormant unless the completed v1 candidate audit proves all of
the following:

1. both registered low-capacity candidates were evaluated under v1;
2. neither candidate has a registered non-degenerate risk-accepted threshold;
3. `candidate_frozen` is false;
4. `semantic_escalation_required` is true; and
5. no calibration, formal, or reserve outcome was opened.

If either v1 candidate is eligible, use the v1 winner and never run this
semantic escalation.  The v1 candidate audit and its SHA-256 must be bound by
the semantic submitter before any feature inference starts.

## Frozen development input

- Ranker rollouts:
  `artifacts/screenqa-train-factorized-v1/ranker-rollouts-v1/merged/rollouts.jsonl`.
- Rollout SHA-256:
  `0437d2a499adccb1b4e19eb0160583789cee00edf244718ecae9e290108bb8c9`.
- Exactly 14,511 decisions, 72,555 action records, 1,510 source components,
  one ANSWER and four ZOOM siblings per decision.
- The ranker-bank manifest, merge, resume, provenance, and label-role audits
  from v1 remain mandatory.
- Cost remains `lambda_cost = 0.05` for one unit-cost crop.

Only these already-opened `ranker_training` outcomes may fit the semantic
candidate.  Semantic extraction may read identity, question, original image,
pre-action uncertainty, candidate boxes, and tool costs, but no correctness,
answer-after, success, entropy-after, or delta-success field.

## Sole semantic representation

The only eligible v2 representation is `hybrid-context-semantic` produced from
frozen `Qwen/Qwen2.5-VL-3B-Instruct` revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`:

1. Base full-image semantic features use bfloat16, SDPA, minimum 200,704
   pixels, maximum 602,112 pixels, `question_feature_mode=input_mean`, ROI mean
   pooling over the four frozen UG boxes, and `--exclude-outcomes`.
2. The question embedding is replaced once with the frozen
   `multimodal-original` question state using batch size 4, SDPA, and the same
   un-cropped original image.
3. Question-to-region attention uses eager attention, the mean of the last
   four layers, mean pooling over heads and exact question tokens, and ROI mean
   pooling normalized across the four candidate regions.  Candidate crops are
   never executed during feature extraction.
4. The factorized error head retains the normalized pre-action context vector.
   Rescue and harm heads use that context plus the existing compact frozen
   semantic action vector.  Raw 2,048-dimensional embeddings are not directly
   fit, and no neural head is trained.

No alternative question embedding, attention layer count, feature mode, model
size, crop proposal, or learned representation is eligible in v2.

## Frozen parallel extraction contract

Feature inference may use exactly four RTX 4090 workers.  Decisions are sorted
globally by `(state_id, replicate_id)`, grouped into the exact batch-size-four
units used by single-GPU multimodal inference, and whole batches are assigned
round-robin to the four workers.  A global batch may not be split across GPUs.
Assignment uses only `state_id`, `replicate_id`, and `source_id`; outcomes are
forbidden.  Source overlap across feature shards is allowed only under the
SHA-256-bound batch plan.

Each worker runs base, multimodal-question, and attention stages on its shard
with exact-contract checkpoint recovery.  Canonical merging must restore the
globally sorted decision order, validate each shard rollout and feature hash,
verify exact full decision/action coverage, rebind stage metadata to the full
rollout bank and preceding canonical feature file, and pass the label-free
semantic audit.

The generic batch preparer and canonical merger inspected when freezing this
protocol had these SHA-256 values:

- `prepare_semantic_feature_batch_shards.py`:
  `28f3e3b06007cb9a14e7cdef0ec7a631a67581cb2a6618dd4249aec2d1da22f1`;
- `merge_semantic_feature_shards.py`:
  `3b1051ea28b07a5aefd70c4c347c43410c1023cc35eed739216dc0d0d1d3ff30`.

Before production use, these exact bytes or an explicitly re-proven successor
must be hash-bound by the submitter.  The existing batch-aligned real-inference
proof on opened ranker data established bitwise equality for base,
multimodal-question, and attention decisions; it is operational evidence, not
a ScreenQA benchmark result.

## Sole model and hyperparameters

Fit exactly one candidate with:

- model family: `factorized-oof`;
- feature mode: `hybrid-context-semantic`;
- five source-held-out folds assigned by `source_id`;
- alpha grid: `[0.1, 1, 10, 100, 1000]`;
- seed: `20260831`;
- source-grouped bootstrap resamples: 2,000;
- sample weighting: equal domain, then equal source, then equal row;
- tail target call rates: `[0.005, 0.01, 0.015, 0.02, 0.03, 0.05]`.

No source component, question, or action sibling may occur in both an OOF
training fold and its held-out fold.

## Eligibility and deterministic decision

Apply the same development-only tail diagnostic as v1:

- family error: 0.05;
- induced-harm upper limit: 0.005;
- net-negative-call-mass upper limit: 0.02;
- minimum source call rate: 0.01;
- minimum source-balanced utility: 0.001;
- threshold testing: Bonferroni bounded-mean KL/LTT over only the registered
  strict-to-permissive tail grid.

If a non-degenerate risk-accepted threshold satisfies every criterion, freeze
the full-development semantic model, its exact feature SHA-256, OOF report,
input hashes, code revision, trigger audit, and candidate bundle.  Clear its
execution threshold; fresh calibration may select only from the frozen tail
grid.

If the semantic candidate is ineligible, stop ranker development.  Do not fit
another representation, relax a risk or utility bound, inspect calibration,
or select answer-now as a publishable candidate.

## Freeze boundary

The activation audit, batch plan, three canonical feature files and audits,
semantic OOF report/model, selection audit, code revision, protocol SHA-256,
and final candidate SHA256SUMS must all exist before any ScreenQA calibration
annotation object or outcome is deserialized.  Calibration may set only a
registered risk threshold.  Formal remains a one-shot evaluation after the
fixed calibration gate passes.
