# Scaled TextVQA source-bank contingency

Status: outcome-independent capacity plan written before the fresh TextVQA
formal result is opened. No train-split manifest, rollout, feature, or model is
created by this note.

## Why another small validation split is not viable

At pinned TextVQA revision
`9c0699cd19768ac5ab97568f6b3cbac4c0062884`, the validation split contains
5,000 questions from 3,166 unique `image_id` source groups. The project has
already assigned hash-rank offsets 0--199 to development, 200--599 to the
first formal bank, and 600--2599 to the fresh formal bank. Only 566 unused
validation sources remain. Repeatedly testing small revisions on those sources
would have low power and an indefensible multiplicity story.

The pinned train split contains 34,602 questions from 21,953 unique image
sources (1.576 questions/source). It can support a new, much larger
source-disjoint learning experiment while retaining a genuinely untouched
formal bank.

## Outcome-independent source allocation

If this plan is activated, rank all train `image_id` groups by

`SHA256("beyond-entropy-textvqa-train-scale-v1" + NUL + 20260828 + NUL + image_id)`.

Assign whole sources before any rollouts:

| Hash-rank offsets | Role | Planned sources | Approx. questions |
| --- | --- | ---: | ---: |
| 0--4,999 | ranker training | 5,000 | 7,881 |
| 5,000--7,999 | threshold/risk calibration | 3,000 | 4,729 |
| 8,000--12,999 | one-shot formal test | 5,000 | 7,881 |
| 13,000 onward | untouched reserve | 8,953 | 14,111 |

Question counts are planning estimates from the split-wide mean. Actual counts
and all artifact hashes must be recorded after outcome-free export.

Before allocation is finalized, decoded RGB hashes must be compared with every
prior ChartQA, DocVQA, TextVQA, ChartQAPro, and HRBench bank. Any collision is
excluded and deterministically backfilled within the same role. Source IDs,
RGB hashes, and questions may be used for this audit; answers and model
outcomes may not influence allocation.

## Scientific role

This is not permission to revise a model on opened validation targets. The
train-split development and calibration banks form a new learning study; the
5,000-source train-split formal bank is opened exactly once after the complete
policy is frozen. Existing validation failures remain negative results.

The larger bank addresses a concrete weakness of the current experiment: its
factorized attention model was learned from only 200 TextVQA sources. The scale
study should compare, using the same source folds and visual action bank:

- the existing low-capacity logistic rescue/harm model;
- listwise or pairwise within-state action ranking;
- a state-level call head separated from the action ranker;
- risk-controlled threshold calibration; and
- data-scale curves at 200, 1,000, 3,000, and 5,000 ranker-training sources.

All architecture and scale selection occurs on the ranker-training sources.
The 3,000 calibration sources may choose only the already enumerated call/risk
thresholds. Neither bank may be reused as the one-shot formal test.

## Compute envelope

At the observed fresh-run throughput of roughly 30--32 questions/minute for
one answer-now plus four sibling crops on an RTX 4090, the 8,000-source
development/calibration bank is approximately 6.5--7 GPU-hours and the
5,000-source formal bank approximately 4--4.5 GPU-hours. Feature extraction and
model fitting add compute but remain substantially below the current account
quota. Jobs can be sharded by deterministic whole-source ranges and merged only
after each shard passes row-count and provenance checks.

These are planning estimates, not reservations. Every submitted shard must use
state-change email notification and a restart-safe checkpoint.

## Activation and stop rules

- Do not start this scale study until the fresh TextVQA result is locked.
- If the fresh result passes, treat scale as independent replication and
  learning-curve evidence; do not alter the successful policy retroactively.
- If the fresh result fails, close the current 200-source policy family and
  treat scale as a new method-development phase, not a retry on the same target.
- Freeze the complete policy, risk tolerance, cost, and primary interval before
  exporting or rolling out the 5,000 formal sources.
- Require a positive point estimate and strictly positive multiplicity-aware
  source-cluster lower bound. A safe zero-call policy is a failure.
- Keep at least the final 8,953 source groups untouched for independent
  replication or debugging of an execution failure; they are not a license for
  repeated hypothesis tests.

## Validity caveat

The source-disjoint train formal bank is a legitimate prospective test of
counterfactual acquisition transfer across unseen images, but it is not a new
benchmark. A main-conference package still needs another benchmark or model
family. Possible benchmark contamination in the frozen base VLM must be
disclosed; the primary quantity is paired improvement over that same model's
answer-now behavior, which mitigates but does not eliminate contamination
concerns.
