# DocVQA-train factorized-v2 prospective preregistration

Status: frozen on 2026-08-29 while TextVQA factorized-v2 label-free feature
extraction job `191717` was still running and before fixed-sequence calibration
job `191792` produced an outcome. No DocVQA-train question, answer, image, model
rollout, feature, score, or allocation result was read to choose this protocol.
No DocVQA-train role has been materialized by this branch.

## Purpose and relation to prior evidence

This is the pre-result specification for one source-disjoint second-benchmark
replication of the factorized acquisition policy. It does not erase or reuse
the earlier DocVQA-validation development and formal banks. Those 200-source
development and 400-source one-shot formal results remain historical evidence,
including the negative formal utility result. They are excluded from training,
calibration, formal evaluation, and method selection in this branch.

The classic DocVQA paper reports 39,463 training questions on 10,194 document
images. The allocation below is therefore planned at document-image level, not
question level. The authoritative public references are the
[DocVQA dataset page](https://www.docvqa.org/datasets/docvqa), the
[DocVQA paper](https://arxiv.org/abs/2007.00398), and the pinned
[LMMs-Lab snapshot](https://huggingface.co/datasets/lmms-lab/DocVQA/tree/539088ef8a8ada01ac8e2e6d4e372586748a265e/DocVQA).
Reported counts are planning bounds only; the allocator must fail closed unless
the pinned snapshot supplies enough eligible unique source groups after every
exclusion below.

## Frozen dataset and identity contract

- dataset: `lmms-lab/DocVQA`, configuration `DocVQA`, split `train`;
- revision: `539088ef8a8ada01ac8e2e6d4e372586748a265e`;
- source-group key: the normalized public `docId`;
- selection namespace: `beyond-entropy-docvqa-train-factorized-v2`;
- seed: `20260829`;
- source rank: `SHA256(namespace + NUL + seed + NUL + docId)`, ascending,
  with normalized `docId` as the deterministic tie breaker;
- every question belonging to an accepted `docId` stays in the same role;
- selection may read only row identity, `docId`, and decoded image bytes needed
  for integrity checks. It must not read answers, question text, question type,
  OCR, model output, correctness, gain, entropy, or any learned feature;
- decoded images are converted to RGB and hashed over an unambiguous
  width/height/byte payload before any outcome-bearing manifest is written.

The allocator must exclude and record a source group if any of its decoded-RGB
digests collides with:

1. another public source ID already accepted in this allocation;
2. either earlier DocVQA-validation bank;
3. any prior Beyond Entropy development, calibration, or formal manifest for
   which a decoded-RGB digest is available; or
4. another role in this allocation.

For each exclusion, take the next source in the same immutable hash order. Do
not change the namespace, seed, role sizes, or ranking after inspecting the
allocation. The audit must report row count, unique source count, unique RGB
count, every exclusion and backfill, pairwise state/source/RGB overlap, and the
SHA-256 of every identity input. Any missing prior-bank digest, duplicate
accepted source, unresolved collision, or insufficient eligible source count
invalidates allocation rather than shrinking a role.

## Frozen roles and access sequence

After deterministic exclusions and backfills, assign eligible ranked sources
in this exact order:

| Eligible rank interval | Role | Source groups | Outcome access |
| --- | --- | ---: | --- |
| 0--3,499 | ranker training | 3,500 | model fitting and OOF diagnostics only |
| 3,500--5,999 | fixed-sequence risk calibration | 2,500 | sole frozen candidate calibration only |
| 6,000--9,499 | one-shot formal | 3,500 | only after calibration success and policy freeze |
| 9,500 onward | deterministic reserve | all remaining | identity backfill only |

The allocation file may store formal `docId` and RGB identities so disjointness
can be audited, but it must not store formal questions, answers, images, target
payloads, or model outcomes. Initially only ranker-training and calibration
manifests may be exported. The formal manifest is not exported and formal
records are not loaded until the successful calibrated policy and complete
implementation are frozen and hashed.

The role sizes are immutable. If fewer than 9,500 eligible unique source groups
remain, this preregistered branch is infeasible and closes without selecting a
smaller or different sample.

## Sole frozen candidate

Fit exactly one benchmark-specific candidate on the 3,500 ranker-training
sources. It reuses the scientific design selected before this preregistration:

- base model: `Qwen/Qwen2.5-VL-3B-Instruct`, revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`;
- actions: answer now or call exactly one of the same four frozen UG-grid crop
  candidates; one executed crop costs `lambda=0.05`;
- model type: `multidomain_factorized_action_value` restricted to the DocVQA
  domain, with a 27-dimensional pre-action baseline-error head and frozen
  original-image semantic crop rescue/harm features;
- feature mode: `hybrid-context-semantic`;
- training protocol: five whole-source folds, seed `20260829`, alpha `1.0`,
  with equal source then equal row weighting inside the sole domain;
- deployment inputs exclude answer text, post-action entropy, target,
  correctness, crop outcome, and reward;
- prompt, image preprocessing, four crop coordinates, generation parameters,
  scorer, label-free feature schema, feature layers, and numerical precision
  must be frozen before the first calibration rollout.

OOF diagnostics on ranker training may verify implementation and report
mechanism, but they cannot select a second feature mode, alpha, cost, model
class, crop set, or base model. Refit the same heads on all 3,500 sources once.

Construct the strict-to-permissive calibration threshold sequence from the
refitted candidate's ranker-training scores only:

1. one floating-point step above the maximum score;
2. observed score order statistics targeting source-balanced call rates
   `0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03`;
3. numeric tie deduplication while retaining strict descending order.

Threshold construction does not read calibration or formal records. Hash the
candidate, threshold sequence, training rollouts, training label-free features,
OOF report, dataset allocation, and this protocol before calibration export.

## Fixed-sequence risk calibration

Run the exact frozen candidate over every question from all 2,500 calibration
sources. Average every tested loss within source, then traverse the frozen
thresholds from strict to permissive. Jointly test:

- induced-harm mass at most `0.005`;
- net-negative-call mass at most `0.02`.

Use the same bounded-mean Bernoulli-KL lower-tail test as factorized-v2,
family error `0.05`, and per-risk cutoff `0.025`. Continue only while both risk
tests pass; stop at the first joint failure and leave every later threshold
untested. Among preceding risk-accepted thresholds, choose the most permissive
threshold satisfying both empirical non-degeneracy floors:

- source-balanced call rate at least `0.01`;
- source-balanced utility at least `0.001`.

If no threshold qualifies, select answer now, retain the calibration as a
negative result, and do not materialize formal. Do not rerun with a changed
threshold order, constraint, cutoff, cost, utility floor, call-rate floor,
weighting, candidate, or feature.

## One-shot formal evaluation

Only after calibration selects a non-degenerate safe threshold may the formal
implementation branch be merged, verified, and frozen. The policy freeze must
pin the candidate and calibrated threshold, allocation and collision audit,
prompt, collector, scorer, feature contract, risk code, evaluator, matched-call
baselines, bootstrap, Slurm commands, code revision, and all input hashes before
formal export.

Evaluate the exact policy once on every question from all 3,500 formal sources.
The primary estimand is source-balanced utility:

`mean_source mean_question call(x) * (gain(x) - 0.05)`.

Use 20,000 whole-source bootstrap resamples, seed `20260829`, and a two-sided
97.5% percentile interval. This branch passes only if all conditions hold:

- source-balanced utility is positive;
- its 97.5% lower bootstrap endpoint is strictly positive;
- question-weighted utility is positive;
- source-balanced call rate is at least 1%;
- the evaluated threshold exactly equals the calibration choice; and
- every frozen hash and identity audit matches.

Mandatory diagnostics are raw gain, call rate, cost, gain per call, induced
harm, negative-call mass, unnecessary calls, correct stopping, crop rescue,
oracle utility/regret, matched-call random and entropy gates, fixed/random crop
controls, and exhaustive UG-style entropy search charged for every candidate
call. A failed condition is retained as negative; no replacement threshold,
candidate, feature, scorer, baseline, or sample is selected on formal outcomes.

## Cross-benchmark claim and execution rule

This protocol is frozen regardless of the pending TextVQA factorized-v2 result.
For resource scheduling only, DocVQA-train materialization and rollout begin
after the TextVQA one-shot branch reaches its preregistered decision. This delay
cannot change any rule above.

The general cross-benchmark claim requires the conjunction of successful
prospective TextVQA and DocVQA one-shot formal decisions. If TextVQA calibration
or formal evaluation fails, a later DocVQA result may be reported as its own
preregistered benchmark-specific replication or negative result, but it cannot
rescue the failed conjunction or be described as evidence that the method is
generally effective. Model-family generality likewise requires an additional
pre-result base-model replication; Qwen2.5-VL-3B alone is not sufficient.

Execution failures may resume an identical hashed collector or feature
extractor. A scientific-contract change requires a new, independently allocated
branch written before accessing its calibration or formal outcomes.
