# InfographicVQA DECAR train-development protocol v1

Status: frozen on 2026-09-01 after the outcome-blind transport/source audit
passed, but before any InfographicVQA question text, answer, task outcome,
teacher-forced answer likelihood, or Qwen task endpoint was read or computed.

DECAR denotes **Decoupled Counterfactual Acquisition Router**. It is a bounded
train-only method family. It tests whether a loss-distilled `where` model and a
separately cross-fitted harm-aware `when` model can jointly select one visual
crop or stop before executing any candidate crop.

## Bound data roles and identities

Use only the pinned public-mirror official-train transport and the accepted
source audit:

- 24 train parquets; 1,981,251,656 aggregate bytes;
- download manifest SHA-256:
  `ecc46c6a073ebd89fc114cba6fee5c711c8600e596b5a785bec981d98b168f13`;
- source-audit report SHA-256:
  `1c801a641c13747a1be2abbcc3c4a8b2d0a32e33599caa36d86208853c866547`;
- source manifest SHA-256:
  `fc577513dd8f9993f40d14454c7ec4ecf48897ff0d1660479fb5c49d3ae9512a`;
- pilot-source manifest SHA-256:
  `75f20c141ccc273dcc36a4527ec7697826e3fea4b2bfc110754027ad9bb9ffe3`;
- population: 23,946 questions, 4,406 images, and 2,204 indivisible
  hostname/RGB-connected source components.

Official validation is sealed calibration: do not download or inspect it until
the train OOF rule below advances exactly one candidate and its implementation,
full-train refit, feature schema, threshold family, and validation gate are
hashed. Official test is sealed one-shot formal evaluation and cannot be opened
unless validation passes. Existing ScreenQA protected roles remain sealed.

## Outcome-blind population and folds

The registered train OOF population is all 23,946 questions. Construct five
outer folds without reading question text, answers, or outcomes:

1. sort source components by descending question count;
2. break equal-count ties by ascending
   `SHA256("infovqa-decar-outer-v1" + NUL + "20260917" + NUL + source_id)`,
   then `source_id`;
3. assign each source to the fold with the smallest current question count,
   then the smallest source count, then the smallest fold index.

The entire connected source component stays in one fold. Every fold assignment
and its row/source counts must be serialized before model endpoints are
computed.

For each outer training split, construct four inner folds by the same algorithm
with namespace `infovqa-decar-inner-v1`, seed `20260917`, and the outer-fold
index included after the seed. Inner predictions train the `when` model; no
question may receive an OOF prediction from a model that saw its source.

The engineering pilot contains the frozen 512 source IDs. Select one question
per pilot source by ascending
`SHA256("infovqa-decar-pilot-question-v1" + NUL + "20260917" + NUL +
source_id + NUL + question_id)`, then `question_id`, for exactly 512 decisions.
The pilot may validate runtime, memory, checkpoint/resume, prompt, scoring, and
output contracts only. Its task endpoints cannot change the population,
features, models, loss weights, hardware class, or advancement rule and must
not be reported as scientific evidence.

## Frozen actor, action family, and task score

- Actor/scorer: `Qwen/Qwen2.5-VL-7B-Instruct`, revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Inference: bfloat16, SDPA, deterministic decoding with seed `0`, no
  quantization, no CPU/disk offload, `min_pixels=200704`,
  `max_pixels=602112`, `max_new_tokens=32`, and system prompt
  `You are a helpful assistant.`
- Model prompt: normalized question followed exactly by
  `\nAnswer the question using a single word or phrase.`
- Stop/ANSWER uses the original image once.
- Four ZOOM actions use the existing `UGGridProposer(candidate_count=4,
  visual_crop_ratio=2.0, visual_cost=1.0)`: construct the complete official-UG
  50%-overlap square grid, then take its deterministic four-anchor
  spatially-balanced subset. Action IDs are `ug-grid-00` through `ug-grid-03`.
- A ZOOM answer receives the original image followed by the selected crop,
  where the crop is resized to the original dimensions with Lanczos exactly as
  in `Qwen25VLBackend`. It executes one crop, never the other three.
- Primary task score is official DocVQA-style ANLS, which the InfographicVQA
  paper specifies as a task metric. For multiple accepted answers, use the
  maximum normalized Levenshtein similarity with the existing `0.5` cutoff.
  Exact normalized accuracy is secondary and cannot select a policy.
- Per executed crop cost is `lambda=0.05`. Thus primary/random/fixed one-crop
  policies pay `0.05`; a method that observes all four candidate crops pays
  `0.20`, even if it returns only one answer.

For crop action `a`, define `delta(a) = ANLS_after(a) - ANLS_before`,
`utility(a) = delta(a) - 0.05`, `rescue=1[delta>0]`, `harm=1[delta<0]`, and
`neutral=1[delta=0]`. No tolerance or answer-type-specific relabeling is
allowed. Also record SCGR, the fraction with entropy reduction above zero but
negative `delta`.

## Inference-visible feature contract

Run one frozen-Qwen encoding of the original image, never a candidate crop, to
obtain:

1. mean final hidden state from text-only contextualization of the normalized
   question (`contextual_text_mean`);
2. the mean raster-restored full-image vision-merger embedding;
3. four ROI-pooled embeddings from that same full-image visual-token grid;
4. normalized candidate `xyxy`, width, height, area, center, full-grid size,
   original aspect ratio, and log pixel count; and
5. answer-now generated-token count, mean and maximum normalized token entropy,
   and mean token log probability.

All features exist before any crop action. The target answer, answer type,
operation/reasoning label, OCR annotation, correctness, `delta`, post-action
answer/entropy, teacher likelihood, source hostname, transport shard, and fold
ID are forbidden as inference inputs. Serialized OOF prediction rows must be
outcome-free; evaluation joins outcomes by identifiers only after predictions
and hashes are final.

## DECAR: decoupled `where` and `when`

Both networks use float32 and deterministic PyTorch algorithms. Standardize
each scalar feature and each embedding dimension using only the applicable
training sources. Constant dimensions receive scale one. Source-balanced row
weight gives equal total mass to every source, then equal mass to questions
within a source, then equal mass to the four candidates of a question.

### Loss-distilled `where` proposer

For every candidate, independently project question, global-image, and ROI
vectors to 128 dimensions. Concatenate projected question, global, ROI,
question-times-ROI, question-times-global, global-times-ROI, ROI-minus-global,
and the 16 registered scalar/geometry features. This is a 912-dimensional
fusion vector. Apply an MLP `912 -> 256 -> 64 -> 1` with GELU after the first
two layers and no dropout, normalization, or shared parameters with the `when`
model.

The training-only target is
`mean_NLL(answer-now target) - mean_NLL(candidate target)` under the exact same
7B prompt/image contract. Standardize this target using the weighted training
fold mean and positive finite standard deviation. Optimize weighted smooth-L1
loss with beta one plus `0.5` times within-question pairwise logistic loss over
all unequal target pairs. At inference choose the largest predicted gap, with
lexicographically smaller action ID breaking an exact tie.

### Cross-fitted `when` triage

Within each outer training split, fit four inner `where` models and obtain one
source-held-out selected action, predicted gap, and top-one/top-two margin for
every outer-training question. Only these cross-fitted selected actions train
the outer `when` model. Refit `where` on all outer-training sources to select
the action for the outer-test questions.

The `when` input is the 912-dimensional selected-candidate fusion vector plus
its predicted gap and top-one/top-two margin. It uses a separate MLP
`914 -> 256 -> 64`, GELU, and two heads:

- three logits for `rescue`, `neutral`, and `harm`;
- one scalar for signed `delta`.

Optimize class-mass-normalized three-way cross entropy plus weighted smooth-L1
signed-delta loss with beta one and equal loss weights. Class normalization
gives each class equal total weighted mass; it may not discard neutral rows.
The fold-specific rescue and harm magnitudes are the source-balanced mean
positive `delta` and mean absolute negative `delta` in cross-fitted gate
training rows.

For an outer-test question define

`class_delta = P(rescue)*rescue_magnitude - P(harm)*harm_magnitude`

and

`score = 0.5*class_delta + 0.5*clip(predicted_delta,-1,1) - 0.05`.

A call is ineligible when `P(harm) >= P(rescue)` or `score <= 0`; otherwise it
is ranked by descending score, descending predicted loss gap, then state ID.
This produces a joint policy over ANSWER and the four crops without executing
an unselected crop.

For every neural fit use seed `20260917 + 100*outer_fold + 10*inner_fold +
variant_index`, full-batch AdamW, 200 fixed epochs, learning rate `0.001`,
weight decay `0.0001`, and no early stopping, scheduler, gradient clipping,
hyperparameter search, pretrained router initialization, or outcome-dependent
retry. A numerical failure stops the run; it cannot trigger a scientific
configuration change.

## Registered ablations and non-learned baselines

Fit exactly these architecture variants under the same folds and optimizer:

1. `decar` (primary): loss-distilled `where` plus three-way harm-aware `when`;
2. `task_value_only`: replace the `where` target with signed `delta` and remove
   teacher NLL, retaining the same nested triage;
3. `loss_only`: loss-distilled `where`; rank decisions by the top predicted
   standardized loss gap at each registered call count, with no triage;
4. `no_harm_head`: primary `where`; binary rescue-versus-other gate plus the
   signed-delta head, with no harm probability or veto.

Report these non-learned comparators on the identical sibling bank:

- ANSWER now;
- answer-now-entropy-gated uniform-random one crop, using exact expectation
  over four actions;
- answer-now-entropy-gated fixed `ug-grid-00` one crop;
- answer-now-entropy-gated UG, which executes all four candidate crops and
  returns the minimum-post-action-entropy answer;
- charged exhaustive UG, which observes all four crops, chooses minimum
  post-action entropy, and pays four crop costs;
- task oracle over the four crops, reported both with one-crop cost and with
  oracle stopping, never as a deployable baseline.

Random/fixed curves use the same complete-tie decision-call counts as the
primary curve. At a primary budget of `C` crop executions, gated UG may act on
at most `floor(C/4)` decisions, with complete ties retained and their true
additional executions charged. Any entropy method that examines all four
post-action answers is charged four executions; it cannot be described as a
one-call method.

## OOF operating points, statistics, and advancement

Evaluate complete-tie top-score operating points at nominal question-balanced
call rates `0.005`, `0.01`, `0.02`, `0.05`, and `0.10`, restricted to eligible
DECAR calls. Report both question-balanced and source-balanced results. Primary
selection and inference use source-balanced utility.

At every operating point report ANLS gain, utility, executed-crop rate, calls
and distinct called sources, gain/call, helpful-call precision, helpful-state
recovery, induced harm, negative-utility call mass, action-selection regret,
oracle-stop regret, entropy disagreement, SCGR, and the full baseline/ablation
table. Use 20,000 iid whole-source bootstrap resamples, seed `20260917`, and
two-sided 95% percentile intervals. Preserve the paired resample indices for
all policy differences.

An operating point qualifies only when:

1. it has at least 100 calls from at least 50 distinct source components;
2. its 95% lower endpoint for source-balanced utility is strictly above zero;
3. its source-balanced utility point estimate is strictly above each feasible
   non-oracle baseline at no greater executed-crop budget;
4. its utility point estimate is strictly above `task_value_only`, `loss_only`,
   and `no_harm_head` at the identical call count;
5. induced harm and negative-utility call mass are each no greater than
   `no_harm_head` and the strongest feasible non-oracle baseline; and
6. all source exclusion, nested-OOF, feature leakage, target leakage, action
   coverage, cost accounting, tie, finite-score, fold balance, serialization,
   and bootstrap audits pass.

If multiple points qualify, select the one with highest source-balanced
utility, then lower induced harm, then lower nominal call rate. Advance exactly
one full-train DECAR refit to sealed validation only if a point qualifies. The
train result is `decar_not_advanced` otherwise; validation remains sealed.

The implementation-only pilot must not apply this advancement rule. No pilot
or train endpoint may alter a registered choice. If train OOF is negative,
retain the complete result and decompose fixed-policy error into action-choice,
gate false-positive, gate false-negative, and source-concentration terms. A
scientifically different successor requires a new dated protocol that cites
the failure; no unreported grid expansion or result deletion is allowed.

## Execution and publication rules

The 512-decision pilot runs before the full bank. It must measure rollout,
teacher-NLL, and original-image feature throughput separately and project full
four-shard wall time and GPU-minutes without using endpoint values. For a full
projection above one hour, compare live queue plus runtime and quota for four
H800/H100 versus four RTX 4090 GPUs. Prefer four H800/H100 when the complete
unquantized 7B run fits quota; every full shard must use one accelerator class.

Use source-aligned shards, complete five-record atomic checkpoints, exact-prefix
resume, immutable input/code hashes, and byte-audited merges. Every submitted
compute task must email `yihangc@connect.hku.hk` for BEGIN, END, FAIL, REQUEUE,
and every other supported execution-state change. Credentials must never enter
scripts, logs, manifests, or artifacts. No GitHub push is authorized.
