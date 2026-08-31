# Non-ScreenQA joint auxiliary action-proposer protocol v1

Status: frozen on 2026-09-01 after the DocVQA reserve ToolGate-style result
was opened, but before fitting any joint auxiliary candidate and while every
ScreenQA risk-calibration, formal, reserve, and untouched outcome remained
sealed.

This is a new development branch. It cannot revise the DocVQA formal or reserve
conclusions and may not use their outcomes for fitting, hyperparameter choice,
or candidate selection. Its purpose is to test the bottleneck identified by the
reserve comparison: selecting the right crop before deciding whether to call.

## Frozen development population

Use only the opened DocVQA ranker-development population:

- sibling rollouts:
  `artifacts/docvqa-train-factorized-v2/ranker-training/qwen3b-c4-seed0/rollouts.jsonl`;
- rollout SHA-256:
  `9109d5c896e59959937c2171e897dd81341237ff51fc5745fb4c56325ddd8fc3`;
- answer-NLL rows:
  `artifacts/docvqa-train-factorized-v2/proxy-to-outcome-cross-domain-v1/full-v1/merged/answer-nll.jsonl`;
- answer-NLL SHA-256:
  `f23e32bfbfa264f0362dd43881443c9c6ed507400d1fe7c2577688db5767e938`;
- label-free semantic features:
  `artifacts/docvqa-train-factorized-v2/ranker-training/attention-semantic-v1/features-question-region-attention-label-free.pt`;
- semantic-feature SHA-256:
  `05b3aaeabeef84993787c0ce676d175ab51a95bae34cabb5e950b0bd4f906686`.

The exact join must contain 3,500 source groups, 13,580 decisions, 54,320 ZOOM
rows, 1,604 positive-gain rows, 1,535 negative-gain rows, and 51,181 neutral
rows. Every decision must contain one ANSWER and four frozen UG-grid crops.
The merged NLL file retains four shard-specific config hashes
`35d5d0e2...`, `0663aaba...`, `4d6937fd...`, and `754b4f3f...`; this is
expected because shard identity is included in the config hash. All five rows
within one decision must share one config, and the merged provenance must bind
all four shards to the same Qwen revision, prompt, dtype, pixel bounds,
hardware family, rollout hash, and target rule.

No ScreenQA row, DocVQA calibration/formal/reserve row, or official benchmark
validation/test row is allowed in development.

## Inference-visible input

Use exactly the existing 46-dimensional `hybrid-context-semantic` action
feature vector. It is computed from the pre-action state, question/image
attention summary, and candidate geometry. It must not contain target answer,
correctness, reward, task gain, post-action entropy, crop answer, or
teacher-forced likelihood at inference.

Standardize each feature using only the training sources in the current OOF
fold. Constant features receive scale one. The serialized full-development
refit stores the means and scales.

## Registered targets

For every candidate crop:

1. `rescue = 1[correct_after > correct_before]`;
2. `harm = 1[correct_after < correct_before]`;
3. `loss_gap = mean_NLL(answer-now) - mean_NLL(candidate crop)`.

The answer target and `loss_gap` are training-only teacher supervision. The
loss gap is standardized by the weighted training-fold mean and standard
deviation before optimization. A non-finite or nonpositive scale fails closed.

## Sole model family and registered ablations

Fit a deterministic shared-trunk multilayer perceptron:

- input 46;
- hidden layers 32 then 16 with GELU;
- no dropout, batch normalization, layer normalization, quantization, or
  pretrained weight initialization;
- scalar rescue logit, harm logit, and loss-gap output heads;
- source-balanced row weights: equal mass to every source, then every crop row
  within the source;
- five whole-source folds from `_source_folds`, seed `20260904`;
- one NVIDIA H800, PyTorch deterministic algorithms, float32, full-batch
  AdamW; the formal runner fails closed on another accelerator family;
- 200 fixed epochs, learning rate `0.003`, weight decay `0.0001`, no early
  stopping, scheduler, clipping, or hyperparameter search.

Binary heads use separately normalized positive/negative weighted logistic
loss, giving equal class mass without changing source balance within a class.
The loss head uses weighted smooth-L1 loss with beta one.

Fit exactly three variants:

- `task_only`: rescue loss plus harm loss;
- `loss_only`: loss-gap loss only;
- `joint`: rescue loss plus harm loss plus `0.5 * loss_gap_loss`.

The task and joint proposal score is `sigmoid(rescue) - sigmoid(harm)`. The
loss-only score is the predicted standardized loss gap. Ties break by
lexicographically smaller `action_id`. There is no tool-call threshold in this
proposal experiment.

## Source-held-out evaluation

Every development decision receives one prediction from a model that excluded
its entire source. Refit each registered variant once on all development
sources only after OOF predictions are complete.

For each variant and for the frozen factorized proposer, entropy top crop,
fixed crop, exact uniform-random crop, and oracle, report:

- source- and question-balanced top-one task gain;
- top-one induced harm;
- helpful-state recovery and action-selection error conditional on at least one
  helpful crop;
- source-balanced pairwise gain differences;
- 20,000 whole-source percentile bootstrap resamples, seed `20260904`, with
  two-sided 95% intervals.

`task_only` is the primary architecture control. `loss_only` tests whether the
teacher proxy alone is sufficient. The existing frozen factorized proposer is
the incumbent method control.

## Mechanical advancement rule

Advance `joint` to a ScreenQA calibration candidate only if all conditions hold
on the source-held-out DocVQA predictions:

1. the 95% lower endpoint of joint minus `task_only` source-balanced top-one
   task gain is strictly positive;
2. the joint top-one task-gain point estimate is strictly above the frozen
   factorized proposer;
3. joint helpful-state recovery is strictly above both `task_only` and the
   frozen factorized proposer;
4. joint induced harm is no greater than `task_only` and no greater than the
   frozen factorized proposer.

If any condition fails, record `joint_auxiliary_proposer_not_advanced`; do not
open ScreenQA calibration. A failure may motivate a newly declared method on
the same opened DocVQA development population, but the failed result must be
retained and no claim may be transferred to ScreenQA without a new freeze.

## Protected next stage

If and only if the mechanical rule passes, serialize and hash the sole full
DocVQA refit, its feature contract, OOF report, implementation, tests, and
threshold family before exporting any ScreenQA calibration outcome. ScreenQA
uses its immutable allocation SHA-256
`ccfc2c0f18d36f6b31a6200c31a991d75ba6bb6ed3160b72ed5cfcca25473c49`:

- risk calibration: 4,001 images in 1,016 source components;
- one-shot formal: 6,000 images in 1,471 source components;
- reserve and untouched roles remain sealed.

Calibration may choose only among a separately frozen finite call-threshold
sequence and must enforce induced-harm, negative-call, minimum-call, and
minimum-utility rules before formal activation. The formal population is
opened once and cannot be reused for revision.

The existing Qwen2.5-VL-7B/H800 result is mechanism evidence only. Any 7B
deployment replication requires its own pre-outcome activation document and
must not silently replace the 3B actor in this development protocol.

Every submitted compute task must email `yihangc@connect.hku.hk` for all state
changes. No GitHub push is authorized by this protocol.

An implementation-only smoke may use fewer epochs and resamples solely to
exercise joins, folds, and serialization. Its metrics are non-scientific, may
not change any registered model or advancement condition, and must not be
reported as the registered OOF result.
