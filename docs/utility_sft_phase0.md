# Counterfactual Utility SFT: Phase 0 audit and implementation contract

Status: NO-GO after the single allowed development correction, 2026-09-06. This is
the user's supervised hypothesis, not a reopening or relabelling of the INCONCLUSIVE
static-router audit. No RL, 7B, continuous boxes, multi-turn agent, or publication
claim was executed. The negative result is retained as the scientific outcome.

## Audited interfaces

| Required field | Existing authoritative field |
| --- | --- |
| state / image / source | `ActionRecord.state_id / image_id / source_id` |
| image and question | `original_image / question`; manifest `AgentState.backend_prompt` preserves MCQ options |
| action identity and bbox | `action_type / action_id / candidate_bbox` (normalized xyxy) |
| paired replicate | `replicate_id / generation_seed` |
| ANSWER outcome | ANSWER sibling `answer_after / correct_after`, consistent with `correct_before` |
| ZOOM outcome | each sibling `answer_after / correct_after` |
| reward and gain | existing official-scorer `correct_after`; subtract paired ANSWER score |
| cost | `tool_cost`, zero for ANSWER; NOT a training target |

Reuse `dataset.validate_sibling_groups` and `group_by_decision`; `BBox` validation;
`benchmarks.load_manifest/scorer_by_name`; `rollout.collect_sibling_rollouts`;
`semantic.roi_pool_spatial_tokens`; `qwen_semantic.reshape_merged_visual_tokens`
and prompt builder; `predictability_audit.audit_split_disjointness`;
`predictability_features.decoded_rgb_sha256`; existing source-paired bootstrap
and exclusive hash-bound artifact writers. No modification to old rollout schema
is necessary. Existing frozen feature extractors detach and use inference mode:
they CANNOT be reused as the trainable forward pass.

Existing development banks contain 3,600/900 ChartQA, 10,861/2,719 DocVQA,
480/160 HRBench train/validation states, five siblings per state. Paths are under
`artifacts/predictability-audit-v1/formal-development-v1/{domain}/{role}/`.
Read and verify `complete.json` hashes before using these banks. They are opened
development data, not new independent tests. Job 208184 consumed the old tests.
New held-out allocation must exclude historical sources AND decoded-RGB hashes;
do not silently resplit an opened bank and call it an untouched test. HRBench 8k
has already been allocated completely; fresh two-domain confirmation may use
ChartQA and DocVQA if unused source/RGB inventory supports it. This remains to be
verified before formal training/test execution.

The materialized Utility-SFT datasets used by the current code are:

| Domain | Train JSON | Validation JSON |
| --- | --- | --- |
| ChartQA | `artifacts/utility-sft-v1/data/chartqa-train-full.json` | `artifacts/utility-sft-v1/data/chartqa-validation-full.json` |
| DocVQA | `artifacts/utility-sft-v1/data/docvqa-train-full.json` | `artifacts/utility-sft-v1/data/docvqa-validation-full.json` |
| HRBench | `artifacts/utility-sft-v1/data/hrbench-train-full.json` | `artifacts/utility-sft-v1/data/hrbench-validation-full.json` |

Their hash-bound inventory is `artifacts/utility-sft-v1/DEVELOPMENT_BUNDLE.json`.
Each JSON contains `inputs` (original image/question and legal actions), training-only
`labels.reward/gain`, paired seed/replicate provenance, and original outcomes for
audit. The selector API receives only `inputs`; it cannot receive labels/outcomes.

## Frozen MVP design choices (before new training outcomes)

- K=4, ANSWER index 0; lexicographically sorted existing crop IDs at indices 1..4.
  Every replicate must have identical identity, box and cost mapping. No candidate
  filtering based on outcomes. Single generation seed first; explicit mean mode
  requires complete, distinct, paired seeds. Original outcomes remain separate.
- Inference inputs are a strict typed allowlist: original image, original prompt,
  identifiers and candidate geometry only. IDs are for joining, not tokenized.
  No generated answers, entropy traces, reward or target fields go to the head.
  No uncertainty feature in v1, avoiding another baseline generation pass.
- Reuse one original-image vision forward. ROI pooling over its merged visual
  grid; final original-image/prompt language state represents the question. Train
  the utility head, visual merger, and last language decoder block. All three SFT
  arms have identical initialization, trainable parameters, examples and steps.
  This is partial VLM fine-tuning, not a frozen-feature head experiment. Verify
  nonzero gradients AND parameter updates in both VLM components on real input.
- Reuse `SemanticGainHead`, with an independent learnable ANSWER representation.
  Scores are ANSWER-anchored: `g_hat = raw - raw[ANSWER]`; `logits = g_hat / T`.
  Main loss is `KL(softmax(g/T) || softmax(logits))`, T=0.25. Thus `g_hat[0]=0`
  and perfect soft-target fitting recovers gains, not arbitrary probability units.
  No lambda or cost enters this loss. Pairwise ranking is an optional unrun ablation.
- Best-action uses ordinary CE with smallest-index tie breaking (ANSWER wins a
  zero-gain tie). Format/support is a gain-free negative control: stable hashed
  state identity assigns a uniform pseudo action class; CE uses no outcomes.
  In a discrete head every argmax is already legal even before SFT. Report that
  support validity is architectural, not evidence of utility or free-form tool use.
- Selection and answering are separate: fixed pretrained answer backend executes
  only the chosen action. Training the selector must not change bank reward
  semantics by also replacing the answer backend with the fine-tuned model.

## Execution and scientific gates

1. CPU contracts: paired grouping, shuffled row-order invariance, stable mapping,
   label/feature isolation, semantic ROI gradient, all losses, cost-only selection,
   source/RGB leakage rejection, and hash/exclusive-write negative tests.
2. Bounded real Qwen2.5-VL-3B train-only subset overfit: loss, positive/negative
   ranking, actual backbone gradients/updates, one vision call per state,
   latency/memory/disk. No test access; do not scale if sanity fails. One bounded
   hyperparameter correction is allowed after implementation issues are excluded.
   The sanity pool uses outcome-independent hash-ranked whole training sources;
   within that pool, select the first positive-gain state and first negative-gain
   state, then fill by state ID. This label-stratified overfit selection is shared
   by all three arms and is diagnostic only, never a generalization estimate.
   An initial two-step engineering run cannot pass the overfit gate regardless
   of its scores. Full sanity uses the three matched 80-step configurations.
   The three-domain pilot samples domain uniformly, then source uniformly, then
   a state within source; high-question-count DocVQA documents therefore cannot
   dominate gradient exposure.
3. Train/validation comparison for all three arms and eight required baselines.
   Export every arm's raw ANSWER-anchored score with no outcome-based rescaling or
   threshold fitting. Fixed lambdas 0, .01, .025, .05, .1, .2; primary .05. Include actual
   selector overhead and original-image token costs separately from crop cost.
4. Question shuffle, image shuffle, region ablation with deterministic, source-aware
   mappings. No expanding training if semantic utility is unaffected.
5. Freeze checkpoint/config/input hashes, calibration, seeds, metric definitions,
   test allocation, and one-shot access ledger BEFORE reading fresh test outcomes.
   Existing transaction primitives will be reused, not the old matrix-specific
   access authorization. Source-balanced paired 95% intervals, 20,000 bootstrap
   resamples; also report question-weighted scores. Replicate training seeds
   17/29/47 for final confirmation, no test selection among seeds.
6. Eight comparators: ANSWER, deterministic random crop, full four-crop UG charged
   four calls, existing frozen VOI (retain its full-tool cost and feature overhead),
   Format, Best-action, Utility, privileged Oracle. Selected single crops cost one;
   oracle is not deployable. Regret uses best raw gain, policy uses gain minus cost.
7. Deliver action-score/ranking plots and accuracy-cost frontiers without retraining.
   Answer the four user questions in `GO_NO_GO.md` only from verified evidence.
   GO needs superiority to frozen VOI and incremental value over Best-action on
   independent tests in at least two domains, lower regret and no exhaustive
   acquisition advantage. Failure/uncertainty never authorizes RL in this goal.

## Deliverable status

Final update 2026-09-06 03:53 HKT: implementation, three real SFT arms, the single
allowed 1024-step correction, frozen validation, eight-baseline evaluation,
20,000-resample source bootstrap, both figures, and semantic controls are complete.
The correction failed the development Go gate, so the protocol stopped before a
fresh test transaction. This is intentional test preservation, not missing evidence
being silently treated as success.

- [x] Data pipeline and structured action space
- [x] Trainable spatial utility head and three matched configurations
- [x] Deterministic eight-baseline evaluation and required metrics
- [x] Fresh-test gate enforced; transaction not created because development was No-Go
- [x] Real Qwen overfit and development-pilot evidence
- [x] Both figures and semantic controls
- [x] Final four-question `GO_NO_GO.md`

The six full development JSON files are materialized from the sealed sibling
banks and then referenced by one `utility_sft_development_bundle_v1` inventory.
The bundle loader accepts train/validation only, reconstructs labels from stored
paired outcomes, and jointly rejects source, image-ID, or decoded-RGB overlap.
It must remain `formal_test_eligible=false` and `test_data_present=false`.

Keep code/results local unless the user separately requests a push. Submitted
compute jobs must send `--mail-type=ALL` to `yihangc@connect.hku.hk`. Compare live
queue time, GPU runtime and GPU-hours before choosing single/multiple GPUs.
