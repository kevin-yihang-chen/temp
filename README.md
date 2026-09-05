# Beyond Entropy

Research scaffold for learning the **pre-action, task-relevant value of visual
information** from counterfactual sibling rollouts.

The learned quantity and deployment decision are deliberately separated:

```text
gain(s, a) = P(success after a | s) - P(success now | s)
VOI(s, a)  = predicted_gain(s, a) - lambda * cost(a)
```

Version 0.3 adds the first frozen-VLM execution path to the v0.2 pipeline:

- type-level isolation between agent-visible state and ground truth;
- additive `ORIGINAL + ZOOM` observation semantics;
- paired stochastic replicates and generation seeds;
- image/source-level train/test splitting;
- cost-independent gain training and runtime lambda sweeps;
- total-executed-cost policy utility;
- adaptive entropy baselines and richer entropy ranking diagnostics;
- serial/batch backend interfaces with an in-memory cache;
- an optional semantic ROI gain head for frozen visual/text embeddings;
- an optional Qwen2.5-VL backend with generated-token predictive entropy;
- UG-compatible overlapping crop geometry with budgeted 4/9-crop subsets;
- V*Bench-style multiple-choice and ChartQA relaxed scorers;
- a portable frozen-slice manifest, state-bootstrap intervals, provenance
  capture, and a Slurm smoke job.

The included smoke fixture is synthetic and validates execution only. It is not
a benchmark result. Frozen V*Bench/ChartQA diagnostics and independent ChartQA
confirmation infrastructure are implemented. A bounded real RL integration has
also run, but it stopped at its preregistered tool-support gate; large-scale
method training is not currently authorized.

## Current status and development record

The repository is an active research codebase, but it does **not** currently
contain a top-conference-ready positive result. The frozen 36-cell audit of
whether pre-action VLM state predicts the value of one fixed four-crop visual
tool is now complete on source- and decoded-RGB-disjoint ChartQA, DocVQA, and
HRBench splits. All 36 predictor/target cells and three fixed seeds were
evaluated in a ledger-first, one-shot held-out transaction with 20,000 paired
whole-source bootstrap resamples.

The terminal status is **INCONCLUSIVE**. The fixed tool has statistically
positive oracle headroom on all three benchmarks, but neither the tested
deployable pre-action policies nor the diagnostic post-action probe established
stable positive utility. None of the preregistered `GO`, `PIVOT`,
`REPRESENTATION`, or `STOP` branches fired; the user explicitly accepted the
protocol's fail-closed inconclusive branch as the final report. The consumed
test is not reused for model, threshold, seed, or verdict selection.

- [项目当前状态](PROJECT_STATUS.md)
- [研究计划与路线决策](PLANS.md)
- [项目发展记录](PROJECT_DEVELOPMENT.md)
- [完整实验账本](EXPERIMENTS.md)
- [固定工具 predictability 协议](docs/predictability_audit_protocol_v1.md)
- [最终 predictability 审计（INCONCLUSIVE）](artifacts/predictability-audit-v1/formal-test-once-v1/PREDICTABILITY_AUDIT.md)
- [N5 回顾性结果](artifacts/docvqa-train-factorized-v2/ops/n5-information-set-retrospective-result-20260903-v1.md)

## Quick start

The core pipeline needs only the Python standard library:

```bash
cd /userhome/cs3/yihangc/Documents/beyond-entropy
PYTHONPATH=src python3 -m beyond_entropy demo --output-dir artifacts/demo-v2
pytest
```

The demo produces an image-grouped train/test split, a serialized gain model,
JSON metrics, tuned entropy baseline thresholds, and a Markdown report.

Individual commands:

```bash
PYTHONPATH=src python3 -m beyond_entropy simulate \
  --output artifacts/counterfactual-v2.jsonl \
  --n-states 600 --num-candidates 4 --questions-per-image 2

PYTHONPATH=src python3 -m beyond_entropy diagnose \
  --data artifacts/counterfactual-v2.jsonl

PYTHONPATH=src python3 -m beyond_entropy train \
  --data artifacts/demo-v2/train.jsonl \
  --output artifacts/demo-v2/gain_model.json

PYTHONPATH=src python3 -m beyond_entropy evaluate \
  --data artifacts/demo-v2/test.jsonl \
  --model artifacts/demo-v2/gain_model.json \
  --lambda-cost 0.05
```

`lambda_cost` is no longer a training argument. The same gain model can be
evaluated under any non-negative lambda.

## Frozen Qwen diagnostic

Prepare a JSONL manifest using the schema in `docs/data_contract.md`, then run:

```bash
PYTHONPATH=src python -m beyond_entropy collect-qwen \
  --manifest data/vstar-frozen/manifest.jsonl \
  --output artifacts/gate1-vstar/rollouts.jsonl \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision <pinned-hugging-face-revision> \
  --scorer vstar --candidate-count 4 --generation-seeds 0
```

The command defaults to offline model loading, deterministic decoding, the UG
crop ratio of two, and four spatially balanced candidates. It writes sibling
rollouts plus `.diagnostic.json` and `.provenance.json` sidecars. `--resume`
checkpoints every completed state and continues safely after preemption or a
time limit. Formal runs should also pass `--expected-manifest-sha256` so a
changed frozen slice fails before model loading.

Freeze a category-balanced V*Bench slice with:

```bash
HF_HOME=/userhome/cs3/yihangc/Data/hf_cache PYTHONPATH=src \
  python scripts/export_benchmark_manifest.py \
  --task vstar --output-dir data/vstar-frozen-64 \
  --count 64 --seed 17 \
  --dataset-revision b44023b4dca749ed8a76b85eb576627d05a1c174
```

To validate the locally cached 7B model on one synthetic image through Slurm:

```bash
scripts/submit_qwen_smoke.sh
```

The frozen 64-state Qwen-3B V*Bench pilot is submitted with:

```bash
scripts/submit_vstar_pilot.sh
```

The submit wrapper reads the notification recipient from the private,
git-ignored `.slurm-notify-email` file and requests email for all Slurm state
changes. This keeps contact information out of the public repository.

## Final fixed-tool predictability checkpoint

The completed formal report is `formal_claim_eligible=true`, frozen before test,
and complete at 36/36 scientific cells (108 seed-specific fits). Oracle utility
is `+0.02320` on ChartQA, `+0.01935` on DocVQA, and `+0.05000` on HRBench, with
all three 95% intervals above zero. However, the primary deployable policy's
paired lower confidence bound is non-positive on every benchmark, and the
post-action diagnostic has no positive lower confidence bound on any benchmark.
This closes the tested static-router claim without claiming that a future,
separately preregistered sequential acquisition method must fail.

## Earlier scientific checkpoints

The complete 2,500-state ChartQA development diagnostic establishes sparse
counterfactual headroom: answer-now accuracy is 0.8128, exhaustive four-crop
entropy search gains 0.0192 accuracy but has utility -0.1808 at
`lambda=0.05`, while oracle VOI gains 0.0504 with utility 0.0479. Lower entropy
is often not task improvement, and indiscriminate search is too expensive.

A factorized pre-action stopping gate is positive under nested image-grouped OOF
evaluation (utility 0.00662; state and image intervals both above zero). Its
frozen transfer to 1,918 image-disjoint ChartQA validation states has utility
0.00342, but the pre-registered state interval `[-0.00003, 0.00719]` narrowly
crosses zero, so the primary confirmation is a failed near miss. A fixed-crop
secondary is positive, but paired contrasts do not establish learned or fixed
spatial action selection over random.

A separately frozen chart-layout proposal advantage also fails to replicate:
its image-disjoint treatment-minus-UG estimate is +0.00175, with both state and
image intervals crossing zero. Its pre-registered follow-up is therefore not
launched.

The unchanged gate subsequently passes its separately registered 4,500-image
high-power replication. At `lambda=0.05`, frozen-policy utility is 0.00363 with
state interval `[0.00070, 0.00669]` and image interval
`[0.00070, 0.00671]`; accuracy gain is 0.00683 at 6.4% tool use. The effect is
stronger on human questions and not individually confirmed on the augmented
stratum. This independently confirms the stopping component of Gate 2, but not
spatial action selection. Gate 3 may therefore open only for a bounded
when-to-call integration/ablation; where-to-look learning remains gated. See
`docs/pilot_results_2026-08-28.md` and
`docs/replication_protocol_chartqa_train.md` for the evidence hierarchy,
frozen protocol, and completed result.

The bounded runtime integration is documented in
`docs/gate3_when_to_call.md`. Its API accepts only pre-action state fields and
returns `ANSWER` or `CALL_VISUAL_TOOL`; it deliberately cannot choose a crop.

The newer action-specific value branch establishes substantial crop oracle
headroom on DocVQA and TextVQA, but both frozen DocVQA learned policies failed
outcome-unseen confirmation. The attention policy's DocVQA utility is
`-0.00573` with a 97.5% source interval fully below zero, despite oracle utility
`+0.03394`; post-hoc decomposition identifies poor stopping precision and
weaker learned ranking as separate transfer failures. These are retained as
negative results. The subsequent pre-registered 2,000-source, source/RGB-
disjoint TextVQA attention confirmation improves raw score by `+0.00467` with
a 97.5% interval `[+0.00111, +0.00855]`, but its 9.29% tool rate leaves utility
at only `+0.00003` with interval `[-0.00354, +0.00385]`. It therefore also
fails the frozen cost-sensitive criterion. The current 200-source attention
family is closed; the next method phase uses separate risk calibration and a
larger TextVQA train source bank. See
`docs/docvqa_attention_action_value_formal_result_2026-08-28.md` and
`docs/textvqa_attention_action_value_fresh_formal_result_2026-08-28.md`.

That scaled bank is now frozen at the identity level. The pinned TextVQA train
split supplies 5,000 ranker-training sources (7,912 questions), 3,000 risk-
calibration sources (4,712 questions), and 5,000 reserved formal sources. A
decoded-RGB audit against 21 prior manifests found zero overlap, and the two
development manifests are mutually source/RGB-disjoint. The formal manifest
and formal rollouts remain unopened until the complete policy is frozen; this
is a data-preparation milestone, not a positive result. See
`docs/scaled_textvqa_train_contingency.md`.

The preregistered scaled pairwise primary subsequently failed independent risk
calibration: no non-degenerate threshold met every frozen condition, so the
formal bank remains sealed. A post-failure factorized OOF diagnostic corrected
an inconsistency between decision-weighted training and source-balanced risk
evaluation. Its context-state/semantic-action branch reaches source-balanced
utility `0.001175` at a 1.03% development-only tail while passing both risk
diagnostics. This is the first non-degenerate development tail above the frozen
`0.001` floor, but it was selected after the original calibration bank was
opened and is not confirmation. See
`docs/textvqa_factorized_source_balanced_oof_result_2026-08-29.md`.

The prospectively isolated factorized-v2 candidate subsequently **failed** its
fresh 3,000-source fixed-sequence calibration. The closest safe threshold
called on 1.25% of sources but reached source-balanced utility `0.0009917`,
only `0.0000083` below the frozen `0.001` floor; the next safe threshold fell
to `0.0009000`, and the following threshold failed the registered negative-call
risk test. No threshold was selected, the answer-now model was retained, and
the 5,953-source formal manifest and outcomes remain unmaterialized. This is a
strict negative calibration result, not permission to relax the floor or reuse
the sealed bank. See
`docs/textvqa_factorized_v2_independent_calibration_result_2026-08-29.md`.

## Semantic gain head

Install the optional PyTorch dependency:

```bash
python3 -m pip install -e '.[semantic]'
```

`SemanticGainHead` fuses:

```text
question embedding
+ global visual embedding
+ ROI-pooled candidate embedding
+ question-region interaction
+ bbox geometry
+ baseline state signals
-> predicted Delta success
```

`extract-qwen-features` now runs the frozen Qwen vision encoder once per original
image and restores its merged visual tokens to raster order. All candidate ROI
embeddings are pooled from that one grid, so no counterfactual crop outcome is
available to the head. `fit-semantic` uses an inner image-grouped validation split
for early stopping and monotone gain calibration, then reports an untouched outer
test split across a runtime lambda sweep.

```bash
PYTHONPATH=src python3 -m beyond_entropy extract-qwen-features \
  --rollouts artifacts/gate1-vstar/rollouts.jsonl \
  --output artifacts/gate2-vstar/features.pt \
  --model-revision <pinned-hugging-face-revision>

PYTHONPATH=src python3 -m beyond_entropy fit-semantic \
  --features artifacts/gate2-vstar/features.pt \
  --rollouts artifacts/gate1-vstar/rollouts.jsonl \
  --output-dir artifacts/gate2-vstar/model
```

## Project map

```text
src/beyond_entropy/
  rollout.py     GT-safe state/action/outcome interfaces, batching, caching
  schema.py      validated paired action-outcome record
  dataset.py     JSONL IO, replicate validation, grouped splitting
  features.py    pre-action scalar encoding and leakage guards
  model.py       cost-independent ridge gain baseline
  semantic.py    optional ROI semantic gain head
  policies.py    stopping/search/learned/oracle policies and threshold tuning
  metrics.py     SCGR, Top-1 mismatch, policy utility and efficiency metrics
  simulate.py    controlled synthetic pipeline test
  cli.py         experiment commands
docs/
  data_contract.md
  reference_integration.md
  research_plan.md
  review_remediation.md
```

The integration design is based on the official
[UG framework](https://github.com/ExplainableML/ug-framework) and
[VTool-R1](https://github.com/VTOOL-R1/vtool-r1) repositories. Their code is not
vendored here.

## Scientific guardrails

- Ground truth is accepted only by the scorer.
- Split by image/source, never by sibling action row.
- Post-action entropy, answers, correctness, labels, and rewards are not model
  inputs.
- The model predicts success gain; policy code alone applies cost preference.
- Entropy search pays for every candidate it executes.
- Synthetic demo numbers are not empirical research results.
- Pin upstream commits, model revisions, prompts, datasets, and seeds before real
  experiments.
