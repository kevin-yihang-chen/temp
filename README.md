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
confirmation infrastructure are now implemented; large-scale RL remains a
future gate.

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

## Current scientific checkpoint

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
