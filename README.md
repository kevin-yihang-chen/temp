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
a benchmark result. Public V*Bench/ChartQA slices and RL remain future gates.

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
rollouts plus `.diagnostic.json` and `.provenance.json` sidecars. To validate the
locally cached 7B model on one synthetic image through Slurm:

```bash
scripts/submit_qwen_smoke.sh
```

The submit wrapper reads the notification recipient from the private,
git-ignored `.slurm-notify-email` file and requests email for all Slurm state
changes. This keeps contact information out of the public repository.

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

`roi_pool_spatial_tokens` extracts all candidate representations from one cached
full-image spatial token grid. A Qwen/lmms-eval feature adapter is still required
before this becomes a real frozen-VLM experiment.

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
