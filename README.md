# Beyond Entropy

Research scaffold for learning the **pre-action, task-relevant value of visual
information** from counterfactual sibling rollouts.

The central target is:

```text
Q_voi(s, a) = P(success after a | s) - P(success now | s) - lambda * cost(a)
```

The current milestone implements Stage 0/1 infrastructure: a strict rollout data
contract, a real-backend collector interface, SCGR diagnostics, a dependency-free
ridge value model, adaptive stopping, policy baselines, and an end-to-end synthetic
control experiment. It has **not** run Qwen2.5-VL, public benchmarks, or RL yet.

## Quick start

No runtime packages outside the Python standard library are required.

```bash
cd /userhome/cs3/yihangc/Documents/beyond-entropy
PYTHONPATH=src python3 -m beyond_entropy demo --output-dir artifacts/demo
pytest
```

The demo produces separate train/test sibling rollouts, a serialized value model,
JSON metrics, and a Markdown report under `artifacts/demo/`.

Individual commands:

```bash
PYTHONPATH=src python3 -m beyond_entropy simulate \
  --output artifacts/counterfactual.jsonl --n-states 600 --num-candidates 4

PYTHONPATH=src python3 -m beyond_entropy diagnose \
  --data artifacts/counterfactual.jsonl

PYTHONPATH=src python3 -m beyond_entropy train \
  --data artifacts/demo/train.jsonl --output artifacts/demo/value_model.json

PYTHONPATH=src python3 -m beyond_entropy evaluate \
  --data artifacts/demo/test.jsonl --model artifacts/demo/value_model.json
```

## Project map

```text
src/beyond_entropy/
  rollout.py     model/backend-independent sibling collector
  schema.py      validated action-rollout record
  dataset.py     JSONL IO, sibling validation, state-level split
  features.py    pre-action-only feature encoding and leakage guards
  model.py       lightweight ridge VOI model
  policies.py    answer/random/fixed/entropy/learned/oracle policies
  metrics.py     SCGR and agent efficiency/stopping metrics
  simulate.py    controlled synthetic pipeline test
  cli.py         experiment commands
docs/
  data_contract.md
  reference_integration.md
  research_plan.md
```

The reference-integration design is based on the official
[UG framework](https://github.com/ExplainableML/ug-framework) and
[VTool-R1](https://github.com/VTOOL-R1/vtool-r1) repositories. Their code is not
vendored here. See `docs/reference_integration.md` for the exact adapter boundary
and `docs/research_plan.md` for experiment gates.

## Scientific guardrails

- Synthetic demo numbers validate code paths only.
- Split by state/image, never by sibling action row.
- `entropy_after`, answers, correctness, labels, and rewards cannot be model
  inputs.
- Entropy search pays for executing every candidate it scores.
- Oracle VOI is a diagnostic upper bound, not a deployable baseline.
- Pin upstream commits, model revisions, prompts, dataset revisions, and seeds
  before real experiments.
