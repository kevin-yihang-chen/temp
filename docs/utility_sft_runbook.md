# Utility-SFT development runbook

This runbook covers opened train/validation data only. It deliberately has no
test loader or test command. Run commands from the repository root with
`PYTHONPATH=src` and the pinned `qwen-vl` environment where GPU execution is
required.

## Materialized data

The six canonical JSON files are in `artifacts/utility-sft-v1/data/`:

- `chartqa-train-full.json` / `chartqa-validation-full.json`
- `docvqa-train-full.json` / `docvqa-validation-full.json`
- `hrbench-train-full.json` / `hrbench-validation-full.json`

`artifacts/utility-sft-v1/DEVELOPMENT_BUNDLE.json` binds their hashes, counts,
roles, source identities and decoded-RGB split audit. It contains no test data.
The current frozen 64-source-per-domain validation view is
`artifacts/utility-sft-v1/validation-pilot-v2/VALIDATION_FREEZE.json`.

Each sample stores:

- `inputs`: original image/question and the fixed `ANSWER, ZOOM_1..ZOOM_4`
  support;
- `labels.reward` and `labels.gain`: training/evaluation targets only;
- paired generation seed and replicate provenance;
- original sibling outcomes for audit and error analysis.

The model accepts only the typed `UtilityInputs` reconstructed from `inputs`.
Labels and outcomes cannot enter its forward API.

## Rebuild a dataset from a sealed sibling bank

For example:

```bash
PYTHONPATH=src python scripts/build_utility_sft_dataset.py \
  --completion artifacts/predictability-audit-v1/formal-development-v1/chartqa/train/complete.json \
  --output /tmp/chartqa-train.json
```

Use the analogous `complete.json` under each domain and train/validation role.
The builder rechecks sealed hashes and recomputes every stored reward with the
official benchmark scorer before writing an exclusive output.

## Training configurations

The initial 128-step development pilot uses
`configs/utility_sft_development_{format,best_action,utility}_v1.json`. The one
allowed 1024-step coverage correction uses
`configs/utility_sft_correction_{format,best_action,utility}_v1.json`. All three
arms are matched except for the objective. The correction plans are immutable
files under `artifacts/utility-sft-v1/correction-*-plan-v1.json`.

Real Qwen2.5-VL-3B runs must go through Slurm. The correction launcher is
`scripts/slurm_utility_sft_correction_2gpu.sh`; it requires the three plan paths
and plan SHA-256 values through its documented environment variables and sends
all execution-state emails to the configured address.

## Frozen validation evaluation

After all three reports and selector hashes pass audit:

```bash
PYTHONPATH=src python scripts/freeze_utility_sft_predictions.py \
  --validation-freeze artifacts/utility-sft-v1/validation-pilot-v2/VALIDATION_FREEZE.json \
  --format-report FORMAT_REPORT.json \
  --best-action-report BEST_REPORT.json \
  --utility-report UTILITY_REPORT.json \
  --output-root PREDICTION_FREEZE_DIRECTORY

PYTHONPATH=src python scripts/evaluate_utility_sft_validation_bundle.py \
  --validation-freeze artifacts/utility-sft-v1/validation-pilot-v2/VALIDATION_FREEZE.json \
  --prediction-freeze PREDICTION_FREEZE_DIRECTORY/PREDICTION_FREEZE.json \
  --output-root EVALUATION_DIRECTORY \
  --resamples 20000 --bootstrap-seed 17

PYTHONPATH=src python scripts/render_utility_sft_figures.py \
  --evaluation-bundle EVALUATION_DIRECTORY/EVALUATION_BUNDLE.json \
  --output-root FIGURE_DIRECTORY
```

The evaluation sweeps cost only in the policy layer. It does not retrain,
rescale scores with outcomes, fit thresholds, or execute candidate crops for an
SFT selector. A fresh test transaction may be created only after the frozen
validation Go conditions pass. The 2026-09-06 correction did not pass, so no test
transaction was created and this runbook must not be used to reopen model selection.
