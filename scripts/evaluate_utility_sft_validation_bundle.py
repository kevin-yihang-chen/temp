"""Run the frozen eight-policy evaluation on all three validation domains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.predictability_matrix_artifacts import (
    atomic_json_write_exclusive,
    sha256_file,
)
try:
    from scripts.evaluate_utility_sft import evaluate
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from evaluate_utility_sft import evaluate


BENCHMARKS = ("chartqa", "docvqa", "hrbench")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def evaluate_bundle(
    *,
    validation_freeze: str | Path,
    prediction_freeze: str | Path,
    output_root: str | Path,
    resamples: int = 20_000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    validation_path = Path(validation_freeze).resolve()
    prediction_path = Path(prediction_freeze).resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if (
        validation.get("schema") != "utility_sft_validation_freeze_v1"
        or predictions.get("schema") != "utility_sft_prediction_freeze_v1"
        or validation.get("test_data_present") is not False
        or predictions.get("test_data_present") is not False
        or predictions.get("validation_freeze_sha256") != sha256_file(validation_path)
        or Path(str(predictions.get("validation_freeze"))).resolve() != validation_path
    ):
        raise ValueError("validation and prediction freezes are not bound")
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evaluation bundle: {destination}")
    stage = destination.with_name(destination.name + ".staging")
    if stage.exists():
        raise FileExistsError(f"evaluation staging output exists: {stage}")
    stage.mkdir(parents=True)
    inventory: dict[str, Any] = {}
    for index, benchmark in enumerate(BENCHMARKS):
        data = _mapping(validation["inventory"][benchmark], benchmark)
        prediction = _mapping(predictions["inventory"][benchmark], benchmark)
        if prediction.get("dataset_sha256") != data.get("dataset_sha256"):
            raise ValueError(f"{benchmark} prediction/dataset identity mismatch")
        report_path = stage / benchmark / "evaluation.json"
        report = evaluate(
            str(data["dataset"]),
            str(prediction["predictions"]),
            str(data["frozen_voi_decisions"]),
            str(report_path),
            resamples=resamples,
            bootstrap_seed=bootstrap_seed + index * 10_000,
        )
        inventory[benchmark] = {
            "evaluation": str(destination / benchmark / "evaluation.json"),
            "evaluation_sha256": sha256_file(report_path),
            "states": data["states"],
            "sources": data["sources"],
            "primary_lambda": report["primary_lambda"],
        }
    bundle = {
        "schema": "utility_sft_validation_evaluation_bundle_v1",
        "role": "validation",
        "formal_claim_eligible": False,
        "test_data_present": False,
        "validation_freeze": str(validation_path),
        "validation_freeze_sha256": sha256_file(validation_path),
        "prediction_freeze": str(prediction_path),
        "prediction_freeze_sha256": sha256_file(prediction_path),
        "bootstrap": {
            "resamples": resamples,
            "base_seed": bootstrap_seed,
            "domain_seed_stride": 10_000,
            "unit": "source_id",
        },
        "inventory": inventory,
    }
    atomic_json_write_exclusive(stage / "EVALUATION_BUNDLE.json", bundle)
    stage.replace(destination)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-freeze", required=True)
    parser.add_argument("--prediction-freeze", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    args = parser.parse_args()
    bundle = evaluate_bundle(
        validation_freeze=args.validation_freeze,
        prediction_freeze=args.prediction_freeze,
        output_root=args.output_root,
        resamples=args.resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output_root).resolve()),
                "sha256": sha256_file(Path(args.output_root) / "EVALUATION_BUNDLE.json"),
                "inventory": bundle["inventory"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
