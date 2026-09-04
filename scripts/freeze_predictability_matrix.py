from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Sequence

from beyond_entropy.predictability_audit import (
    AUDIT_SEEDS,
    PREDICTOR_LEVELS,
    TARGET_FAMILIES,
)
from beyond_entropy.predictability_matrix import (
    STRONG_BASELINE_RANDOM_SEED,
    fit_predictability_matrix,
    save_frozen_predictability_matrix,
)
from beyond_entropy.predictability_matrix_artifacts import (
    load_development_input_spec,
)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit all predictability choices on train/validation and persist a "
            "test-free frozen bundle"
        )
    )
    parser.add_argument("--input-spec", required=True)
    parser.add_argument("--input-spec-sha256", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args(argv)

    datasets, protocol, provenance = load_development_input_spec(
        args.input_spec,
        expected_sha256=args.input_spec_sha256,
        repo_root=args.repo_root,
    )
    fixed_tool = protocol["fixed_visual_tool"]
    training = protocol["training"]
    if (
        tuple(training["seeds"]) != AUDIT_SEEDS
        or int(training["maximum_seeds"]) != len(AUDIT_SEEDS)
        or tuple(protocol["predictor_ladder"]) != PREDICTOR_LEVELS
        or tuple(protocol["target_families"]) != TARGET_FAMILIES
        or int(
            protocol["strong_baseline_implementation"]["random_gate_fixed_visual_tool"][
                "seed"
            ]
        )
        != STRONG_BASELINE_RANDOM_SEED
    ):
        raise ValueError("protocol differs from the complete formal matrix contract")
    provenance["packages"] = {
        "numpy": _version("numpy"),
        "scikit-learn": _version("scikit-learn"),
        "torch": _version("torch"),
    }
    frozen = fit_predictability_matrix(
        datasets,
        lambda_cost=float(fixed_tool["lambda_cost"]),
        formal_claim_eligible=True,
        provenance=provenance,
    )
    report = save_frozen_predictability_matrix(
        frozen,
        model_path=Path(args.model_output),
        report_path=Path(args.report_output),
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "formal_claim_eligible": report["formal_claim_eligible"],
                "test_data_present": report["test_data_present"],
                "model_path": report["model_path"],
                "model_sha256": report["model_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
