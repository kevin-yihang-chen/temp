"""Freeze one immutable, test-free Utility-SFT coverage-correction arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.predictability_matrix_artifacts import (
    atomic_json_write_exclusive,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if (
        config.get("schema") != "utility_sft_train_config_v1"
        or config.get("scope") != "three_domain_development_correction"
        or config.get("domain_sampling") != "uniform_domain_then_source_cycle"
        or config.get("test_authorized") is not False
        or config.get("steps") != 1024
        or config.get("train_sources_per_benchmark") != 3600
        or config.get("validation_sources_per_benchmark") != 64
    ):
        raise ValueError("invalid pre-registered coverage-correction config")
    bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    if (
        bundle.get("test_data_present") is not False
        or bundle.get("formal_test_eligible") is not False
        or bundle.get("split_audit", {}).get("passed") is not True
    ):
        raise ValueError("invalid development-only bundle")
    script_names = (
        "train_utility_sft_development.py",
        "execute_utility_sft_development.py",
        "slurm_utility_sft_correction_2gpu.sh",
        "freeze_utility_sft_correction.py",
    )
    paths = sorted((root / "src/beyond_entropy").glob("*.py")) + [
        root / "scripts" / name for name in script_names
    ]
    payload = {
        "schema": "utility_sft_development_correction_plan_v1",
        "test_authorized": False,
        "formal_claim_eligible": False,
        "correction_budget": "one pre-registered coverage correction; no loss, temperature, learning-rate, validation, or policy-lambda search",
        "method": config["method"],
        "config": {
            "path": str(Path(args.config).resolve()),
            "sha256": sha256_file(args.config),
        },
        "bundle": {
            "path": str(Path(args.bundle).resolve()),
            "sha256": sha256_file(args.bundle),
        },
        "code_hashes": {
            str(path.relative_to(root)): sha256_file(path) for path in paths
        },
        "output_root": str(Path(args.output_root).resolve()),
        "logical_gpu": "1 H800",
        "maximum_logical_gpu_hours": 1.5,
        "resource_rationale": "Three matched arms use two H800s: Format and Best-Action in parallel, then Utility on the first released GPU. A 3-GPU request previously had an approximately 23-hour queue while two H800s started immediately; the 3-hour allocation cap trades at most 6 allocated GPU-hours for much shorter queue-plus-runtime.",
    }
    atomic_json_write_exclusive(args.plan, payload)
    print(sha256_file(args.plan))


if __name__ == "__main__":
    main()
