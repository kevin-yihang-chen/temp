"""Freeze one test-free semantic-ablation execution plan."""
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
    parser.add_argument("--report", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--validation-freeze", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    validation = json.loads(Path(args.validation_freeze).read_text(encoding="utf-8"))
    selector_sha = sha256_file(args.selector)
    if (
        config.get("method") != "utility"
        or config.get("test_authorized") is not False
        or report.get("method") != "utility"
        or report.get("test_accessed") is not False
        or report.get("selector_sha256") != selector_sha
        or validation.get("schema") != "utility_sft_validation_freeze_v1"
        or validation.get("test_data_present") is not False
    ):
        raise ValueError("invalid development-only ablation inputs")
    script_names = (
        "run_utility_sft_ablation.py", "execute_utility_sft_ablation.py",
        "slurm_utility_sft_ablation.sh", "freeze_utility_sft_ablation.py",
        "render_utility_sft_figures.py",
    )
    paths = sorted((root / "src/beyond_entropy").glob("*.py")) + [
        root / "scripts" / name for name in script_names
    ]
    payload = {
        "schema": "utility_sft_semantic_ablation_plan_v1",
        "test_authorized": False,
        "formal_claim_eligible": False,
        "config": {"path": str(Path(args.config).resolve()), "sha256": sha256_file(args.config)},
        "report": {"path": str(Path(args.report).resolve()), "sha256": sha256_file(args.report)},
        "selector": {"path": str(Path(args.selector).resolve()), "sha256": selector_sha},
        "validation_freeze": {
            "path": str(Path(args.validation_freeze).resolve()),
            "sha256": sha256_file(args.validation_freeze),
        },
        "code_hashes": {str(path.relative_to(root)): sha256_file(path) for path in paths},
        "output_root": str(Path(args.output_root).resolve()),
        "conditions": ["original", "question_shuffle", "image_shuffle", "region_ablation"],
        "resources": {"gpu": "1 H800", "maximum_gpu_hours": .5},
    }
    atomic_json_write_exclusive(args.plan, payload)
    print(sha256_file(args.plan))


if __name__ == "__main__":
    main()
