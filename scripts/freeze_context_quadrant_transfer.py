from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.transfer_gate import fit_context_quadrant_action_ranker_transfer


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze source-only quadrant action ranker")
    parser.add_argument("--source-rollouts", type=Path, required=True)
    parser.add_argument("--frozen-gate-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()

    source_records = read_jsonl(args.source_rollouts)
    gate_model = json.loads(args.frozen_gate_model.read_text(encoding="utf-8"))
    evaluation, action_model = fit_context_quadrant_action_ranker_transfer(
        source_records,
        gate_model,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=0,
        seed=17,
    )
    report: dict[str, object] = {
        "scientific_status": "source-only frozen secondary-policy fit",
        "run": {
            "source_rollouts": str(args.source_rollouts.resolve()),
            "source_rollouts_sha256": hashlib.sha256(
                args.source_rollouts.read_bytes()
            ).hexdigest(),
            "frozen_gate_model": str(args.frozen_gate_model.resolve()),
            "frozen_gate_model_sha256": hashlib.sha256(
                args.frozen_gate_model.read_bytes()
            ).hexdigest(),
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        "evaluation": evaluation,
    }
    write_json(action_model, args.output_dir / "model.json")
    write_json(report, args.output_dir / "report.json")
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
