from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from beyond_entropy.dataset import group_by_decision, read_jsonl
from beyond_entropy.metrics import paired_bootstrap_policy_difference
from beyond_entropy.policies import Policy
from beyond_entropy.rescue_gate import (
    PrecomputedActionGatePolicy,
    PrecomputedRescueGatePolicy,
)
from beyond_entropy.transfer_gate import (
    score_frozen_factorized_context_model,
    select_frozen_context_quadrant_actions,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc paired contrasts for the frozen ChartQA confirmation"
    )
    parser.add_argument("--target-rollouts", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--secondary-action-model", type=Path, required=True)
    parser.add_argument("--secondary-text-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()

    expected_hashes = {
        "target_rollouts": "a0d11b785ee6683dc34277740e3abfcd7d84323a740d88da5ef68ddb2eb98257",
        "frozen_model": "5d5c0f781a7141726e786d6ad87b861a6395c489bcf9ad8567a2e9ca825c3330",
        "secondary_action_model": "5989974482785b31868473e7a925708d15f6f1fbac3095906ded7a88def53bbd",
        "secondary_text_model": "175c044ceca6b755b8cc16b3f106604cfc1b54396b695bf9c685a81ffd162fa5",
    }
    paths = {
        "target_rollouts": args.target_rollouts,
        "frozen_model": args.frozen_model,
        "secondary_action_model": args.secondary_action_model,
        "secondary_text_model": args.secondary_text_model,
    }
    actual_hashes = {name: sha256(path) for name, path in paths.items()}
    if actual_hashes != expected_hashes:
        raise ValueError(f"confirmation contrast input hash mismatch: {actual_hashes}")

    records = read_jsonl(args.target_rollouts)
    grouped = group_by_decision(records)
    if len(records) != 9590 or len(grouped) != 1918:
        raise ValueError(
            f"expected 9590 records and 1918 decisions, got {len(records)} and {len(grouped)}"
        )
    frozen_model = read_json(args.frozen_model)
    action_model = read_json(args.secondary_action_model)
    text_model = read_json(args.secondary_text_model)

    gate_scores = score_frozen_factorized_context_model(frozen_model, records)
    gate_threshold = frozen_model["threshold"]
    if not isinstance(gate_threshold, (int, float)):
        raise ValueError("frozen gate threshold must be numeric")
    random_policy = PrecomputedRescueGatePolicy(
        gate_scores,
        threshold=float(gate_threshold),
        name="frozen_factorized_context_uniform_random_expectation",
    )
    fixed_zero_policy = PrecomputedActionGatePolicy(
        {
            key: (
                sorted(
                    (record for record in siblings if record.action_type == "ZOOM"),
                    key=lambda record: record.action_id,
                )[0].action_id
                if gate_scores[key] >= float(gate_threshold)
                else None
            )
            for key, siblings in grouped.items()
        },
        name="frozen_factorized_context_fixed_crop_0",
    )
    ranked_actions = select_frozen_context_quadrant_actions(action_model, records)
    ranked_policy = PrecomputedActionGatePolicy(
        {
            key: action_id if gate_scores[key] >= float(gate_threshold) else None
            for key, action_id in ranked_actions.items()
        },
        name="frozen_factorized_context_quadrant",
    )
    text_scores = score_frozen_factorized_context_model(text_model, records)
    text_threshold = text_model["threshold"]
    if not isinstance(text_threshold, (int, float)):
        raise ValueError("frozen text gate threshold must be numeric")
    text_policy = PrecomputedRescueGatePolicy(
        text_scores,
        threshold=float(text_threshold),
        name="frozen_factorized_context_text_uniform_random_expectation",
    )

    contrasts: dict[str, tuple[Policy, Policy]] = {
        "fixed_crop_0_minus_uniform_random": (random_policy, fixed_zero_policy),
        "ranked_quadrant_minus_uniform_random": (random_policy, ranked_policy),
        "fixed_crop_0_minus_ranked_quadrant": (ranked_policy, fixed_zero_policy),
        "text_gate_minus_full_context_gate": (random_policy, text_policy),
    }
    paired: dict[str, dict[str, object]] = {}
    for index, (name, (left_policy, right_policy)) in enumerate(contrasts.items()):
        paired[name] = {
            "state_cluster": paired_bootstrap_policy_difference(
                records,
                left_policy,
                records,
                right_policy,
                lambda_cost=0.05,
                n_resamples=args.bootstrap_resamples,
                seed=index,
                cluster_by="state_id",
            ),
            "image_cluster": paired_bootstrap_policy_difference(
                records,
                left_policy,
                records,
                right_policy,
                lambda_cost=0.05,
                n_resamples=args.bootstrap_resamples,
                seed=index,
                cluster_by="image_id",
            ),
        }

    report: dict[str, object] = {
        "scientific_status": (
            "post-hoc paired action and feature contrasts after the frozen primary "
            "confirmation; these results cannot alter the primary criterion"
        ),
        "run": {
            **{
                name: str(path.resolve())
                for name, path in paths.items()
            },
            **{
                f"{name}_sha256": digest
                for name, digest in actual_hashes.items()
            },
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "bootstrap_resamples": args.bootstrap_resamples,
            "lambda_cost": 0.05,
        },
        "n_decisions": len(grouped),
        "paired_contrasts": paired,
    }
    write_json(report, args.output_dir / "report.json")
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
