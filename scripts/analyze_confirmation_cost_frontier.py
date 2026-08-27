from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from beyond_entropy.dataset import group_by_decision, read_jsonl
from beyond_entropy.metrics import bootstrap_policy_evaluation, evaluate_policy
from beyond_entropy.rescue_gate import PrecomputedRescueGatePolicy
from beyond_entropy.transfer_gate import score_frozen_factorized_context_model


EXPECTED_ROLLOUTS_SHA256 = (
    "a0d11b785ee6683dc34277740e3abfcd7d84323a740d88da5ef68ddb2eb98257"
)
EXPECTED_MODEL_SHA256 = (
    "5d5c0f781a7141726e786d6ad87b861a6395c489bcf9ad8567a2e9ca825c3330"
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
        description="Post-hoc cost frontier for the frozen confirmation gate"
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lambda-costs",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125],
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()

    actual_hashes = {
        "rollouts": sha256(args.rollouts),
        "frozen_model": sha256(args.frozen_model),
    }
    expected_hashes = {
        "rollouts": EXPECTED_ROLLOUTS_SHA256,
        "frozen_model": EXPECTED_MODEL_SHA256,
    }
    if actual_hashes != expected_hashes:
        raise ValueError(f"confirmation cost-frontier hash mismatch: {actual_hashes}")
    records = read_jsonl(args.rollouts)
    if len(records) != 9590 or len(group_by_decision(records)) != 1918:
        raise ValueError("confirmation cost-frontier input is incomplete")
    model = read_json(args.frozen_model)
    threshold = model["threshold"]
    if not isinstance(threshold, (int, float)):
        raise ValueError("frozen stopping threshold must be numeric")
    policy = PrecomputedRescueGatePolicy(
        score_frozen_factorized_context_model(model, records),
        threshold=float(threshold),
        name="frozen_factorized_context_uniform_random_expectation",
    )

    frontier = []
    for index, lambda_cost in enumerate(args.lambda_costs):
        if lambda_cost < 0.0:
            raise ValueError("lambda costs must be non-negative")
        point: dict[str, object] = dict(
            evaluate_policy(records, policy, lambda_cost=lambda_cost)
        )
        state_bootstrap = bootstrap_policy_evaluation(
            records,
            policy,
            lambda_cost=lambda_cost,
            n_resamples=args.bootstrap_resamples,
            seed=index,
            cluster_by="state_id",
        )
        image_bootstrap = bootstrap_policy_evaluation(
            records,
            policy,
            lambda_cost=lambda_cost,
            n_resamples=args.bootstrap_resamples,
            seed=index,
            cluster_by="image_id",
        )
        frontier.append(
            {
                "lambda_cost": lambda_cost,
                "point": point,
                "state_bootstrap": state_bootstrap,
                "image_bootstrap": image_bootstrap,
                "registered_primary_cost": abs(lambda_cost - 0.05) < 1e-12,
            }
        )
    reference = frontier[0]["point"]
    if not isinstance(reference, dict):
        raise RuntimeError("cost-frontier reference point is malformed")
    accuracy_gain = reference["accuracy_gain"]
    tool_rate = reference["tool_use_rate"]
    if not isinstance(accuracy_gain, (int, float)) or not isinstance(
        tool_rate, (int, float)
    ):
        raise RuntimeError("cost-frontier reference metrics are not numeric")
    break_even_cost = (
        float(accuracy_gain) / float(tool_rate) if float(tool_rate) > 0.0 else None
    )
    report: dict[str, object] = {
        "scientific_status": (
            "post-hoc cost sensitivity for a byte-identical frozen policy; "
            "lambda=0.05 remains the failed registered primary"
        ),
        "run": {
            "rollouts": str(args.rollouts.resolve()),
            "rollouts_sha256": actual_hashes["rollouts"],
            "frozen_model": str(args.frozen_model.resolve()),
            "frozen_model_sha256": actual_hashes["frozen_model"],
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "bootstrap_resamples": args.bootstrap_resamples,
            "lambda_costs": args.lambda_costs,
        },
        "n_decisions": 1918,
        "frozen_threshold": float(threshold),
        "point_estimate_break_even_cost": break_even_cost,
        "frontier": frontier,
    }
    write_json(report, args.output_dir / "report.json")
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
