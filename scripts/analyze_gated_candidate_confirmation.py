from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from beyond_entropy.candidate_ablation import compare_candidate_sets
from beyond_entropy.dataset import group_by_decision, read_jsonl
from beyond_entropy.metrics import (
    bootstrap_policy_evaluation,
    paired_bootstrap_policy_difference,
)
from beyond_entropy.rescue_gate import PrecomputedRescueGatePolicy
from beyond_entropy.schema import ActionRecord
from beyond_entropy.transfer_gate import score_frozen_factorized_context_model


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_manifest_states(path: Path) -> set[str]:
    states: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict) or "state_id" not in value:
                raise ValueError(f"invalid manifest row {line_number}")
            state_id = str(value["state_id"])
            if state_id in states:
                raise ValueError(f"duplicate manifest state: {state_id}")
            states.add(state_id)
    return states


def validate_provenance(
    provenance: dict[str, object],
    rollouts: Path,
    *,
    manifest_sha256: str,
    code_revision: str,
    examples: int,
    proposer: str | None,
) -> dict[str, object]:
    expected: dict[str, object] = {
        "code_revision": code_revision,
        "manifest_sha256": manifest_sha256,
        "output_sha256": sha256(rollouts),
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "scorer": "chartqa",
        "examples": examples,
        "completed_examples": examples,
        "candidate_count": 4,
        "generation_seeds": [0],
        "max_new_tokens": 16,
        "min_pixels": 200704,
        "max_pixels": 602112,
        "attention_implementation": "sdpa",
        "system_prompt": (
            "Answer with only the final answer: a single number, word, or short phrase. "
            "Do not explain."
        ),
        "local_files_only": True,
        "proposer": proposer,
    }
    mismatches = {
        name: {"expected": expected_value, "actual": provenance.get(name)}
        for name, expected_value in expected.items()
        if provenance.get(name) != expected_value
    }
    if mismatches:
        raise ValueError(f"candidate confirmation provenance mismatch: {mismatches}")
    return expected


def primary_uniform_row(report: dict[str, object]) -> dict[str, object]:
    differences = report["policy_differences"]
    if not isinstance(differences, list):
        raise ValueError("candidate comparison lacks policy differences")
    matches = [
        row
        for row in differences
        if isinstance(row, dict)
        and row.get("right_policy") == "uniform_random_zoom_expectation"
    ]
    if len(matches) != 1:
        raise ValueError("candidate comparison lacks a unique random-crop row")
    return matches[0]


def utility_interval(result: dict[str, object]) -> dict[str, object]:
    metrics = result["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("bootstrap result lacks metrics")
    utility = metrics["mean_policy_utility"]
    if not isinstance(utility, dict):
        raise ValueError("bootstrap result lacks utility interval")
    return utility


def numeric_value(values: dict[str, object], name: str) -> float:
    value = values[name]
    if not isinstance(value, (int, float)):
        raise ValueError(f"expected numeric {name}, found {value!r}")
    return float(value)


def validate_records(
    records: list[ActionRecord],
    manifest_states: set[str],
    *,
    examples: int,
) -> None:
    if len(records) != examples * 5:
        raise ValueError(f"expected {examples * 5} action records, found {len(records)}")
    grouped = group_by_decision(records)
    if len(grouped) != examples:
        raise ValueError(f"expected {examples} decisions, found {len(grouped)}")
    if {key[0] for key in grouped} != manifest_states:
        raise ValueError("candidate rollout states differ from the frozen manifest")


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm chart-layout proposals with a frozen stopping gate"
    )
    parser.add_argument("--baseline-rollouts", type=Path, required=True)
    parser.add_argument("--baseline-provenance", type=Path, required=True)
    parser.add_argument("--treatment-rollouts", type=Path, required=True)
    parser.add_argument("--treatment-provenance", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-baseline-code-revision", required=True)
    parser.add_argument("--expected-treatment-code-revision", required=True)
    parser.add_argument("--expected-examples", type=int, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()

    actual_manifest_sha256 = sha256(args.target_manifest)
    actual_model_sha256 = sha256(args.frozen_model)
    if (
        actual_manifest_sha256 != args.expected_manifest_sha256
        or actual_model_sha256 != args.expected_model_sha256
    ):
        raise ValueError("candidate confirmation manifest or model hash mismatch")
    baseline_provenance = read_json(args.baseline_provenance)
    treatment_provenance = read_json(args.treatment_provenance)
    baseline_protocol = validate_provenance(
        baseline_provenance,
        args.baseline_rollouts,
        manifest_sha256=actual_manifest_sha256,
        code_revision=args.expected_baseline_code_revision,
        examples=args.expected_examples,
        proposer=None,
    )
    treatment_protocol = validate_provenance(
        treatment_provenance,
        args.treatment_rollouts,
        manifest_sha256=actual_manifest_sha256,
        code_revision=args.expected_treatment_code_revision,
        examples=args.expected_examples,
        proposer="chart-layout",
    )

    manifest_states = read_manifest_states(args.target_manifest)
    if len(manifest_states) != args.expected_examples:
        raise ValueError("frozen manifest has the wrong number of states")
    baseline = read_jsonl(args.baseline_rollouts)
    treatment = read_jsonl(args.treatment_rollouts)
    validate_records(baseline, manifest_states, examples=args.expected_examples)
    validate_records(treatment, manifest_states, examples=args.expected_examples)

    unconditional_state = compare_candidate_sets(
        baseline,
        treatment,
        lambda_cost=0.05,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=0,
        cluster_by="state_id",
    )
    unconditional_image = compare_candidate_sets(
        baseline,
        treatment,
        lambda_cost=0.05,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=0,
        cluster_by="image_id",
    )

    model = read_json(args.frozen_model)
    scores = score_frozen_factorized_context_model(model, baseline)
    threshold = model["threshold"]
    if not isinstance(threshold, (int, float)):
        raise ValueError("frozen stopping threshold must be numeric")
    gate_policy = PrecomputedRescueGatePolicy(
        scores,
        threshold=float(threshold),
        name="frozen_factorized_context_uniform_random_expectation",
    )
    gated_state = paired_bootstrap_policy_difference(
        baseline,
        gate_policy,
        treatment,
        gate_policy,
        lambda_cost=0.05,
        n_resamples=args.bootstrap_resamples,
        seed=0,
        cluster_by="state_id",
    )
    gated_image = paired_bootstrap_policy_difference(
        baseline,
        gate_policy,
        treatment,
        gate_policy,
        lambda_cost=0.05,
        n_resamples=args.bootstrap_resamples,
        seed=0,
        cluster_by="image_id",
    )
    treatment_state = bootstrap_policy_evaluation(
        treatment,
        gate_policy,
        lambda_cost=0.05,
        n_resamples=args.bootstrap_resamples,
        seed=0,
        cluster_by="state_id",
    )
    treatment_image = bootstrap_policy_evaluation(
        treatment,
        gate_policy,
        lambda_cost=0.05,
        n_resamples=args.bootstrap_resamples,
        seed=0,
        cluster_by="image_id",
    )

    proposal_state = primary_uniform_row(unconditional_state)
    proposal_image = primary_uniform_row(unconditional_image)
    proposal_state_utility = utility_interval(proposal_state)
    proposal_image_utility = utility_interval(proposal_image)
    proposal_criterion = {
        "positive_point_estimate": numeric_value(proposal_state_utility, "estimate")
        > 0.0,
        "state_ci_lower_above_zero": numeric_value(proposal_state_utility, "ci_low")
        > 0.0,
        "image_ci_lower_above_zero": numeric_value(proposal_image_utility, "ci_low")
        > 0.0,
    }
    proposal_criterion["passed"] = all(proposal_criterion.values())

    gated_state_utility = utility_interval(gated_state)
    gated_image_utility = utility_interval(gated_image)
    treatment_state_utility = utility_interval(treatment_state)
    treatment_image_utility = utility_interval(treatment_image)
    composed_criterion = {
        "positive_paired_point_estimate": numeric_value(
            gated_state_utility, "estimate"
        )
        > 0.0,
        "paired_state_ci_lower_above_zero": numeric_value(
            gated_state_utility, "ci_low"
        )
        > 0.0,
        "paired_image_ci_lower_above_zero": numeric_value(
            gated_image_utility, "ci_low"
        )
        > 0.0,
        "positive_absolute_utility": numeric_value(
            treatment_state_utility, "estimate"
        )
        > 0.0,
        "absolute_state_ci_lower_above_zero": numeric_value(
            treatment_state_utility, "ci_low"
        )
        > 0.0,
        "absolute_image_ci_lower_above_zero": numeric_value(
            treatment_image_utility, "ci_low"
        )
        > 0.0,
    }
    composed_criterion["passed"] = all(composed_criterion.values())

    report: dict[str, object] = {
        "scientific_status": (
            "conditionally launched independent chart-layout and composed-policy confirmation"
        ),
        "run": {
            "baseline_rollouts": str(args.baseline_rollouts.resolve()),
            "baseline_rollouts_sha256": sha256(args.baseline_rollouts),
            "baseline_provenance_sha256": sha256(args.baseline_provenance),
            "treatment_rollouts": str(args.treatment_rollouts.resolve()),
            "treatment_rollouts_sha256": sha256(args.treatment_rollouts),
            "treatment_provenance_sha256": sha256(args.treatment_provenance),
            "target_manifest_sha256": actual_manifest_sha256,
            "frozen_model_sha256": actual_model_sha256,
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "bootstrap_resamples": args.bootstrap_resamples,
            "lambda_cost": 0.05,
        },
        "input_validation": {
            "states": len(manifest_states),
            "baseline_records": len(baseline),
            "treatment_records": len(treatment),
            "baseline_protocol": baseline_protocol,
            "treatment_protocol": treatment_protocol,
        },
        "proposal_confirmation_criterion": proposal_criterion,
        "composed_policy_confirmation_criterion": composed_criterion,
        "unconditional_state_comparison": unconditional_state,
        "unconditional_image_comparison": unconditional_image,
        "gated_treatment_minus_baseline_state": gated_state,
        "gated_treatment_minus_baseline_image": gated_image,
        "composed_treatment_state_bootstrap": treatment_state,
        "composed_treatment_image_bootstrap": treatment_image,
    }
    write_json(report, args.output_dir / "report.json")
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
