from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from beyond_entropy.dataset import group_by_decision, read_jsonl
from beyond_entropy.metrics import bootstrap_policy_evaluation, evaluate_policy
from beyond_entropy.rescue_gate import (
    PrecomputedActionGatePolicy,
    PrecomputedRescueGatePolicy,
)
from beyond_entropy.schema import ActionRecord
from beyond_entropy.transfer_gate import (
    evaluate_frozen_composed_context_quadrant_policy,
    evaluate_frozen_factorized_context_model,
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


def read_manifest(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"manifest row {line_number} is not an object")
            state_id = str(value["state_id"])
            if state_id in result:
                raise ValueError(f"duplicate target manifest state: {state_id}")
            result[state_id] = value
    return result


def validate_confirmation_inputs(
    records: list[ActionRecord],
    manifest: dict[str, dict[str, object]],
    provenance: dict[str, object],
    *,
    target_rollouts: Path,
    expected_code_revision: str,
    expected_manifest_sha256: str,
    expected_examples: int,
) -> dict[str, object]:
    """Fail closed on incomplete or protocol-drifted confirmation inputs."""

    expected_provenance: dict[str, object] = {
        "code_revision": expected_code_revision,
        "manifest_sha256": expected_manifest_sha256,
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "scorer": "chartqa",
        "examples": expected_examples,
        "completed_examples": expected_examples,
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
    }
    mismatches = {
        key: {"expected": expected, "actual": provenance.get(key)}
        for key, expected in expected_provenance.items()
        if provenance.get(key) != expected
    }
    actual_rollout_sha256 = sha256(target_rollouts)
    if provenance.get("output_sha256") != actual_rollout_sha256:
        mismatches["output_sha256"] = {
            "expected": actual_rollout_sha256,
            "actual": provenance.get("output_sha256"),
        }
    if mismatches:
        raise ValueError(f"confirmation rollout provenance mismatch: {mismatches}")
    expected_records = expected_examples * 5
    if len(manifest) != expected_examples or len(records) != expected_records:
        raise ValueError(
            f"confirmation input is incomplete: {len(manifest)} states, {len(records)} records"
        )
    grouped = group_by_decision(records)
    if len(grouped) != len(manifest):
        raise ValueError(f"expected one decision per target state, found {len(grouped)}")
    record_states = {state_id for state_id, _ in grouped}
    if record_states != set(manifest):
        raise ValueError("confirmation rollout state IDs differ from the frozen manifest")
    expected_actions = {"answer-now", *(f"ug-grid-{index:02d}" for index in range(4))}
    for (state_id, replicate_id), siblings in grouped.items():
        row = manifest[state_id]
        if replicate_id != "replicate-000" or len(siblings) != 5:
            raise ValueError(f"unexpected replicate or sibling count for {state_id}")
        if {record.action_id for record in siblings} != expected_actions:
            raise ValueError(f"unexpected action IDs for {state_id}")
        exemplar = siblings[0]
        if (
            exemplar.generation_seed != 0
            or exemplar.image_id != str(row["image_id"])
            or exemplar.source_id != str(row["source_id"])
            or exemplar.question != str(row["question"])
            or Path(exemplar.original_image).name != Path(str(row["image_path"])).name
        ):
            raise ValueError(f"rollout content differs from target manifest for {state_id}")
    return {
        "validated_states": len(manifest),
        "validated_records": len(records),
        "validated_decisions": len(grouped),
        "target_rollouts_sha256": actual_rollout_sha256,
        "target_provenance_sha256": sha256(target_rollouts.with_suffix(".provenance.json")),
        "protocol_fields": expected_provenance,
    }


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_markdown(report: dict[str, object], *, title: str) -> str:
    evaluation = report["evaluation"]
    assert isinstance(evaluation, dict)
    policies = evaluation["policies"]
    assert isinstance(policies, dict)
    criterion = evaluation["primary_confirmation_criterion"]
    assert isinstance(criterion, dict)
    lines = [
        f"# {title}",
        "",
        f"> Primary criterion passed: **{criterion['passed']}**.",
        "> The model, scaler, regularization, and absolute threshold were frozen before target outcomes were inspected.",
        "",
        "| Policy | Accuracy gain | Tool rate | Utility [95% state-bootstrap CI] |",
        "|---|---:|---:|---:|",
    ]
    for name, result in policies.items():
        assert isinstance(result, dict)
        bootstrap = result["bootstrap"]
        assert isinstance(bootstrap, dict)
        metrics = bootstrap["metrics"]
        assert isinstance(metrics, dict)
        interval = metrics["mean_policy_utility"]
        assert isinstance(interval, dict)
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} [{:.4f}, {:.4f}] |".format(
                name,
                result["accuracy_gain"],
                result["tool_use_rate"],
                result["mean_policy_utility"],
                interval["ci_low"],
                interval["ci_high"],
            )
        )
    lines.extend(["", "## Criterion", ""])
    for name, passed in criterion.items():
        lines.append(f"- {name}: {passed}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen confirmation model")
    parser.add_argument("--target-rollouts", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-provenance", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--secondary-action-model", type=Path, required=True)
    parser.add_argument("--secondary-source-report", type=Path, required=True)
    parser.add_argument("--secondary-text-model", type=Path, required=True)
    parser.add_argument("--secondary-text-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-source-report-sha256", required=True)
    parser.add_argument("--expected-target-manifest-sha256", required=True)
    parser.add_argument("--expected-secondary-action-model-sha256", required=True)
    parser.add_argument("--expected-secondary-source-report-sha256", required=True)
    parser.add_argument("--expected-secondary-text-model-sha256", required=True)
    parser.add_argument("--expected-secondary-text-report-sha256", required=True)
    parser.add_argument("--expected-rollout-code-revision", required=True)
    parser.add_argument("--expected-examples", type=int, required=True)
    parser.add_argument(
        "--report-title",
        default="Independent ChartQA validation confirmation",
    )
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    actual_hashes = {
        "frozen_model": sha256(args.frozen_model),
        "source_report": sha256(args.source_report),
        "target_manifest": sha256(args.target_manifest),
        "secondary_action_model": sha256(args.secondary_action_model),
        "secondary_source_report": sha256(args.secondary_source_report),
        "secondary_text_model": sha256(args.secondary_text_model),
        "secondary_text_report": sha256(args.secondary_text_report),
    }
    expected_hashes = {
        "frozen_model": args.expected_model_sha256,
        "source_report": args.expected_source_report_sha256,
        "target_manifest": args.expected_target_manifest_sha256,
        "secondary_action_model": args.expected_secondary_action_model_sha256,
        "secondary_source_report": args.expected_secondary_source_report_sha256,
        "secondary_text_model": args.expected_secondary_text_model_sha256,
        "secondary_text_report": args.expected_secondary_text_report_sha256,
    }
    if actual_hashes != expected_hashes:
        raise ValueError(f"confirmation input hash mismatch: {actual_hashes}")
    model = read_json(args.frozen_model)
    secondary_action_model = read_json(args.secondary_action_model)
    secondary_text_model = read_json(args.secondary_text_model)
    source_report = read_json(args.source_report)
    source_evaluation = source_report["evaluation"]
    if not isinstance(source_evaluation, dict):
        raise ValueError("source report lacks evaluation metadata")
    source_entropy_threshold = float(source_evaluation["source_entropy_threshold"])
    target_records = read_jsonl(args.target_rollouts)
    manifest = read_manifest(args.target_manifest)
    target_provenance = read_json(args.target_provenance)
    input_validation = validate_confirmation_inputs(
        target_records,
        manifest,
        target_provenance,
        target_rollouts=args.target_rollouts,
        expected_code_revision=args.expected_rollout_code_revision,
        expected_manifest_sha256=args.expected_target_manifest_sha256,
        expected_examples=args.expected_examples,
    )
    target_strata = {
        state_id: str(row["stratum"])
        for state_id, row in manifest.items()
    }
    evaluation = evaluate_frozen_factorized_context_model(
        model,
        target_records,
        source_entropy_threshold=source_entropy_threshold,
        lambda_cost=args.lambda_cost,
        target_strata=target_strata,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    composed_result = evaluate_frozen_composed_context_quadrant_policy(
        model,
        secondary_action_model,
        target_records,
        lambda_cost=args.lambda_cost,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    policies = evaluation["policies"]
    if not isinstance(policies, dict):
        raise RuntimeError("primary evaluation lacks policy results")
    policies["frozen_factorized_context_quadrant"] = composed_result
    text_evaluation = evaluate_frozen_factorized_context_model(
        secondary_text_model,
        target_records,
        source_entropy_threshold=source_entropy_threshold,
        lambda_cost=args.lambda_cost,
        target_strata=target_strata,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    text_policies = text_evaluation["policies"]
    if not isinstance(text_policies, dict):
        raise RuntimeError("text-only evaluation lacks policy results")
    policies["frozen_factorized_context_text"] = text_policies[
        "frozen_factorized_context"
    ]
    frozen_gate_scores = score_frozen_factorized_context_model(model, target_records)
    frozen_threshold = model["threshold"]
    if not isinstance(frozen_threshold, (int, float)):
        raise ValueError("frozen gate threshold must be numeric")
    grouped_target = group_by_decision(target_records)
    fixed_policies = {}
    for action_index in range(4):
        selected_actions = {}
        for key, siblings in grouped_target.items():
            zooms = sorted(
                (record for record in siblings if record.action_type == "ZOOM"),
                key=lambda record: record.action_id,
            )
            selected_actions[key] = (
                zooms[action_index].action_id
                if frozen_gate_scores[key] >= float(frozen_threshold)
                else None
            )
        fixed_policy = PrecomputedActionGatePolicy(
            selected_actions,
            name=f"frozen_factorized_context_fixed_crop_{action_index}",
        )
        fixed_result: dict[str, object] = dict(
            evaluate_policy(target_records, fixed_policy, lambda_cost=args.lambda_cost)
        )
        fixed_result["bootstrap"] = bootstrap_policy_evaluation(
            target_records,
            fixed_policy,
            lambda_cost=args.lambda_cost,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
        policy_name = f"frozen_factorized_context_fixed_crop_{action_index}"
        policies[policy_name] = fixed_result
        fixed_policies[policy_name] = fixed_policy
    strata = evaluation["strata"]
    if not isinstance(strata, dict):
        raise RuntimeError("primary evaluation lacks stratum results")
    for stratum, stratum_results in strata.items():
        if not isinstance(stratum_results, dict):
            raise RuntimeError(f"invalid stratum result: {stratum}")
        subset = [
            record
            for record in target_records
            if target_strata[record.state_id] == stratum
        ]
        stratum_results["frozen_factorized_context_quadrant"] = (
            evaluate_frozen_composed_context_quadrant_policy(
                model,
                secondary_action_model,
                subset,
                lambda_cost=args.lambda_cost,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
        text_strata = text_evaluation["strata"]
        if not isinstance(text_strata, dict):
            raise RuntimeError("text-only evaluation lacks strata")
        text_stratum = text_strata[stratum]
        if not isinstance(text_stratum, dict):
            raise RuntimeError(f"invalid text-only stratum: {stratum}")
        stratum_results["frozen_factorized_context_text"] = text_stratum[
            "frozen_factorized_context"
        ]
        for policy_name, fixed_policy in fixed_policies.items():
            fixed_stratum_result: dict[str, object] = dict(
                evaluate_policy(subset, fixed_policy, lambda_cost=args.lambda_cost)
            )
            fixed_stratum_result["bootstrap"] = bootstrap_policy_evaluation(
                subset,
                fixed_policy,
                lambda_cost=args.lambda_cost,
                n_resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed,
            )
            stratum_results[policy_name] = fixed_stratum_result
    primary_policy = PrecomputedRescueGatePolicy(
        frozen_gate_scores,
        threshold=float(frozen_threshold),
        name="frozen_factorized_context_uniform_random_expectation",
    )
    frozen_top_actions = select_frozen_context_quadrant_actions(
        secondary_action_model,
        target_records,
    )
    secondary_policy = PrecomputedActionGatePolicy(
        {
            key: (
                action_id
                if frozen_gate_scores[key] >= float(frozen_threshold)
                else None
            )
            for key, action_id in frozen_top_actions.items()
        },
        name="frozen_factorized_context_quadrant_action",
    )
    text_gate_scores = score_frozen_factorized_context_model(
        secondary_text_model,
        target_records,
    )
    text_threshold = secondary_text_model["threshold"]
    if not isinstance(text_threshold, (int, float)):
        raise ValueError("frozen text-only gate threshold must be numeric")
    text_policy = PrecomputedRescueGatePolicy(
        text_gate_scores,
        threshold=float(text_threshold),
        name="frozen_factorized_context_text_uniform_random_expectation",
    )
    image_cluster_robustness = {
        "frozen_factorized_context": bootstrap_policy_evaluation(
            target_records,
            primary_policy,
            lambda_cost=args.lambda_cost,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
            cluster_by="image_id",
        ),
        "frozen_factorized_context_quadrant": bootstrap_policy_evaluation(
            target_records,
            secondary_policy,
            lambda_cost=args.lambda_cost,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
            cluster_by="image_id",
        ),
        "frozen_factorized_context_fixed_crop_0": bootstrap_policy_evaluation(
            target_records,
            fixed_policies["frozen_factorized_context_fixed_crop_0"],
            lambda_cost=args.lambda_cost,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
            cluster_by="image_id",
        ),
        "frozen_factorized_context_text": bootstrap_policy_evaluation(
            target_records,
            text_policy,
            lambda_cost=args.lambda_cost,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
            cluster_by="image_id",
        ),
    }
    report: dict[str, object] = {
        "scientific_status": "independent confirmation with frozen source model",
        "run": {
            "target_rollouts": str(args.target_rollouts.resolve()),
            "target_rollouts_sha256": sha256(args.target_rollouts),
            "target_provenance": str(args.target_provenance.resolve()),
            "target_provenance_sha256": sha256(args.target_provenance),
            "target_manifest": str(args.target_manifest.resolve()),
            "target_manifest_sha256": actual_hashes["target_manifest"],
            "frozen_model": str(args.frozen_model.resolve()),
            "frozen_model_sha256": actual_hashes["frozen_model"],
            "source_report": str(args.source_report.resolve()),
            "source_report_sha256": actual_hashes["source_report"],
            "secondary_action_model": str(args.secondary_action_model.resolve()),
            "secondary_action_model_sha256": actual_hashes["secondary_action_model"],
            "secondary_source_report": str(args.secondary_source_report.resolve()),
            "secondary_source_report_sha256": actual_hashes["secondary_source_report"],
            "secondary_text_model": str(args.secondary_text_model.resolve()),
            "secondary_text_model_sha256": actual_hashes["secondary_text_model"],
            "secondary_text_report": str(args.secondary_text_report.resolve()),
            "secondary_text_report_sha256": actual_hashes["secondary_text_report"],
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "lambda_cost": args.lambda_cost,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "input_validation": input_validation,
        "image_cluster_bootstrap_robustness": image_cluster_robustness,
        "evaluation": evaluation,
    }
    json_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    write_json(report, json_path)
    markdown_path.write_text(
        build_markdown(report, title=args.report_title),
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
