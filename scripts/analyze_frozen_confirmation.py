from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from beyond_entropy.dataset import group_by_decision, read_jsonl
from beyond_entropy.schema import ActionRecord
from beyond_entropy.transfer_gate import evaluate_frozen_factorized_context_model


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
) -> dict[str, object]:
    """Fail closed on incomplete or protocol-drifted confirmation inputs."""

    expected_provenance: dict[str, object] = {
        "code_revision": "ff193264c295dca977c653b6d999290786bf3bba",
        "manifest_sha256": "d3178218853b10447228963e839716f0eac768b51bdc0f5b4a83268d3819b58b",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "scorer": "chartqa",
        "examples": 1918,
        "completed_examples": 1918,
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
    if len(manifest) != 1918 or len(records) != 9590:
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


def build_markdown(report: dict[str, object]) -> str:
    evaluation = report["evaluation"]
    assert isinstance(evaluation, dict)
    policies = evaluation["policies"]
    assert isinstance(policies, dict)
    criterion = evaluation["primary_confirmation_criterion"]
    assert isinstance(criterion, dict)
    lines = [
        "# Independent ChartQA validation confirmation",
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-source-report-sha256", required=True)
    parser.add_argument("--expected-target-manifest-sha256", required=True)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    actual_hashes = {
        "frozen_model": sha256(args.frozen_model),
        "source_report": sha256(args.source_report),
        "target_manifest": sha256(args.target_manifest),
    }
    expected_hashes = {
        "frozen_model": args.expected_model_sha256,
        "source_report": args.expected_source_report_sha256,
        "target_manifest": args.expected_target_manifest_sha256,
    }
    if actual_hashes != expected_hashes:
        raise ValueError(f"confirmation input hash mismatch: {actual_hashes}")
    model = read_json(args.frozen_model)
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
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "lambda_cost": args.lambda_cost,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "input_validation": input_validation,
        "evaluation": evaluation,
    }
    json_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    write_json(report, json_path)
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
