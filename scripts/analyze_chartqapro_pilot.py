from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from beyond_entropy.chartqapro import chartqapro_match, chartqapro_spec_match
from beyond_entropy.dataset import group_by_decision, read_jsonl
from beyond_entropy.metrics import bootstrap_policy_evaluation
from beyond_entropy.rescue_gate import PrecomputedRescueGatePolicy
from beyond_entropy.rollout import GroundTruth
from beyond_entropy.schema import ActionRecord
from beyond_entropy.transfer_gate import (
    evaluate_frozen_factorized_context_model,
    score_frozen_factorized_context_model,
)


EXPECTED_MODEL_SHA256 = (
    "5d5c0f781a7141726e786d6ad87b861a6395c489bcf9ad8567a2e9ca825c3330"
)
EXPECTED_SOURCE_REPORT_SHA256 = (
    "1f05ddeef52fa9abced549479cdb8fa386578d12600fb874a964a12a4d927462"
)
EXPECTED_MANIFEST_SHA256 = (
    "e02f62ae794125c5e4565493e54b72855b1db96542257c57a15049de65f6a722"
)
EXPECTED_ROLLOUT_CODE_REVISION = "a5778dbf64583ac8177edb36b67cb61b8f901b4d"
EXPECTED_STATES = 309
EXPECTED_RECORDS = EXPECTED_STATES * 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"manifest row {line_number} is not an object")
            state_id = str(value["state_id"])
            if state_id in rows:
                raise ValueError(f"duplicate manifest state: {state_id}")
            rows[state_id] = value
    return rows


def _validate_inputs(
    records: Sequence[ActionRecord],
    manifest: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
    *,
    rollouts: Path,
) -> dict[str, Any]:
    expected_provenance = {
        "code_revision": EXPECTED_ROLLOUT_CODE_REVISION,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "scorer": "chartqapro",
        "examples": EXPECTED_STATES,
        "completed_examples": EXPECTED_STATES,
        "candidate_count": 4,
        "generation_seeds": [0],
        "max_new_tokens": 16,
        "min_pixels": 200704,
        "max_pixels": 602112,
        "attention_implementation": "sdpa",
        "system_prompt": "You are a helpful assistant.",
        "local_files_only": True,
    }
    mismatches = {
        name: {"expected": expected, "actual": provenance.get(name)}
        for name, expected in expected_provenance.items()
        if provenance.get(name) != expected
    }
    rollout_hash = _sha256(rollouts)
    if provenance.get("output_sha256") != rollout_hash:
        mismatches["output_sha256"] = {
            "expected": rollout_hash,
            "actual": provenance.get("output_sha256"),
        }
    if mismatches:
        raise ValueError(f"pilot provenance mismatch: {mismatches}")
    if len(manifest) != EXPECTED_STATES or len(records) != EXPECTED_RECORDS:
        raise ValueError(
            f"incomplete pilot: {len(manifest)} states and {len(records)} records"
        )
    grouped = group_by_decision(records)
    if len(grouped) != EXPECTED_STATES:
        raise ValueError(f"expected {EXPECTED_STATES} decisions, found {len(grouped)}")
    if {state_id for state_id, _ in grouped} != set(manifest):
        raise ValueError("rollout and manifest state IDs differ")
    expected_actions = {"answer-now", *(f"ug-grid-{index:02d}" for index in range(4))}
    for (state_id, replicate_id), siblings in grouped.items():
        row = manifest[state_id]
        if replicate_id != "replicate-000" or len(siblings) != 5:
            raise ValueError(f"invalid sibling group: {state_id}")
        if {record.action_id for record in siblings} != expected_actions:
            raise ValueError(f"invalid action set: {state_id}")
        for record in siblings:
            if (
                record.generation_seed != 0
                or record.image_id != str(row["image_id"])
                or record.source_id != str(row["source_id"])
                or record.question != str(row["question"])
                or Path(record.original_image).name
                != Path(str(row["image_path"])).name
            ):
                raise ValueError(f"rollout content differs from manifest: {state_id}")
            target = GroundTruth(row["target"])
            expected_before = chartqapro_match(record.answer_before, target)
            expected_after = chartqapro_match(record.answer_after, target)
            if (
                abs(record.correct_before - expected_before) > 1e-12
                or abs(record.correct_after - expected_after) > 1e-12
            ):
                raise ValueError(f"released scorer mismatch: {state_id}/{record.action_id}")
    return {
        "validated_states": len(manifest),
        "validated_records": len(records),
        "rollouts_sha256": rollout_hash,
        "rollout_provenance_sha256": _sha256(
            rollouts.with_suffix(".provenance.json")
        ),
        "protocol_fields": expected_provenance,
    }


def _rescore_spec(
    records: Sequence[ActionRecord],
    manifest: Mapping[str, Mapping[str, Any]],
) -> list[ActionRecord]:
    rescored: list[ActionRecord] = []
    for record in records:
        target = GroundTruth(manifest[record.state_id]["target"])
        rescored.append(
            replace(
                record,
                correct_before=chartqapro_spec_match(record.answer_before, target),
                correct_after=chartqapro_spec_match(record.answer_after, target),
            )
        )
    return rescored


def _output_compatibility(
    records: Sequence[ActionRecord],
    manifest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    capped = 0
    baseline_capped = 0
    empty = 0
    constrained_total = 0
    constrained_valid = 0
    explanatory = 0
    for record in records:
        metadata_name = (
            "baseline_backend" if record.action_type == "ANSWER" else "action_backend"
        )
        metadata = record.metadata.get(metadata_name)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"missing {metadata_name}: {record.state_id}")
        generated_tokens = int(metadata.get("generated_tokens", -1))
        if generated_tokens < 1 or generated_tokens > 16:
            raise ValueError(f"invalid generated token count: {record.state_id}")
        if generated_tokens == 16:
            capped += 1
            if record.action_type == "ANSWER":
                baseline_capped += 1
        answer = record.answer_after.strip()
        if not answer:
            empty += 1
        normalized = answer.strip(".\n ").casefold()
        question_type = str(manifest[record.state_id]["stratum"])
        if question_type in {"Multi Choice", "Fact Checking"}:
            constrained_total += 1
            allowed = (
                {"a", "b", "c", "d", "unanswerable"}
                if question_type == "Multi Choice"
                else {"true", "false", "unanswerable"}
            )
            constrained_valid += int(normalized in allowed)
        explanatory += int(
            "\n" in answer
            or normalized.startswith("the answer is")
            or normalized.startswith("answer:")
        )
    baseline_count = EXPECTED_STATES
    return {
        "outputs": len(records),
        "empty_outputs": empty,
        "max_token_capped_outputs": capped,
        "max_token_capped_rate": capped / len(records),
        "baseline_max_token_capped_outputs": baseline_capped,
        "baseline_max_token_capped_rate": baseline_capped / baseline_count,
        "constrained_outputs": constrained_total,
        "constrained_format_valid": constrained_valid,
        "constrained_format_compliance": (
            constrained_valid / constrained_total if constrained_total else None
        ),
        "obvious_explanatory_outputs": explanatory,
        "obvious_explanatory_rate": explanatory / len(records),
    }


def _evaluate(
    records: Sequence[ActionRecord],
    *,
    model: Mapping[str, Any],
    source_entropy_threshold: float,
    strata: Mapping[str, str],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    evaluation = evaluate_frozen_factorized_context_model(
        model,
        records,
        source_entropy_threshold=source_entropy_threshold,
        lambda_cost=0.05,
        target_strata=strata,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    scores = score_frozen_factorized_context_model(model, records)
    policy = PrecomputedRescueGatePolicy(
        scores,
        threshold=float(model["threshold"]),
        name="frozen_factorized_context_uniform_random_expectation",
    )
    evaluation["primary_image_bootstrap"] = bootstrap_policy_evaluation(
        records,
        policy,
        lambda_cost=0.05,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        cluster_by="image_id",
    )
    evaluation["gate_score_summary"] = {
        "minimum": min(scores.values()),
        "maximum": max(scores.values()),
        "mean": sum(scores.values()) / len(scores),
        "threshold": float(model["threshold"]),
        "calls": sum(score >= float(model["threshold"]) for score in scores.values()),
        "states": len(scores),
    }
    return evaluation


def _policy_rows(evaluation: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    policies = evaluation["policies"]
    if not isinstance(policies, Mapping):
        raise ValueError("evaluation has no policies")
    names = (
        "frozen_factorized_context",
        "frozen_source_entropy",
        "always_random",
        "exhaustive_entropy",
        "oracle",
    )
    return [(name, policies[name]) for name in names]


def _build_markdown(report: Mapping[str, Any]) -> str:
    compatibility = report["compatibility"]
    assert isinstance(compatibility, Mapping)
    acceptance = report["compatibility_acceptance"]
    assert isinstance(acceptance, Mapping)
    lines = [
        "# ChartQAPro compatibility pilot",
        "",
        "> Compatibility-only pilot; these 309 questions are excluded from formal evaluation.",
        "",
        f"- Compatibility accepted: **{acceptance['passed']}**",
        f"- Empty outputs: {compatibility['empty_outputs']}",
        "- Baseline max-token cap rate: {:.4f}".format(
            compatibility["baseline_max_token_capped_rate"]
        ),
        "- Constrained-format compliance: {:.4f}".format(
            compatibility["constrained_format_compliance"]
        ),
        "",
    ]
    for scorer_name in ("released", "paper_spec"):
        evaluation = report[scorer_name]
        assert isinstance(evaluation, Mapping)
        lines.extend(
            [
                f"## {scorer_name.replace('_', ' ').title()} scorer",
                "",
                "| Policy | Score gain | Tool rate | Utility |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, result in _policy_rows(evaluation):
            lines.append(
                "| {} | {:.4f} | {:.4f} | {:.4f} |".format(
                    name,
                    result["accuracy_gain"],
                    result["tool_use_rate"],
                    result["mean_policy_utility"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze frozen ChartQAPro pilot")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actual_hashes = {
        "manifest": _sha256(args.manifest),
        "frozen_model": _sha256(args.frozen_model),
        "source_report": _sha256(args.source_report),
    }
    expected_hashes = {
        "manifest": EXPECTED_MANIFEST_SHA256,
        "frozen_model": EXPECTED_MODEL_SHA256,
        "source_report": EXPECTED_SOURCE_REPORT_SHA256,
    }
    if actual_hashes != expected_hashes:
        raise ValueError(f"pilot input hash mismatch: {actual_hashes}")
    records = read_jsonl(args.rollouts)
    manifest = _read_manifest(args.manifest)
    provenance = _read_object(args.rollouts.with_suffix(".provenance.json"))
    input_validation = _validate_inputs(
        records,
        manifest,
        provenance,
        rollouts=args.rollouts,
    )
    model = _read_object(args.frozen_model)
    source_report = _read_object(args.source_report)
    source_evaluation = source_report.get("evaluation")
    if not isinstance(source_evaluation, Mapping):
        raise ValueError("source report has no evaluation object")
    source_entropy_threshold = float(source_evaluation["source_entropy_threshold"])
    strata = {state_id: str(row["stratum"]) for state_id, row in manifest.items()}
    compatibility = _output_compatibility(records, manifest)
    spec_records = _rescore_spec(records, manifest)
    released = _evaluate(
        records,
        model=model,
        source_entropy_threshold=source_entropy_threshold,
        strata=strata,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    paper_spec = _evaluate(
        spec_records,
        model=model,
        source_entropy_threshold=source_entropy_threshold,
        strata=strata,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    acceptance = {
        "complete": input_validation["validated_states"] == EXPECTED_STATES,
        "no_empty_outputs": compatibility["empty_outputs"] == 0,
        "baseline_cap_rate_at_most_0_05": (
            compatibility["baseline_max_token_capped_rate"] <= 0.05
        ),
        "all_output_cap_rate_at_most_0_05": (
            compatibility["max_token_capped_rate"] <= 0.05
        ),
        "constrained_format_compliance_at_least_0_95": (
            compatibility["constrained_format_compliance"] >= 0.95
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    report = {
        "scientific_status": (
            "compatibility-only pilot; excluded from untouched formal evaluation"
        ),
        "input_validation": input_validation,
        "input_hashes": actual_hashes,
        "compatibility": compatibility,
        "compatibility_acceptance": acceptance,
        "released": released,
        "paper_spec": paper_spec,
    }
    report_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    _write_json(report_path, report)
    markdown_path.write_text(_build_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": _sha256(report_path),
                "markdown": str(markdown_path),
                "compatibility_passed": acceptance["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
