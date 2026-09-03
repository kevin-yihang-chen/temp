#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.action_value import predict_frozen_factorized_action_values
from beyond_entropy.dataset import read_jsonl
from beyond_entropy.information_set_retrospective import (
    decide_n5_calibration_opening,
    evaluate_information_set_retrospective,
)
from beyond_entropy.qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _cost_metric_suffix(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("registered extra cost must be numeric")
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p")


def _resolve_inputs(
    repo: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, str]]:
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise ValueError("N5 config inputs must be a mapping")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, raw in raw_inputs.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"N5 input {name!r} must be a mapping")
        relative = Path(str(raw.get("path", "")))
        expected = str(raw.get("sha256", ""))
        if relative.is_absolute() or len(expected) != 64:
            raise ValueError(f"N5 input {name!r} has an invalid path or digest")
        path = (repo / relative).resolve()
        try:
            path.relative_to(repo)
        except ValueError as exc:
            raise ValueError(f"N5 input {name!r} escapes the repository") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"N5 input {name!r} SHA-256 mismatch")
        paths[str(name)] = path
        hashes[str(name)] = actual
    return paths, hashes


def _validate_models(lower: Mapping[str, Any], higher: Mapping[str, Any]) -> None:
    for name, model, feature_mode in (
        ("lower", lower, "context-geometry"),
        ("higher", higher, "semantic-context"),
    ):
        _require(
            model.get("model_type"),
            "multidomain_factorized_action_value",
            f"{name} model type",
        )
        _require(model.get("feature_mode"), feature_mode, f"{name} feature mode")
        _require(model.get("domains"), ["docvqa"], f"{name} domains")
        _require(
            model.get("training_protocol"),
            "source_grouped_oof_v1",
            f"{name} training protocol",
        )
        _require(model.get("n_folds"), 5, f"{name} folds")
        _require(model.get("seed"), 20260828, f"{name} seed")
        _require(model.get("selected_alpha"), 10, f"{name} alpha")
        _require(
            _finite(model.get("lambda_cost"), f"{name} lambda"), 0.05, f"{name} lambda"
        )
        _finite(model.get("threshold"), f"{name} threshold")


def _existing_evaluation_metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("existing evaluation is missing metrics")
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen N5 same-bank information-set retrospective audit"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    config = _load_json(config_path, "N5 config")
    _require(
        config.get("schema"),
        "n5_information_set_retrospective_config_v1",
        "N5 config schema",
    )
    paths, input_hashes = _resolve_inputs(repo, config)

    n4 = _load_json(paths["n4_report"], "N4 report")
    _require(
        n4.get("decision"),
        "n4_information_boundary_candidate_survives_formal_gate",
        "N4 decision",
    )
    n4_checks = n4.get("checks")
    if not isinstance(n4_checks, Mapping) or not all(
        value is True for value in n4_checks.values()
    ):
        raise ValueError("N4 checks are not all true")

    allocation = _load_json(
        paths["screenqa_allocation_audit"], "ScreenQA allocation audit"
    )
    risk_role = allocation.get("roles", {}).get("risk_calibration", {})
    if not isinstance(risk_role, Mapping):
        raise ValueError("ScreenQA risk-calibration allocation is missing")
    screenqa_cost = config.get("screenqa_cost_if_opened")
    if not isinstance(screenqa_cost, Mapping):
        raise ValueError("ScreenQA cost contract is missing")
    _require(
        risk_role.get("allocated_images"),
        screenqa_cost.get("allocated_images"),
        "ScreenQA calibration images",
    )
    _require(
        risk_role.get("allocated_qa_rows_identity_only"),
        screenqa_cost.get("identity_only_qa_rows"),
        "ScreenQA calibration decisions",
    )
    _require(
        int(screenqa_cost["identity_only_qa_rows"])
        * int(screenqa_cost["action_records_per_decision"]),
        int(screenqa_cost["estimated_action_records"]),
        "ScreenQA action-record cost",
    )

    lower_model = _load_json(paths["docvqa_lower_model"], "lower model")
    higher_model = _load_json(paths["docvqa_higher_model"], "higher model")
    _validate_models(lower_model, higher_model)
    cost = config.get("cost")
    matched_budget = config.get("matched_budget")
    bootstrap = config.get("bootstrap")
    if not all(
        isinstance(value, Mapping) for value in (cost, matched_budget, bootstrap)
    ):
        raise ValueError("N5 metric configuration is incomplete")
    assert isinstance(cost, Mapping)
    assert isinstance(matched_budget, Mapping)
    assert isinstance(bootstrap, Mapping)
    _require(
        _finite(lower_model["lambda_cost"], "lower lambda"),
        _finite(cost["lambda_cost"], "configured lambda"),
        "configured lower lambda",
    )
    _require(
        _finite(higher_model["lambda_cost"], "higher lambda"),
        _finite(cost["lambda_cost"], "configured lambda"),
        "configured higher lambda",
    )

    feature_audit = _load_json(
        paths["docvqa_higher_features_audit"], "higher feature audit"
    )
    _require(feature_audit.get("decisions"), 1608, "feature-audit decisions")
    _require(
        feature_audit.get("rollouts_sha256"),
        input_hashes["docvqa_rollouts"],
        "feature-audit rollout binding",
    )
    _require(
        feature_audit.get("features_sha256"),
        input_hashes["docvqa_higher_features"],
        "feature-audit feature binding",
    )
    _require(
        feature_audit.get("outcomes_included_metadata"),
        False,
        "feature outcome metadata",
    )
    _require(feature_audit.get("outcome_fields_present"), [], "feature outcome fields")

    records = read_jsonl(paths["docvqa_rollouts"])
    feature_payload = load_semantic_feature_dataset(paths["docvqa_higher_features"])
    validate_semantic_feature_dataset(
        feature_payload,
        records,
        require_outcomes=False,
    )
    semantic_decisions = {
        (str(decision["state_id"]), str(decision["replicate_id"])): decision
        for decision in feature_payload["decisions"]
    }
    lower_actions, lower_scores = predict_frozen_factorized_action_values(
        lower_model,
        records,
    )
    higher_actions, higher_scores = predict_frozen_factorized_action_values(
        higher_model,
        records,
        semantic_decisions=semantic_decisions,
    )
    evaluation = evaluate_information_set_retrospective(
        records,
        lower_actions=lower_actions,
        lower_scores=lower_scores,
        lower_threshold=_finite(lower_model["threshold"], "lower threshold"),
        higher_actions=higher_actions,
        higher_scores=higher_scores,
        higher_threshold=_finite(higher_model["threshold"], "higher threshold"),
        matched_call_rate=_finite(matched_budget["call_rate"], "matched call rate"),
        lambda_cost=_finite(cost["lambda_cost"], "lambda cost"),
        higher_information_extra_costs=tuple(
            float(value) for value in cost["registered_nonnegative_extra_costs"]
        ),
        bootstrap_resamples=int(bootstrap["n_resamples"]),
        bootstrap_confidence=_finite(
            bootstrap["confidence_level"], "bootstrap confidence"
        ),
        bootstrap_seed=int(bootstrap["seed"]),
    )

    lower_existing = _load_json(
        paths["docvqa_existing_lower_evaluation"], "lower existing evaluation"
    )
    higher_existing = _load_json(
        paths["docvqa_existing_higher_evaluation"], "higher existing evaluation"
    )
    lower_existing_metrics = _existing_evaluation_metrics(lower_existing)
    higher_existing_metrics = _existing_evaluation_metrics(higher_existing)
    reproduction = {
        "lower_question_weighted_utility": math.isclose(
            float(evaluation["question_weighted"]["lower_frozen_utility"]),
            float(lower_existing_metrics["mean_policy_utility"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "higher_question_weighted_utility": math.isclose(
            float(evaluation["question_weighted"]["higher_frozen_utility"]),
            float(higher_existing_metrics["mean_policy_utility"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "lower_question_weighted_call_rate": math.isclose(
            float(evaluation["question_weighted"]["lower_frozen_call"]),
            float(lower_existing_metrics["tool_use_rate"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "higher_question_weighted_call_rate": math.isclose(
            float(evaluation["question_weighted"]["higher_frozen_call"]),
            float(higher_existing_metrics["tool_use_rate"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
    }
    if not all(reproduction.values()):
        raise ValueError("N5 failed to reproduce the two existing frozen evaluations")

    screenqa_lower = _load_json(
        paths["screenqa_lower_oof_report"], "ScreenQA lower OOF report"
    )
    screenqa_higher = _load_json(
        paths["screenqa_higher_oof_report"], "ScreenQA higher OOF report"
    )
    _require(
        screenqa_lower.get("development_decisions"), 14511, "ScreenQA lower decisions"
    )
    _require(
        screenqa_higher.get("development_decisions"), 14511, "ScreenQA higher decisions"
    )
    _require(
        screenqa_lower.get("feature_mode"), "context-geometry", "ScreenQA lower mode"
    )
    _require(
        screenqa_higher.get("feature_mode"),
        "hybrid-context-semantic",
        "ScreenQA higher mode",
    )
    lower_oof = screenqa_lower.get("oof_policy_result")
    higher_oof = screenqa_higher.get("oof_policy_result")
    lower_tail = screenqa_lower.get("development_tail_risk_diagnostic")
    higher_tail = screenqa_higher.get("development_tail_risk_diagnostic")
    if not all(
        isinstance(value, Mapping)
        for value in (lower_oof, higher_oof, lower_tail, higher_tail)
    ):
        raise ValueError("ScreenQA OOF evidence is incomplete")
    assert isinstance(lower_oof, Mapping)
    assert isinstance(higher_oof, Mapping)
    assert isinstance(lower_tail, Mapping)
    assert isinstance(higher_tail, Mapping)
    lower_screenqa_utility = _finite(
        lower_oof["mean_policy_utility"], "ScreenQA lower utility"
    )
    higher_screenqa_utility = _finite(
        higher_oof["mean_policy_utility"], "ScreenQA higher utility"
    )
    screenqa_gap = higher_screenqa_utility - lower_screenqa_utility
    screenqa_higher_safe = (
        higher_tail.get("selection_status") == "selected_non_degenerate_safe_threshold"
    )

    information_sets = config.get("information_sets")
    if not isinstance(information_sets, Mapping):
        raise ValueError("N5 information-set ledger is missing")
    unavailable = information_sets.get("n4_required_but_unavailable")
    if not isinstance(unavailable, list):
        raise ValueError("N5 unavailable information sets must be listed")
    exact_sets_available = len(unavailable) == 0
    same_method_factorial = (
        "same_method_family_refit_under_each_information_set" not in unavailable
    )
    stop_rules = config.get("stop_rules")
    if not isinstance(stop_rules, Mapping):
        raise ValueError("N5 stop rules are missing")
    gate = decide_n5_calibration_opening(
        evaluation,
        minimum_material_utility=_finite(
            stop_rules["higher_minus_lower_matched_minimum_utility"],
            "minimum utility",
        ),
        screenqa_higher_minus_lower_utility=screenqa_gap,
        screenqa_higher_has_safe_non_degenerate_threshold=screenqa_higher_safe,
        exact_registered_information_sets_available=exact_sets_available,
        same_method_factorial_available=same_method_factorial,
    )

    zero_metric = evaluation["source_balanced"]["higher_matched_utility_extra_cost_0"]
    cost_values = [
        float(
            evaluation["source_balanced"][
                f"higher_matched_utility_extra_cost_{_cost_metric_suffix(value)}"
            ]
        )
        for value in cost["registered_nonnegative_extra_costs"]
    ]
    cost_monotone = all(
        right <= left + 1e-15 for left, right in zip(cost_values, cost_values[1:])
    ) and math.isclose(float(zero_metric), cost_values[0], abs_tol=1e-15)
    artifact_checks = {
        "all_input_hashes_match": True,
        "n4_gate_is_bound_and_passed": True,
        "docvqa_bank_is_identical": True,
        "both_models_share_training_protocol_seed_folds_alpha_and_lambda": True,
        "semantic_features_are_label_free_and_exactly_cover_bank": True,
        "existing_frozen_evaluations_are_reproduced": all(reproduction.values()),
        "matched_call_sets_are_outcome_blind": evaluation["matched_budget"][
            "selection_uses_outcomes"
        ]
        is False,
        "higher_information_cost_sensitivity_is_monotone": cost_monotone,
        "screenqa_calibration_is_still_unopened": config["data_role"][
            "screenqa_risk_calibration_opened"
        ]
        is False,
        "screenqa_formal_and_reserve_remain_sealed": (
            config["data_role"]["screenqa_formal_test_opened"] is False
            and config["data_role"]["screenqa_reserve_opened"] is False
        ),
    }
    if not all(artifact_checks.values()):
        raise ValueError("N5 artifact contract failed")

    result = {
        "schema": "n5_information_set_retrospective_report_v1",
        "scientific_status": (
            "retrospective route-falsification evidence; not confirmatory or formal"
        ),
        "decision": gate["decision"],
        "passed": gate["passed"],
        "artifact_checks": artifact_checks,
        "scientific_gate": gate,
        "information_set_identifiability": {
            "exact_n4_rank_reversal_identified": False,
            "unavailable_requirements": unavailable,
            "model_capacity_matched": False,
            "interpretation": (
                "tests whether current nested-information candidates justify calibration; "
                "does not identify the causal value of visual information"
            ),
        },
        "docvqa": evaluation,
        "screenqa_opened_development": {
            "n_decisions": 14511,
            "lower_feature_mode": screenqa_lower["feature_mode"],
            "higher_feature_mode": screenqa_higher["feature_mode"],
            "lower_question_weighted_utility": lower_screenqa_utility,
            "higher_question_weighted_utility": higher_screenqa_utility,
            "higher_minus_lower_question_weighted_utility": screenqa_gap,
            "lower_selection_status": lower_tail.get("selection_status"),
            "higher_selection_status": higher_tail.get("selection_status"),
        },
        "screenqa_cost_avoided": {
            "risk_calibration_images": int(screenqa_cost["allocated_images"]),
            "risk_calibration_decisions": int(screenqa_cost["identity_only_qa_rows"]),
            "action_records_not_generated": int(
                screenqa_cost["estimated_action_records"]
            ),
        },
        "authorized_new_gpu_jobs": 0,
        "authorized_new_checkpoints": 0,
        "opened_new_outcome_records": 0,
        "run": {
            "base_code_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "config": str(config_path.relative_to(repo)),
            "config_sha256": _sha256(config_path),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "module_sha256": _sha256(
                repo / "src/beyond_entropy/information_set_retrospective.py"
            ),
            "input_sha256": input_hashes,
            "existing_evaluation_reproduction": reproduction,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
