#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from beyond_entropy.counterfactual_action_credit import CounterfactualActionPair
from beyond_entropy.vtool_action_credit import (
    ACTION_CREDIT_KEY,
    ACTION_TOKEN_COUNT_KEY,
    ANSWER_TOKEN_COUNT_KEY,
    OBSERVATION_TOKEN_COUNT_KEY,
    PAIR_VALID_KEY,
    TRAJECTORY_ID_KEY,
)

REPORT_SCHEMA = "vtool_action_credit_g1_rollout_analysis_v1"
ROLLOUT_AUDIT_SCHEMA = "vtool_action_credit_rollout_audit_v1"
ROLLOUT_AUDIT_JSON_KEY = "vtool_action_credit_audit_json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit paired-signed G1 rollout dumps and apply frozen stop rules."
    )
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-arm", default="paired_signed_credit")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be a JSON number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("score must be finite and in [0, 1]")
    return result


def _strict_bool(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("expected a JSON boolean")
    return value


def _strict_nonnegative_int(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("expected a non-negative JSON integer")
    return value


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path.name}:{line_number}")
            value = json.loads(line)
            rows.append(_require_mapping(value, name=f"{path.name}:{line_number}"))
    return rows


def analyze_rollouts(
    *,
    rollout_dir: Path,
    config_path: Path,
    expected_arm: str,
) -> tuple[dict[str, Any], int]:
    rollout_dir = rollout_dir.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = _require_mapping(
        json.loads(config_path.read_text(encoding="utf-8")), name="config"
    )
    if config.get("schema") != "vtool_action_credit_g1_config_v1":
        raise ValueError("unsupported G1 config schema")
    arms = _require_mapping(config.get("arms"), name="arms")
    arm = _require_mapping(arms.get(expected_arm), name=f"arms.{expected_arm}")
    credit = _require_mapping(arm.get("action_credit"), name="action_credit")
    if not (
        expected_arm == "paired_signed_credit"
        and arm.get("dataset_family") == "paired"
        and credit.get("enabled") is True
        and credit.get("mode") == "signed"
    ):
        raise ValueError("rollout analyzer is restricted to frozen paired-signed G1")
    training = _require_mapping(config.get("training"), name="training")
    stop_rules = _require_mapping(config.get("stop_rules"), name="stop_rules")
    expected_steps = int(training["total_optimizer_steps"])
    expected_rows_per_step = int(training["data_train_batch_size"]) * int(
        training["rollout_n"]
    )
    tool_call_threshold = float(stop_rules["tool_call_rate_below"])

    checks = {
        "all_step_files_present": True,
        "rows_per_step_match": True,
        "step_labels_match": True,
        "audit_payloads_present": True,
        "audit_schemas_match": True,
        "trajectory_ids_unique": True,
        "direct_contracts_valid": True,
        "tool_contracts_valid": True,
        "scores_valid": True,
        "pair_contracts_valid": True,
    }
    failures: list[dict[str, Any]] = []
    rollout_hashes: dict[str, str] = {}
    step_rows: dict[int, list[Mapping[str, Any]]] = {}
    for step in range(1, expected_steps + 1):
        path = rollout_dir / f"{step}.jsonl"
        if not path.is_file():
            checks["all_step_files_present"] = False
            failures.append({"step": step, "failure": "missing_step_file"})
            step_rows[step] = []
            continue
        rollout_hashes[path.name] = sha256_file(path)
        try:
            rows = _load_jsonl(path)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            checks["audit_payloads_present"] = False
            failures.append(
                {"step": step, "failure": "invalid_jsonl", "detail": str(exc)}
            )
            rows = []
        step_rows[step] = rows
        if len(rows) != expected_rows_per_step:
            checks["rows_per_step_match"] = False
            failures.append(
                {
                    "step": step,
                    "failure": "row_count",
                    "actual": len(rows),
                    "expected": expected_rows_per_step,
                }
            )

    seen_trajectory_ids: set[str] = set()
    pair_mismatch_count = 0
    scorer_failure_count = 0
    tool_count = 0
    tool_success_count = 0
    harmful_count = 0
    rescue_count = 0
    no_effect_count = 0
    task_scores: list[float] = []
    realized_utilities: list[float] = []
    action_credits: list[float] = []
    per_step: dict[str, dict[str, Any]] = {}

    for step, rows in step_rows.items():
        step_task_scores: list[float] = []
        step_utilities: list[float] = []
        step_credits: list[float] = []
        step_tools = 0
        step_harmful = 0
        step_rescue = 0
        for row_index, row in enumerate(rows):
            location = {"step": step, "row_index": row_index}
            if row.get("step") != step:
                checks["step_labels_match"] = False
                failures.append({**location, "failure": "step_label"})
            try:
                row_score = _finite_score(row.get("score"))
                acc = _finite_score(row.get("acc"))
                if not math.isclose(row_score, acc, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError("score and acc differ")
            except (TypeError, ValueError):
                checks["scores_valid"] = False
                scorer_failure_count += 1
                failures.append({**location, "failure": "scorer_contract"})
                continue
            raw_audit = row.get(ROLLOUT_AUDIT_JSON_KEY)
            if not isinstance(raw_audit, str):
                checks["audit_payloads_present"] = False
                failures.append({**location, "failure": "missing_audit_json"})
                continue
            try:
                audit = _require_mapping(
                    json.loads(raw_audit), name=ROLLOUT_AUDIT_JSON_KEY
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                checks["audit_payloads_present"] = False
                failures.append({**location, "failure": "invalid_audit_json"})
                continue
            if audit.get("schema") != ROLLOUT_AUDIT_SCHEMA:
                checks["audit_schemas_match"] = False
                failures.append({**location, "failure": "audit_schema"})
                continue
            try:
                trajectory_id = str(audit[TRAJECTORY_ID_KEY])
                if not trajectory_id or trajectory_id in seen_trajectory_ids:
                    raise ValueError("trajectory ID is empty or duplicated")
                seen_trajectory_ids.add(trajectory_id)
                attempted = _strict_bool(audit["vtool_tool_attempted"])
                success = _strict_bool(audit["vtool_tool_success"])
                pair_valid = _strict_bool(audit[PAIR_VALID_KEY])
                action_count = _strict_nonnegative_int(audit[ACTION_TOKEN_COUNT_KEY])
                observation_count = _strict_nonnegative_int(
                    audit[OBSERVATION_TOKEN_COUNT_KEY]
                )
                answer_count = _strict_nonnegative_int(audit[ANSWER_TOKEN_COUNT_KEY])
                action_credit = float(audit[ACTION_CREDIT_KEY])
                generation_seconds = float(
                    audit["vtool_counterfactual_generation_seconds"]
                )
                if not math.isfinite(action_credit) or not math.isfinite(
                    generation_seconds
                ):
                    raise ValueError("audit floats must be finite")
                if generation_seconds < 0.0 or answer_count <= 0:
                    raise ValueError("invalid generation time or answer token count")
                if not isinstance(audit["vtool_final_response_text"], str):
                    raise ValueError("final response text must be a string")
            except (KeyError, TypeError, ValueError):
                checks["trajectory_ids_unique"] = len(seen_trajectory_ids) == sum(
                    len(value) for value in step_rows.values()
                )
                checks["tool_contracts_valid"] = False
                failures.append({**location, "failure": "audit_field_contract"})
                continue

            task_scores.append(row_score)
            step_task_scores.append(row_score)
            if not attempted:
                direct_valid = (
                    not success
                    and not pair_valid
                    and action_count == 0
                    and observation_count == 0
                    and action_credit == 0.0
                    and audit.get("vtool_action_credit_pair") is None
                    and audit.get("vtool_counterfactual_response_text") is None
                    and generation_seconds == 0.0
                )
                if not direct_valid:
                    checks["direct_contracts_valid"] = False
                    failures.append({**location, "failure": "direct_contract"})
                realized_utilities.append(row_score)
                step_utilities.append(row_score)
                continue

            tool_count += 1
            step_tools += 1
            if success:
                tool_success_count += 1
            tool_shape_valid = (
                pair_valid
                and action_count > 0
                and observation_count > 0
                and isinstance(audit.get("vtool_counterfactual_response_text"), str)
            )
            if not tool_shape_valid:
                checks["tool_contracts_valid"] = False
                failures.append({**location, "failure": "tool_shape_contract"})
            try:
                pair = CounterfactualActionPair.from_dict(
                    _require_mapping(
                        audit.get("vtool_action_credit_pair"), name="action pair"
                    )
                )
                if pair.trajectory_id != trajectory_id:
                    raise ValueError("pair trajectory ID mismatch")
                if not math.isclose(
                    pair.action_credit,
                    action_credit,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("action credit mismatch")
                if not math.isclose(
                    pair.factual.task_score, row_score, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError("factual score mismatch")
            except (KeyError, TypeError, ValueError):
                pair_mismatch_count += 1
                checks["pair_contracts_valid"] = False
                failures.append({**location, "failure": "pair_contract"})
                continue
            action_credits.append(pair.action_credit)
            step_credits.append(pair.action_credit)
            utility = (
                pair.factual.task_score - pair.lambda_cost * pair.factual.action_cost
            )
            realized_utilities.append(utility)
            step_utilities.append(utility)
            if pair.action_credit < 0.0:
                harmful_count += 1
                step_harmful += 1
            elif pair.action_credit > 0.0:
                rescue_count += 1
                step_rescue += 1
            else:
                no_effect_count += 1

        per_step[str(step)] = {
            "rows": len(rows),
            "task_score_mean": mean(step_task_scores) if step_task_scores else None,
            "realized_cost_adjusted_utility_mean": (
                mean(step_utilities) if step_utilities else None
            ),
            "tool_call_count": step_tools,
            "tool_call_rate": step_tools / len(rows) if rows else None,
            "mean_signed_action_credit": (mean(step_credits) if step_credits else None),
            "harmful_call_rate": step_harmful / step_tools if step_tools else None,
            "rescue_rate": step_rescue / step_tools if step_tools else None,
        }

    total_rows = sum(len(rows) for rows in step_rows.values())
    checks["trajectory_ids_unique"] = len(seen_trajectory_ids) == total_rows
    artifact_valid = all(checks.values()) and not failures
    tool_call_rate = tool_count / total_rows if total_rows else 0.0
    stop_reasons: list[str] = []
    if tool_call_rate < tool_call_threshold:
        stop_reasons.append("tool_call_rate_below_frozen_threshold")
    if pair_mismatch_count > 0:
        stop_reasons.append("pair_mismatch_count_above_zero")
    if scorer_failure_count > 0:
        stop_reasons.append("judge_failure_count_above_zero")
    if not artifact_valid:
        decision = "paired_signed_g1_rollout_artifact_invalid"
        exit_code = 2
    elif stop_reasons:
        decision = "paired_signed_g1_stop_rule_triggered"
        exit_code = 0
    else:
        decision = "paired_signed_g1_smoke_gate_passed"
        exit_code = 0
    report = {
        "schema": REPORT_SCHEMA,
        "decision": decision,
        "arm": expected_arm,
        "checks": checks,
        "failures": failures,
        "stop_reasons": stop_reasons,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "rollout_dir": str(rollout_dir),
        "rollout_sha256": rollout_hashes,
        "expected_steps": expected_steps,
        "expected_rows_per_step": expected_rows_per_step,
        "rows": total_rows,
        "pair_mismatch_count": pair_mismatch_count,
        "judge_failure_count": scorer_failure_count,
        "tool_call_count": tool_count,
        "tool_call_rate": tool_call_rate,
        "tool_call_rate_frozen_minimum": tool_call_threshold,
        "tool_success_rate": tool_success_count / tool_count if tool_count else None,
        "task_score_mean": mean(task_scores) if task_scores else None,
        "realized_cost_adjusted_utility_mean": (
            mean(realized_utilities) if realized_utilities else None
        ),
        "mean_signed_action_credit": (mean(action_credits) if action_credits else None),
        "harmful_call_rate": harmful_count / tool_count if tool_count else None,
        "rescue_rate": rescue_count / tool_count if tool_count else None,
        "no_effect_call_rate": no_effect_count / tool_count if tool_count else None,
        "per_step": per_step,
        "protected_split_contents_accessed": False,
    }
    return report, exit_code


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite G1 analysis: {output_path}")
    report, exit_code = analyze_rollouts(
        rollout_dir=args.rollout_dir,
        config_path=args.config,
        expected_arm=args.expected_arm,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": report["decision"], "output": str(output_path)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
