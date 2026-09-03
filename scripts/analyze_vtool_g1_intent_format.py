#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


REPORT_SCHEMA = "vtool_g1_intent_format_posthoc_v1"
OFFICIAL_REPORT_SCHEMA = "vtool_action_credit_g1_rollout_analysis_v1"
ROLLOUT_AUDIT_SCHEMA = "vtool_action_credit_rollout_audit_v1"
ROLLOUT_AUDIT_JSON_KEY = "vtool_action_credit_audit_json"

ALLOWED_FOCUS_FUNCTIONS = frozenset(
    f"focus_on_{axis}_values_with_{mode}"
    for axis in ("x", "y")
    for mode in ("draw", "highlight", "mask")
)
FINAL_ANSWER_PREFIX = re.compile(r"^FINAL ANSWER\s*:", re.IGNORECASE)
FOCUS_PREFIX = re.compile(r"^(focus_on_[xy]_values_with_(?:draw|highlight|mask))\s*\(")
FENCED_PYTHON = re.compile(r"\A\s*```python\s*(.*?)```\s*\Z", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc, read-only audit of parser-valid calls versus raw tool intent "
            "in a completed paired-signed G1 rollout."
        )
    )
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--official-analysis", type=Path, required=True)
    parser.add_argument("--runtime-vtool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path.name}:{line_number}")
            rows.append(
                _require_mapping(json.loads(line), name=f"{path.name}:{line_number}")
            )
    return rows


def context_assignment_keys(path: Path) -> tuple[str, ...]:
    """Statically recover names assigned into the tool execution context."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    keys: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets.append(node.target)
        for target in targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not isinstance(target.value, ast.Name) or target.value.id != "context":
                continue
            key_node = target.slice
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                keys.add(key_node.value)
    return tuple(sorted(keys))


def _axis_labels(prompt: str, axis: str) -> tuple[str, ...]:
    pattern = re.compile(
        rf"^Available {re.escape(axis)}-axis labels:\s*(.+)$", re.MULTILINE
    )
    matches = pattern.findall(prompt)
    if len(matches) != 1:
        raise ValueError(f"prompt must contain exactly one {axis}-axis label list")
    labels = json.loads(matches[0])
    if not isinstance(labels, list) or not all(
        isinstance(label, str) for label in labels
    ):
        raise ValueError(f"{axis}-axis labels must be a JSON string list")
    if len(labels) != len(set(labels)):
        raise ValueError(f"{axis}-axis labels must be unique")
    return tuple(labels)


def extract_prompt_labels(prompt: str) -> dict[str, tuple[str, ...]]:
    return {axis: _axis_labels(prompt, axis) for axis in ("x", "y")}


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _single_focus_call(
    tree: ast.Module,
) -> tuple[ast.Call | None, bool, bool]:
    """Return focus call, direct-expression flag, and display-wrapper flag."""

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None, False, False
    expression = tree.body[0].value
    if (
        isinstance(expression, ast.Call)
        and _call_name(expression) in ALLOWED_FOCUS_FUNCTIONS
    ):
        return expression, True, False
    if (
        isinstance(expression, ast.Call)
        and _call_name(expression) == "display"
        and len(expression.args) == 1
        and not expression.keywords
        and isinstance(expression.args[0], ast.Call)
        and _call_name(expression.args[0]) in ALLOWED_FOCUS_FUNCTIONS
    ):
        return expression.args[0], False, True
    return None, False, False


def _literal_string_list(node: ast.AST) -> tuple[str, ...] | None:
    if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
        return None
    values: list[str] = []
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            return None
        values.append(element.value)
    return tuple(values)


def inspect_output(
    output: str, *, prompt_labels: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    stripped = output.strip()
    fenced_match = FENCED_PYTHON.fullmatch(stripped)
    has_python_fence = "```python" in stripped
    code = fenced_match.group(1).strip() if fenced_match else stripped
    focus_prefix_match = FOCUS_PREFIX.match(stripped)
    bare_tool_intent = focus_prefix_match is not None and not has_python_fence
    direct_final_answer = FINAL_ANSWER_PREFIX.match(stripped) is not None

    python_ast_parseable = False
    focus_call: ast.Call | None = None
    direct_focus_expression = False
    display_wrapper = False
    try:
        tree = ast.parse(code, mode="exec")
        python_ast_parseable = True
        focus_call, direct_focus_expression, display_wrapper = _single_focus_call(tree)
    except SyntaxError:
        tree = None

    single_allowed_focus_call = focus_call is not None
    function_name = _call_name(focus_call) if focus_call is not None else None
    axis = function_name[len("focus_on_")] if function_name is not None else None
    expected_bbox_name = (
        "columns_bbox" if axis == "x" else "rows_bbox" if axis == "y" else None
    )
    runtime_argument_schema_valid = False
    requested_labels: tuple[str, ...] | None = None
    axis_label_contract_valid: bool | None = None
    if focus_call is not None and axis in {"x", "y"}:
        requested_labels = (
            _literal_string_list(focus_call.args[1])
            if len(focus_call.args) == 3
            else None
        )
        runtime_argument_schema_valid = bool(
            len(focus_call.args) == 3
            and not focus_call.keywords
            and isinstance(focus_call.args[0], ast.Name)
            and focus_call.args[0].id == "image_1"
            and requested_labels is not None
            and isinstance(focus_call.args[2], ast.Name)
            and focus_call.args[2].id == expected_bbox_name
        )
        if runtime_argument_schema_valid and requested_labels is not None:
            available = set(prompt_labels[axis])
            axis_label_contract_valid = bool(
                len(requested_labels) == len(set(requested_labels))
                and all(label in available for label in requested_labels)
            )

    fence_only_repair_executable = bool(
        bare_tool_intent
        and python_ast_parseable
        and direct_focus_expression
        and runtime_argument_schema_valid
        and axis_label_contract_valid
    )
    prompt_contract_valid = bool(
        fenced_match
        and python_ast_parseable
        and display_wrapper
        and runtime_argument_schema_valid
        and axis_label_contract_valid
    )

    failure_reasons: list[str] = []
    if bare_tool_intent:
        failure_reasons.append("missing_python_fence")
    if bare_tool_intent and not python_ast_parseable:
        failure_reasons.append("python_syntax_or_extra_text")
    if bare_tool_intent and python_ast_parseable and not single_allowed_focus_call:
        failure_reasons.append("not_single_allowed_focus_expression")
    if bare_tool_intent and single_allowed_focus_call and not display_wrapper:
        failure_reasons.append("missing_display_wrapper")
    if (
        bare_tool_intent
        and single_allowed_focus_call
        and not runtime_argument_schema_valid
    ):
        failure_reasons.append("runtime_argument_schema_invalid")
    if (
        bare_tool_intent
        and runtime_argument_schema_valid
        and axis_label_contract_valid is False
    ):
        failure_reasons.append("axis_label_contract_invalid")

    return {
        "raw_direct_final_answer": direct_final_answer,
        "raw_bare_tool_intent": bare_tool_intent,
        "python_fence_present": has_python_fence,
        "proper_single_python_fence": fenced_match is not None,
        "python_ast_parseable": python_ast_parseable,
        "single_allowed_focus_call": single_allowed_focus_call,
        "display_wrapper_present": display_wrapper,
        "runtime_argument_schema_valid": runtime_argument_schema_valid,
        "axis_label_contract_valid": axis_label_contract_valid,
        "fence_only_repair_executable": fence_only_repair_executable,
        "prompt_contract_valid": prompt_contract_valid,
        "function_name": function_name,
        "axis": axis,
        "expected_bbox_name": expected_bbox_name,
        "requested_labels": (
            list(requested_labels) if requested_labels is not None else None
        ),
        "failure_reasons": failure_reasons,
    }


def analyze_intent_format(
    *, rollout_dir: Path, official_analysis_path: Path, runtime_vtool_path: Path
) -> tuple[dict[str, Any], int]:
    rollout_dir = rollout_dir.resolve(strict=True)
    official_analysis_path = official_analysis_path.resolve(strict=True)
    runtime_vtool_path = runtime_vtool_path.resolve(strict=True)
    assigned_context_keys = context_assignment_keys(runtime_vtool_path)
    official = _require_mapping(
        json.loads(official_analysis_path.read_text(encoding="utf-8")),
        name="official analysis",
    )
    expected_steps = _require_nonnegative_int(
        official.get("expected_steps"), name="expected_steps"
    )
    expected_rows_per_step = _require_nonnegative_int(
        official.get("expected_rows_per_step"), name="expected_rows_per_step"
    )
    official_tool_count = _require_nonnegative_int(
        official.get("tool_call_count"), name="tool_call_count"
    )

    checks = {
        "official_analysis_schema_matches": official.get("schema")
        == OFFICIAL_REPORT_SCHEMA,
        "official_stop_decision_preserved": official.get("decision")
        == "paired_signed_g1_stop_rule_triggered",
        "official_zero_parser_valid_support": official_tool_count == 0,
        "official_protected_split_closed": official.get(
            "protected_split_contents_accessed"
        )
        is False,
        "all_step_files_present": True,
        "rows_per_step_match": True,
        "step_labels_match": True,
        "rollout_audit_payloads_valid": True,
        "audit_output_matches_rollout_output": True,
        "trajectory_ids_unique": True,
        "parser_valid_count_matches_official": True,
        "all_outputs_accounted_for": True,
        "runtime_context_exposes_required_legacy_aliases": {
            "image_1",
            "columns_bbox",
            "rows_bbox",
        }.issubset(assigned_context_keys),
    }
    failures: list[dict[str, Any]] = []
    rollout_hashes: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    trajectory_ids: set[str] = set()
    parser_valid_count = 0

    for step in range(1, expected_steps + 1):
        path = rollout_dir / f"{step}.jsonl"
        if not path.is_file():
            checks["all_step_files_present"] = False
            failures.append({"step": step, "failure": "missing_step_file"})
            continue
        rollout_hashes[path.name] = sha256_file(path)
        try:
            rows = _load_jsonl(path)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            checks["rollout_audit_payloads_valid"] = False
            failures.append(
                {"step": step, "failure": "invalid_jsonl", "detail": str(exc)}
            )
            continue
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
        for row_index, row in enumerate(rows):
            location = {"step": step, "row_index": row_index}
            if row.get("step") != step:
                checks["step_labels_match"] = False
                failures.append({**location, "failure": "step_label"})
            output = row.get("output")
            prompt = row.get("input")
            raw_audit = row.get(ROLLOUT_AUDIT_JSON_KEY)
            if (
                not isinstance(output, str)
                or not isinstance(prompt, str)
                or not isinstance(raw_audit, str)
            ):
                checks["rollout_audit_payloads_valid"] = False
                failures.append({**location, "failure": "missing_text_or_audit"})
                continue
            try:
                audit = _require_mapping(json.loads(raw_audit), name="rollout audit")
                if audit.get("schema") != ROLLOUT_AUDIT_SCHEMA:
                    raise ValueError("unexpected rollout audit schema")
                attempted = audit.get("vtool_tool_attempted")
                if type(attempted) is not bool:
                    raise ValueError("vtool_tool_attempted must be boolean")
                trajectory_id = audit.get("vtool_action_credit_trajectory_id")
                if not isinstance(trajectory_id, str) or not trajectory_id:
                    raise ValueError("trajectory ID must be non-empty")
                labels = extract_prompt_labels(prompt)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                checks["rollout_audit_payloads_valid"] = False
                failures.append(
                    {**location, "failure": "invalid_rollout_audit", "detail": str(exc)}
                )
                continue
            if trajectory_id in trajectory_ids:
                checks["trajectory_ids_unique"] = False
                failures.append({**location, "failure": "duplicate_trajectory_id"})
            trajectory_ids.add(trajectory_id)
            if audit.get("vtool_final_response_text") != output:
                checks["audit_output_matches_rollout_output"] = False
                failures.append({**location, "failure": "audit_output_mismatch"})
            if attempted:
                parser_valid_count += 1

            inspection = inspect_output(output, prompt_labels=labels)
            output_class_accounted = bool(
                attempted
                or inspection["raw_direct_final_answer"]
                or inspection["raw_bare_tool_intent"]
            )
            if not output_class_accounted:
                checks["all_outputs_accounted_for"] = False
            records.append(
                {
                    **location,
                    "trajectory_id": trajectory_id,
                    "output": output,
                    "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                    "parser_valid_tool_call": attempted,
                    "output_class_accounted": output_class_accounted,
                    **inspection,
                }
            )

    checks["parser_valid_count_matches_official"] = (
        parser_valid_count == official_tool_count
    )
    expected_total_rows = expected_steps * expected_rows_per_step
    if len(records) != expected_total_rows:
        checks["rows_per_step_match"] = False

    count_fields = (
        "parser_valid_tool_call",
        "raw_direct_final_answer",
        "raw_bare_tool_intent",
        "python_fence_present",
        "python_ast_parseable",
        "single_allowed_focus_call",
        "display_wrapper_present",
        "runtime_argument_schema_valid",
        "axis_label_contract_valid",
        "fence_only_repair_executable",
        "prompt_contract_valid",
    )

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rows = len(selected)
        counts = {
            field: sum(record.get(field) is True for record in selected)
            for field in count_fields
        }
        return {
            "rows": rows,
            "counts": counts,
            "rates": {
                field: counts[field] / rows if rows else None for field in count_fields
            },
        }

    total_summary = summarize(records)
    per_step = {
        str(step): summarize([record for record in records if record["step"] == step])
        for step in range(1, expected_steps + 1)
    }
    bare_records = [record for record in records if record["raw_bare_tool_intent"]]
    failure_reason_counts = Counter(
        reason for record in bare_records for reason in record["failure_reasons"]
    )

    artifact_valid = all(checks.values()) and not failures
    bare_count = total_summary["counts"]["raw_bare_tool_intent"]
    fence_repair_count = total_summary["counts"]["fence_only_repair_executable"]
    if not artifact_valid:
        decision = "g1_intent_format_diagnostic_invalid"
        exit_code = 2
    elif bare_count == 0:
        decision = "g1_zero_parser_valid_support_without_detected_tool_intent"
        exit_code = 0
    elif fence_repair_count == 0:
        decision = "g1_zero_parser_valid_support_with_malformed_bare_tool_intent"
        exit_code = 0
    else:
        decision = "g1_zero_parser_valid_support_with_recoverable_bare_tool_intent"
        exit_code = 0

    report = {
        "schema": REPORT_SCHEMA,
        "decision": decision,
        "analyzer": str(Path(__file__).resolve()),
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "checks": checks,
        "failures": failures,
        "official_g1_decision": official.get("decision"),
        "official_g1_decision_changed": False,
        "official_analysis": str(official_analysis_path),
        "official_analysis_sha256": sha256_file(official_analysis_path),
        "rollout_dir": str(rollout_dir),
        "rollout_sha256": rollout_hashes,
        "runtime_contract": {
            "vtool_path": str(runtime_vtool_path),
            "vtool_sha256": sha256_file(runtime_vtool_path),
            "context_assignment_keys": list(assigned_context_keys),
            "prompt_declared_axis_bbox_variables_exposed": {
                "x_values_bbox",
                "y_values_bbox",
            }.issubset(assigned_context_keys),
            "typed_bbox_name_by_axis": {
                "x": "columns_bbox",
                "y": "rows_bbox",
            },
        },
        "expected_steps": expected_steps,
        "expected_rows_per_step": expected_rows_per_step,
        "summary": total_summary,
        "per_step": per_step,
        "bare_tool_intent_failure_reason_counts": dict(
            sorted(failure_reason_counts.items())
        ),
        "bare_tool_intent_records": bare_records,
        "interpretation": {
            "formal_parser_valid_support": "zero",
            "latent_tool_intent": "present" if bare_count else "not_detected",
            "format_only_repair_sufficient": fence_repair_count > 0,
            "protocol_status": "posthoc_diagnostic_only",
        },
        "protected_split_contents_accessed": False,
        "model_weights_loaded": False,
    }
    return report, exit_code


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite post-hoc diagnostic: {output_path}"
        )
    report, exit_code = analyze_intent_format(
        rollout_dir=args.rollout_dir,
        official_analysis_path=args.official_analysis,
        runtime_vtool_path=args.runtime_vtool,
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
