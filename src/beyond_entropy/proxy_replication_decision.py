from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .answer_likelihood import sha256_file
from .proxy_outcome_audit import AUDIT_SCHEMA


DECISION_SCHEMA = "visual_action_proxy_replication_decision_v1"
COMPLETION_SCHEMA = "visual_action_proxy_replication_completion_v1"
SPARSE_CALL_RATES = (0.005, 0.01, 0.02, 0.05, 0.10)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON document: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _metric(container: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = container.get(name)
    _require(isinstance(value, Mapping), f"missing metric: {name}")
    for field in ("point", "ci_low", "ci_high", "valid_resamples"):
        _require(field in value, f"metric {name} is missing {field}")
    _number(value["point"], f"{name}.point")
    _number(value["ci_low"], f"{name}.ci_low")
    _number(value["ci_high"], f"{name}.ci_high")
    _require(int(value["valid_resamples"]) == 2000, f"{name} resample count mismatch")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"staging file exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"staging file exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _render_markdown(decision: Mapping[str, Any]) -> str:
    lines = [
        "# DocVQA proxy replication decision",
        "",
        f"Decision: **{decision['decision']}**.",
        "",
        "This mechanically applies the five conditions frozen before the full",
        "DocVQA answer-likelihood bank was observed. It selects no score threshold",
        "and does not authorize use of protected outcomes.",
        "",
        "## Conditions",
        "",
    ]
    for name, condition in decision["conditions"].items():
        mark = "PASS" if condition["passed"] else "FAIL"
        lines.append(f"- `{name}`: **{mark}** — {condition['summary']}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            decision["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def decide_proxy_replication(
    *,
    report: str | Path,
    protocol: str | Path,
    output_dir: str | Path,
    expected_report_sha256: str | None = None,
    expected_protocol_sha256: str | None = None,
    expected_study_label: str = "DocVQA ranker development",
    code_revision: str,
) -> dict[str, Any]:
    """Apply the frozen DocVQA proxy-replication decision without tuning."""

    report_path = Path(report).resolve()
    protocol_path = Path(protocol).resolve()
    report_sha256 = sha256_file(report_path)
    protocol_sha256 = sha256_file(protocol_path)
    if expected_report_sha256 is not None:
        _require(report_sha256 == expected_report_sha256, "report SHA-256 mismatch")
    if expected_protocol_sha256 is not None:
        _require(protocol_sha256 == expected_protocol_sha256, "protocol SHA-256 mismatch")
    payload = _read_object(report_path)
    _require(payload.get("schema") == AUDIT_SCHEMA, "proxy audit schema mismatch")
    study = payload.get("study")
    _require(isinstance(study, Mapping), "proxy audit study metadata is missing")
    _require(study.get("label") == expected_study_label, "proxy audit study mismatch")
    inputs = payload.get("inputs")
    _require(isinstance(inputs, Mapping), "proxy audit inputs are missing")
    _require(inputs.get("protocol_sha256") == protocol_sha256, "report/protocol mismatch")
    bootstrap = payload.get("bootstrap")
    _require(isinstance(bootstrap, Mapping), "proxy audit bootstrap metadata is missing")
    _require(int(bootstrap.get("n_resamples", 0)) == 2000, "bootstrap count mismatch")
    _require(int(bootstrap.get("seed", 0)) == 20260901, "bootstrap seed mismatch")
    _require(
        math.isclose(_number(bootstrap.get("confidence_level"), "confidence"), 0.95),
        "bootstrap confidence mismatch",
    )
    outcome_use = payload.get("outcome_use")
    _require(isinstance(outcome_use, Mapping), "outcome-use metadata is missing")
    for name in (
        "candidate_search_reopened",
        "calibration_or_formal_inputs_used",
        "reserve_validation_or_test_inputs_used",
        "protected_role_inputs_used",
    ):
        _require(outcome_use.get(name) is False, f"forbidden outcome use: {name}")

    correlations = payload.get("correlations")
    top_one = payload.get("top_one")
    call_rate_grid = payload.get("call_rate_grid")
    _require(isinstance(correlations, Mapping), "correlations are missing")
    _require(isinstance(top_one, Mapping), "top-one metrics are missing")
    _require(isinstance(call_rate_grid, Mapping), "call-rate grid is missing")

    answer_correlation = correlations.get("answer_loss_gap")
    _require(isinstance(answer_correlation, Mapping), "answer-loss correlation missing")
    answer_spearman = _metric(answer_correlation, "spearman")
    selectors: dict[str, Mapping[str, Any]] = {}
    for name in ("answer_loss_gap", "entropy_reduction", "random_expected"):
        selector = top_one.get(name)
        _require(isinstance(selector, Mapping), f"top-one selector missing: {name}")
        metrics = selector.get("metrics")
        _require(isinstance(metrics, Mapping), f"selector metrics missing: {name}")
        selectors[name] = metrics
    answer_gain = _metric(selectors["answer_loss_gap"], "mean_task_gain")
    entropy_gain = _metric(selectors["entropy_reduction"], "mean_task_gain")
    random_gain = _metric(selectors["random_expected"], "mean_task_gain")
    answer_harm = _metric(selectors["answer_loss_gap"], "induced_harm_rate")
    entropy_harm = _metric(selectors["entropy_reduction"], "induced_harm_rate")
    random_harm = _metric(selectors["random_expected"], "induced_harm_rate")

    grid = call_rate_grid.get("answer_loss_gap")
    _require(isinstance(grid, Sequence) and not isinstance(grid, (str, bytes)), "answer-loss grid missing")
    sparse_rows: dict[float, Mapping[str, Any]] = {}
    for row in grid:
        _require(isinstance(row, Mapping), "malformed answer-loss grid row")
        rate = _number(row.get("target_call_rate"), "target call rate")
        if any(math.isclose(rate, expected, abs_tol=1e-12) for expected in SPARSE_CALL_RATES):
            sparse_rows[rate] = row
    _require(len(sparse_rows) == len(SPARSE_CALL_RATES), "sparse call-rate grid incomplete")
    qualifying_rates: list[float] = []
    sparse_evidence: list[dict[str, float]] = []
    for expected_rate in SPARSE_CALL_RATES:
        matches = [row for rate, row in sparse_rows.items() if math.isclose(rate, expected_rate)]
        _require(len(matches) == 1, "duplicate sparse call-rate row")
        metrics = matches[0].get("metrics")
        _require(isinstance(metrics, Mapping), "sparse grid metrics missing")
        utility = _metric(metrics, "mean_policy_utility")
        ci_low = _number(utility["ci_low"], "sparse utility ci_low")
        point = _number(utility["point"], "sparse utility point")
        sparse_evidence.append(
            {"target_call_rate": expected_rate, "point": point, "ci_low": ci_low}
        )
        if ci_low > 0.0:
            qualifying_rates.append(expected_rate)

    condition_1 = _number(answer_spearman["ci_low"], "Spearman ci_low") > 0.0
    condition_2 = _number(answer_gain["ci_low"], "task-gain ci_low") > 0.0
    condition_3 = (
        _number(answer_gain["point"], "answer gain")
        > _number(entropy_gain["point"], "entropy gain")
        and _number(answer_gain["point"], "answer gain")
        > _number(random_gain["point"], "random gain")
    )
    condition_4 = bool(qualifying_rates)
    condition_5 = (
        _number(answer_harm["point"], "answer harm")
        < _number(entropy_harm["point"], "entropy harm")
        and _number(answer_harm["point"], "answer harm")
        < _number(random_harm["point"], "random harm")
    )
    passed = [condition_1, condition_2, condition_3, condition_4, condition_5]
    if all(passed):
        decision = "replicated_alignment"
        interpretation = (
            "The frozen cross-domain replication gate passed. This authorizes writing "
            "a separate DocVQA-development to ScreenQA-untouched surrogate protocol; "
            "it does not validate or select that future method."
        )
    elif condition_1 and condition_2:
        decision = "partial_alignment"
        interpretation = (
            "The correlation and top-one gain gates passed, but at least one ranking, "
            "sparse-utility, or harm gate failed. The method-transfer branch remains "
            "closed; only diagnostic follow-up is authorized."
        )
    else:
        decision = "non_replication"
        interpretation = (
            "The cross-domain mechanistic gate did not replicate. The method-transfer "
            "branch closes and the result supports only proxy-failure and harm analysis."
        )

    result = {
        "schema": DECISION_SCHEMA,
        "decision": decision,
        "conditions": {
            "answer_loss_spearman_ci_low_above_zero": {
                "passed": condition_1,
                "summary": f"ci_low={float(answer_spearman['ci_low']):.8f}",
            },
            "answer_loss_top_one_gain_ci_low_above_zero": {
                "passed": condition_2,
                "summary": f"ci_low={float(answer_gain['ci_low']):.8f}",
            },
            "answer_loss_top_one_gain_exceeds_entropy_and_random": {
                "passed": condition_3,
                "summary": (
                    f"answer={float(answer_gain['point']):.8f}, "
                    f"entropy={float(entropy_gain['point']):.8f}, "
                    f"random={float(random_gain['point']):.8f}"
                ),
            },
            "positive_sparse_utility_lower_endpoint": {
                "passed": condition_4,
                "summary": f"qualifying_rates={qualifying_rates}",
                "evidence": sparse_evidence,
            },
            "answer_loss_top_one_harm_below_entropy_and_random": {
                "passed": condition_5,
                "summary": (
                    f"answer={float(answer_harm['point']):.8f}, "
                    f"entropy={float(entropy_harm['point']):.8f}, "
                    f"random={float(random_harm['point']):.8f}"
                ),
            },
        },
        "interpretation": interpretation,
        "selection": {
            "score_threshold_selected": False,
            "call_rate_selected": False,
            "protected_outcome_used": False,
        },
        "inputs": {
            "report": str(report_path),
            "report_sha256": report_sha256,
            "protocol": str(protocol_path),
            "protocol_sha256": protocol_sha256,
            "code_revision": code_revision,
            "decision_module_sha256": sha256_file(Path(__file__)),
        },
    }
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    decision_path = output_path / "decision.json"
    markdown_path = output_path / "decision.md"
    completion_path = output_path / "decision.complete.json"
    _require(
        not any(path.exists() for path in (decision_path, markdown_path, completion_path)),
        "replication decision output already exists",
    )
    _atomic_json(decision_path, result)
    _atomic_text(markdown_path, _render_markdown(result))
    completion = {
        "schema": COMPLETION_SCHEMA,
        "decision": decision,
        "decision_json": str(decision_path),
        "decision_json_sha256": sha256_file(decision_path),
        "decision_markdown": str(markdown_path),
        "decision_markdown_sha256": sha256_file(markdown_path),
        "report_sha256": report_sha256,
        "protocol_sha256": protocol_sha256,
        "code_revision": code_revision,
    }
    _atomic_json(completion_path, completion)
    return result
