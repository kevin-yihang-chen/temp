from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .answer_likelihood import sha256_file
from .proxy_outcome_audit import AUDIT_SCHEMA


DECISION_SCHEMA = "visual_action_backbone_replication_decision_v1"
COMPLETION_SCHEMA = "visual_action_backbone_replication_completion_v1"
EXPECTED_DECISIONS = 512
EXPECTED_SOURCES = 512
EXPECTED_ZOOM_ACTIONS = 2048
EXPECTED_SCORE_RECORDS = 2560
EXPECTED_RESAMPLES = 5000
EXPECTED_BOOTSTRAP_SEED = 20260903


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
    _require(
        int(value["valid_resamples"]) == EXPECTED_RESAMPLES,
        f"{name} resample count mismatch",
    )
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
        "# Qwen2.5-VL-7B backbone replication decision",
        "",
        f"Decision: **{decision['decision']}**.",
        "",
        "This mechanically applies the four conditions frozen before the full",
        "512-state Qwen2.5-VL-7B result was observed. It selects no threshold",
        "or call rate and does not authorize opening a protected role.",
        "",
        "## Conditions",
        "",
    ]
    for name, condition in decision["conditions"].items():
        mark = "PASS" if condition["passed"] else "FAIL"
        lines.append(f"- `{name}`: **{mark}** - {condition['summary']}")
    lines.extend(["", "## Boundary", "", str(decision["interpretation"]), ""])
    return "\n".join(lines)


def decide_backbone_replication(
    *,
    report: str | Path,
    protocol: str | Path,
    output_dir: str | Path,
    expected_report_sha256: str | None = None,
    expected_protocol_sha256: str | None = None,
    expected_study_label: str = "ScreenQA Qwen2.5-VL-7B opened development",
    code_revision: str,
) -> dict[str, Any]:
    """Apply the frozen four-condition backbone decision without tuning."""

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

    population = payload.get("population")
    _require(isinstance(population, Mapping), "proxy audit population is missing")
    expected_population = {
        "decisions": EXPECTED_DECISIONS,
        "sources": EXPECTED_SOURCES,
        "zoom_actions": EXPECTED_ZOOM_ACTIONS,
        "score_records": EXPECTED_SCORE_RECORDS,
    }
    for name, expected in expected_population.items():
        _require(int(population.get(name, -1)) == expected, f"population mismatch: {name}")

    bootstrap = payload.get("bootstrap")
    _require(isinstance(bootstrap, Mapping), "proxy audit bootstrap metadata is missing")
    _require(
        int(bootstrap.get("n_resamples", 0)) == EXPECTED_RESAMPLES,
        "bootstrap count mismatch",
    )
    _require(
        int(bootstrap.get("seed", 0)) == EXPECTED_BOOTSTRAP_SEED,
        "bootstrap seed mismatch",
    )
    _require(
        math.isclose(_number(bootstrap.get("confidence_level"), "confidence"), 0.95),
        "bootstrap confidence mismatch",
    )
    outcome_use = payload.get("outcome_use")
    _require(isinstance(outcome_use, Mapping), "outcome-use metadata is missing")
    _require(
        outcome_use.get("opened_ranker_development_used") is True,
        "opened development use was not declared",
    )
    for name in (
        "candidate_search_reopened",
        "calibration_or_formal_inputs_used",
        "reserve_validation_or_test_inputs_used",
        "protected_role_inputs_used",
    ):
        _require(outcome_use.get(name) is False, f"forbidden outcome use: {name}")

    correlations = payload.get("correlations")
    top_one = payload.get("top_one")
    _require(isinstance(correlations, Mapping), "correlations are missing")
    _require(isinstance(top_one, Mapping), "top-one metrics are missing")
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

    condition_1 = _number(answer_spearman["ci_low"], "Spearman ci_low") > 0.0
    condition_2 = _number(answer_gain["ci_low"], "task-gain ci_low") > 0.0
    condition_3 = (
        _number(answer_gain["point"], "answer gain")
        > _number(entropy_gain["point"], "entropy gain")
        and _number(answer_gain["point"], "answer gain")
        > _number(random_gain["point"], "random gain")
    )
    condition_4 = (
        _number(answer_harm["point"], "answer harm")
        < _number(entropy_harm["point"], "entropy harm")
        and _number(answer_harm["point"], "answer harm")
        < _number(random_harm["point"], "random harm")
    )
    if all((condition_1, condition_2, condition_3, condition_4)):
        decision = "strong_backbone_replication"
        interpretation = (
            "The frozen Qwen2.5-VL-7B mechanism gate passed. This supports only "
            "persistence of the proxy hierarchy across the Qwen 3B/7B scale on "
            "opened ScreenQA development sources."
        )
    elif condition_1 and condition_2:
        decision = "partial_backbone_replication"
        interpretation = (
            "Correlation and top-one gain are positive, but at least one ranking or "
            "harm comparison failed. The result is partial mechanism evidence only."
        )
    else:
        decision = "backbone_non_replication"
        interpretation = (
            "The frozen Qwen2.5-VL-7B mechanism gate did not replicate. This failure "
            "must be reported and cannot be repaired by changing the population or hardware."
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
            "answer_loss_top_one_harm_below_entropy_and_random": {
                "passed": condition_4,
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
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    decision_path = destination / "decision.json"
    markdown_path = destination / "decision.md"
    completion_path = destination / "decision.complete.json"
    _require(
        not any(path.exists() for path in (decision_path, markdown_path, completion_path)),
        "backbone decision output already exists",
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
