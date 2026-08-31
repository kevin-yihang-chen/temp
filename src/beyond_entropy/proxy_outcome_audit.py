from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .answer_likelihood import SCHEMA as SCORE_SCHEMA
from .answer_likelihood import TARGET_RULE, sha256_file


MERGE_SCHEMA = "visual_action_answer_nll_merged_v1"
AUDIT_SCHEMA = "visual_action_proxy_outcome_audit_v1"
LAMBDA_COST = 0.05
CALL_RATES = (0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0)
BOOTSTRAP_CONFIDENCE = 0.95


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON document: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"staging file exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"staging file exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _provenance_path(path: Path) -> Path:
    return path.with_suffix(".provenance.json")


def _decision_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["state_id"]), str(row["replicate_id"])


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    state_id, replicate_id = _decision_key(row)
    return state_id, replicate_id, str(row["action_id"])


def _row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, bool, str]:
    state_id, replicate_id = _decision_key(row)
    return (
        state_id,
        replicate_id,
        str(row["action_type"]) != "ANSWER",
        str(row["action_id"]),
    )


def _validate_score_row(row: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "config_sha256",
        "state_id",
        "replicate_id",
        "source_id",
        "image_id",
        "action_id",
        "action_type",
        "target_answer_sha256",
        "target_answer_index",
        "target_answer_votes",
        "target_answer_count",
        "answer_mean_nll",
        "answer_sum_nll",
        "answer_token_count",
        "entropy_before",
        "entropy_after",
        "correct_before",
        "correct_after",
        "tool_cost",
    }
    _require(required <= set(row), "answer-likelihood row is missing required fields")
    forbidden = {"target_answer", "target_text", "raw_target", "answer_text"}
    _require(not forbidden.intersection(row), "raw target text leaked into score row")
    _require(row["schema"] == SCORE_SCHEMA, "answer-likelihood row schema mismatch")
    _require(str(row["action_type"]) in {"ANSWER", "ZOOM"}, "invalid action type")
    _require(bool(str(row["state_id"])), "empty state_id")
    _require(bool(str(row["replicate_id"])), "empty replicate_id")
    _require(bool(str(row["source_id"])), "empty source_id")
    _require(bool(str(row["action_id"])), "empty action_id")
    _require(len(str(row["target_answer_sha256"])) == 64, "invalid target hash")
    mean_nll = _finite(row["answer_mean_nll"], "answer mean NLL")
    sum_nll = _finite(row["answer_sum_nll"], "answer sum NLL")
    token_count = int(row["answer_token_count"])
    _require(mean_nll >= 0.0 and sum_nll >= 0.0, "negative answer NLL")
    _require(token_count > 0, "answer token count must be positive")
    _require(
        math.isclose(mean_nll * token_count, sum_nll, rel_tol=1e-6, abs_tol=1e-7),
        "answer mean and sum NLL disagree",
    )
    before = _finite(row["correct_before"], "correct_before")
    after = _finite(row["correct_after"], "correct_after")
    _require(0.0 <= before <= 1.0 and 0.0 <= after <= 1.0, "invalid correctness")
    _require(_finite(row["entropy_before"], "entropy_before") >= 0.0, "negative entropy")
    _require(_finite(row["entropy_after"], "entropy_after") >= 0.0, "negative entropy")
    _require(_finite(row["tool_cost"], "tool_cost") >= 0.0, "negative tool cost")


def _validate_decisions(
    rows: Sequence[Mapping[str, Any]], *, expected_zoom_count: int = 4
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        _validate_score_row(row)
        row_key = _row_key(row)
        _require(row_key not in keys, f"duplicate score row: {row_key}")
        keys.add(row_key)
        grouped.setdefault(_decision_key(row), []).append(row)
    for key, siblings in grouped.items():
        answers = [row for row in siblings if row["action_type"] == "ANSWER"]
        zooms = [row for row in siblings if row["action_type"] == "ZOOM"]
        _require(len(answers) == 1, f"decision {key} must contain one ANSWER")
        _require(
            len(zooms) == expected_zoom_count,
            f"decision {key} must contain {expected_zoom_count} ZOOM rows",
        )
        answer = answers[0]
        immutable = (
            "source_id",
            "image_id",
            "target_answer_sha256",
            "target_answer_index",
            "target_answer_votes",
            "target_answer_count",
            "answer_token_count",
            "correct_before",
            "entropy_before",
        )
        for sibling in siblings:
            _require(
                all(sibling[name] == answer[name] for name in immutable),
                f"decision {key} has inconsistent sibling metadata",
            )
        _require(_finite(answer["tool_cost"], "ANSWER tool cost") == 0.0, "ANSWER cost")
        _require(
            _finite(answer["correct_after"], "ANSWER correctness")
            == _finite(answer["correct_before"], "ANSWER correctness"),
            "ANSWER must preserve correctness",
        )
        _require(
            all(_finite(row["tool_cost"], "ZOOM tool cost") > 0.0 for row in zooms),
            "ZOOM rows must have positive cost",
        )
    return grouped


def merge_answer_likelihood_shards(
    *,
    shards: Sequence[str | Path],
    output: str | Path,
    expected_shard_count: int = 4,
    expected_decisions: int | None = None,
    expected_records: int | None = None,
    expected_sources: int | None = None,
) -> dict[str, Any]:
    """Verify and merge state-aligned score shards without rewriting score rows."""

    _require(len(shards) == expected_shard_count, "unexpected score shard count")
    shard_paths = [Path(path).resolve() for path in shards]
    provenances: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    immutable_names = (
        "schema",
        "target_rule",
        "manifest_sha256",
        "rollouts_sha256",
        "model",
        "model_revision",
        "measurement_config",
        "code_revision",
        "shard_count",
        "scientific_status",
    )
    baseline: dict[str, Any] | None = None
    observed_indices: set[int] = set()
    shard_entries: list[dict[str, Any]] = []
    for shard in shard_paths:
        provenance_path = _provenance_path(shard)
        provenance = _read_json(provenance_path)
        rows = _read_jsonl(shard)
        _require(provenance.get("schema") == SCORE_SCHEMA, "score provenance schema mismatch")
        _require(provenance.get("target_rule") == TARGET_RULE, "target rule mismatch")
        _require(provenance.get("raw_targets_written") is False, "raw target contract failed")
        _require(
            sha256_file(shard) == provenance.get("output_sha256"),
            "score shard SHA-256 mismatch",
        )
        _require(int(provenance.get("records", -1)) == len(rows), "score record count mismatch")
        shard_count = int(provenance.get("shard_count", -1))
        shard_index = int(provenance.get("shard_index", -1))
        _require(shard_count == expected_shard_count, "score shard-count contract mismatch")
        _require(0 <= shard_index < expected_shard_count, "invalid score shard index")
        _require(shard_index not in observed_indices, "duplicate score shard index")
        observed_indices.add(shard_index)
        if baseline is None:
            baseline = provenance
        else:
            _require(
                all(provenance.get(name) == baseline.get(name) for name in immutable_names),
                "score shard provenance mismatch",
            )
        config_sha256 = str(provenance.get("config_sha256", ""))
        _require(
            all(row.get("config_sha256") == config_sha256 for row in rows),
            "score row configuration hash mismatch",
        )
        _validate_decisions(rows)
        provenances.append(provenance)
        all_rows.extend(rows)
        shard_entries.append(
            {
                "path": str(shard),
                "provenance": str(provenance_path),
                "shard_index": shard_index,
                "decisions": int(provenance["decisions"]),
                "records": len(rows),
                "output_sha256": str(provenance["output_sha256"]),
                "provenance_sha256": sha256_file(provenance_path),
                "config_sha256": config_sha256,
            }
        )
    _require(observed_indices == set(range(expected_shard_count)), "score shard coverage mismatch")
    grouped = _validate_decisions(all_rows)
    sources = {str(rows[0]["source_id"]) for rows in grouped.values()}
    if expected_decisions is not None:
        _require(len(grouped) == expected_decisions, "merged decision count mismatch")
    if expected_records is not None:
        _require(len(all_rows) == expected_records, "merged score record count mismatch")
    if expected_sources is not None:
        _require(len(sources) == expected_sources, "merged source count mismatch")
    ordered = sorted(all_rows, key=_row_sort_key)
    destination = Path(output).resolve()
    _atomic_write_jsonl(destination, ordered)
    assert baseline is not None
    provenance = {
        "schema": MERGE_SCHEMA,
        "score_schema": SCORE_SCHEMA,
        "target_rule": TARGET_RULE,
        "manifest_sha256": baseline["manifest_sha256"],
        "rollouts_sha256": baseline["rollouts_sha256"],
        "model": baseline["model"],
        "model_revision": baseline["model_revision"],
        "measurement_config": baseline["measurement_config"],
        "code_revision": baseline["code_revision"],
        "scientific_status": baseline["scientific_status"],
        "shard_count": expected_shard_count,
        "decisions": len(grouped),
        "records": len(ordered),
        "sources": len(sources),
        "output": str(destination),
        "output_sha256": sha256_file(destination),
        "raw_targets_written": False,
        "input_shards": sorted(shard_entries, key=lambda row: int(row["shard_index"])),
    }
    _atomic_write_json(_provenance_path(destination), provenance)
    return provenance


@dataclass(frozen=True)
class _RankPlan:
    order: Any
    starts: Any
    group_ids_sorted: Any


def _rank_plan(values: Any) -> _RankPlan:
    import numpy as np  # type: ignore[import-not-found]

    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.flatnonzero(ordered[1:] != ordered[:-1]) + 1)
    )
    ends = np.concatenate((starts[1:], np.asarray([len(values)], dtype=np.int64)))
    group_ids = np.repeat(np.arange(len(starts), dtype=np.int64), ends - starts)
    return _RankPlan(order=order, starts=starts, group_ids_sorted=group_ids)


def _weighted_midranks(weights: Any, plan: _RankPlan) -> Any:
    import numpy as np  # type: ignore[import-not-found]

    ordered_weights = weights[plan.order]
    group_weights = np.add.reduceat(ordered_weights, plan.starts)
    before = np.cumsum(group_weights) - group_weights
    group_ranks = before + (group_weights + 1.0) / 2.0
    ordered_ranks = group_ranks[plan.group_ids_sorted]
    ranks = np.empty_like(ordered_ranks, dtype=np.float64)
    ranks[plan.order] = ordered_ranks
    return ranks


def _weighted_correlation(x: Any, y: Any, weights: Any) -> float:
    import numpy as np  # type: ignore[import-not-found]

    total = float(weights.sum())
    if total <= 0.0:
        return math.nan
    x_mean = float(np.dot(weights, x) / total)
    y_mean = float(np.dot(weights, y) / total)
    x_centered = x - x_mean
    y_centered = y - y_mean
    covariance = float(np.dot(weights, x_centered * y_centered))
    denominator = math.sqrt(
        float(np.dot(weights, x_centered * x_centered))
        * float(np.dot(weights, y_centered * y_centered))
    )
    return covariance / denominator if denominator > 0.0 else math.nan


def _interval(point: float, draws: Any, confidence: float) -> dict[str, Any]:
    import numpy as np  # type: ignore[import-not-found]

    valid = np.asarray(draws, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    alpha = 1.0 - confidence
    return {
        "point": float(point),
        "ci_low": float(np.quantile(valid, alpha / 2.0)) if len(valid) else None,
        "ci_high": float(np.quantile(valid, 1.0 - alpha / 2.0)) if len(valid) else None,
        "valid_resamples": int(len(valid)),
    }


def _source_sums(values: Any, source_indices: Any, source_count: int) -> Any:
    import numpy as np  # type: ignore[import-not-found]

    return np.bincount(source_indices, weights=values, minlength=source_count).astype(
        np.float64
    )


def _bootstrap_ratio(
    numerator: Any,
    denominator: Any,
    source_indices: Any,
    source_weights: Any,
    source_count: int,
    confidence: float,
) -> dict[str, Any]:
    import numpy as np  # type: ignore[import-not-found]

    numerator_by_source = _source_sums(numerator, source_indices, source_count)
    denominator_by_source = _source_sums(denominator, source_indices, source_count)
    point_denominator = float(denominator_by_source.sum())
    _require(point_denominator > 0.0, "metric denominator is zero")
    point = float(numerator_by_source.sum() / point_denominator)
    draw_numerator = source_weights @ numerator_by_source
    draw_denominator = source_weights @ denominator_by_source
    draws = np.divide(
        draw_numerator,
        draw_denominator,
        out=np.full_like(draw_numerator, np.nan, dtype=np.float64),
        where=draw_denominator > 0.0,
    )
    return _interval(point, draws, confidence)


def _selection_metrics(
    *,
    gain: Any,
    utility: Any,
    rescue: Any,
    harm: Any,
    helpful: Any,
    oracle_gain: Any,
    oracle_utility: Any,
    source_indices: Any,
    source_weights: Any,
    source_count: int,
    confidence: float,
) -> dict[str, Any]:
    import numpy as np  # type: ignore[import-not-found]

    ones = np.ones_like(gain, dtype=np.float64)
    return {
        "mean_task_gain": _bootstrap_ratio(
            gain, ones, source_indices, source_weights, source_count, confidence
        ),
        "mean_utility": _bootstrap_ratio(
            utility, ones, source_indices, source_weights, source_count, confidence
        ),
        "rescue_within_helpful_states": _bootstrap_ratio(
            rescue, helpful, source_indices, source_weights, source_count, confidence
        ),
        "induced_harm_rate": _bootstrap_ratio(
            harm, ones, source_indices, source_weights, source_count, confidence
        ),
        "mean_task_gain_regret": _bootstrap_ratio(
            oracle_gain - gain,
            ones,
            source_indices,
            source_weights,
            source_count,
            confidence,
        ),
        "mean_utility_regret": _bootstrap_ratio(
            oracle_utility - utility,
            ones,
            source_indices,
            source_weights,
            source_count,
            confidence,
        ),
    }


def _correlation_report(
    *,
    x: Any,
    y: Any,
    row_source_indices: Any,
    source_weights: Any,
    source_count: int,
    confidence: float,
) -> dict[str, Any]:
    import numpy as np  # type: ignore[import-not-found]

    ones = np.ones_like(x, dtype=np.float64)
    point_pearson = _weighted_correlation(x, y, ones)
    x_plan = _rank_plan(x)
    y_plan = _rank_plan(y)
    point_spearman = _weighted_correlation(
        _weighted_midranks(ones, x_plan),
        _weighted_midranks(ones, y_plan),
        ones,
    )

    count_by_source = _source_sums(ones, row_source_indices, source_count)
    x_by_source = _source_sums(x, row_source_indices, source_count)
    y_by_source = _source_sums(y, row_source_indices, source_count)
    xx_by_source = _source_sums(x * x, row_source_indices, source_count)
    yy_by_source = _source_sums(y * y, row_source_indices, source_count)
    xy_by_source = _source_sums(x * y, row_source_indices, source_count)
    n = source_weights @ count_by_source
    sx = source_weights @ x_by_source
    sy = source_weights @ y_by_source
    sxx = source_weights @ xx_by_source
    syy = source_weights @ yy_by_source
    sxy = source_weights @ xy_by_source
    covariance = sxy - sx * sy / n
    x_variance = sxx - sx * sx / n
    y_variance = syy - sy * sy / n
    denominator = np.sqrt(np.maximum(0.0, x_variance * y_variance))
    pearson_draws = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > 0.0,
    )

    spearman_draws = np.empty(len(source_weights), dtype=np.float64)
    for index, draw in enumerate(source_weights):
        row_weights = draw[row_source_indices].astype(np.float64, copy=False)
        x_ranks = _weighted_midranks(row_weights, x_plan)
        y_ranks = _weighted_midranks(row_weights, y_plan)
        spearman_draws[index] = _weighted_correlation(x_ranks, y_ranks, row_weights)
    return {
        "pearson": _interval(point_pearson, pearson_draws, confidence),
        "spearman": _interval(point_spearman, spearman_draws, confidence),
    }


def _render_interval(metric: Mapping[str, Any]) -> str:
    point = metric["point"]
    low = metric["ci_low"]
    high = metric["ci_high"]
    if low is None or high is None:
        return f"{point:.6f} [NA, NA]"
    return f"{point:.6f} [{low:.6f}, {high:.6f}]"


def render_proxy_audit_markdown(report: Mapping[str, Any]) -> str:
    study = report.get("study", {})
    _require(isinstance(study, Mapping), "proxy audit study metadata is malformed")
    study_label = str(study.get("label", "ScreenQA"))
    interpretation_boundary = str(
        study.get(
            "interpretation_boundary",
            "This audit measures proxy/outcome alignment on an already opened "
            "development bank. Its thresholds are descriptive only and cannot "
            "reopen candidate search or authorize protected-role evaluation.",
        )
    )
    lines = [
        f"# {study_label} proxy-to-outcome audit",
        "",
        f"Status: {report['scientific_status']}.",
        "",
        "## Population",
        "",
        f"- Sources: {report['population']['sources']}",
        f"- Decisions: {report['population']['decisions']}",
        f"- ZOOM actions: {report['population']['zoom_actions']}",
        "",
        "## Proxy correlation with signed task gain",
        "",
        "| Proxy | Pearson (95% source bootstrap CI) | Spearman (95% source bootstrap CI) |",
        "|---|---:|---:|",
    ]
    for name, values in report["correlations"].items():
        lines.append(
            f"| {name} | {_render_interval(values['pearson'])} | "
            f"{_render_interval(values['spearman'])} |"
        )
    lines.extend(
        [
            "",
            "## Always-call top-one crop",
            "",
            "| Selector | Task gain | Utility | Helpful-state rescue | Harm | Task-gain regret |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, policy in report["top_one"].items():
        metrics = policy["metrics"]
        lines.append(
            f"| {name} | {_render_interval(metrics['mean_task_gain'])} | "
            f"{_render_interval(metrics['mean_utility'])} | "
            f"{_render_interval(metrics['rescue_within_helpful_states'])} | "
            f"{_render_interval(metrics['induced_harm_rate'])} | "
            f"{_render_interval(metrics['mean_task_gain_regret'])} |"
        )
    lines.extend(
        [
            "",
            "## Fixed descriptive call-rate grid",
            "",
            "| Proxy | Target rate | Calls | Utility | Gain/call | Helpful-state rescue | Harm/call | Unnecessary/call |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for proxy, rows in report["call_rate_grid"].items():
        for row in rows:
            metrics = row["metrics"]
            lines.append(
                f"| {proxy} | {row['target_call_rate']:.3f} | {row['calls']} | "
                f"{_render_interval(metrics['mean_policy_utility'])} | "
                f"{_render_interval(metrics['task_gain_per_call'])} | "
                f"{_render_interval(metrics['rescue_within_helpful_states'])} | "
                f"{_render_interval(metrics['induced_harm_per_call'])} | "
                f"{_render_interval(metrics['unnecessary_calls_per_call'])} |"
            )
    disagreement = report["disagreements"]
    lines.extend(
        [
            "",
            "## Loss/task disagreement",
            "",
            f"- Positive loss gap but correctness falls: {disagreement['loss_improves_task_falls']['count']} actions.",
            f"- Correctness improves without positive loss gap: {disagreement['task_improves_without_positive_loss_gap']['count']} actions.",
            "",
            "## Interpretation boundary",
            "",
            interpretation_boundary,
            "",
        ]
    )
    return "\n".join(lines)


def analyze_proxy_outcomes(
    *,
    scores: str | Path,
    protocol: str | Path,
    implementation_contract: str | Path,
    output_dir: str | Path,
    expected_scores_sha256: str | None = None,
    expected_protocol_sha256: str | None = None,
    expected_implementation_contract_sha256: str | None = None,
    expected_decisions: int | None = None,
    expected_sources: int | None = None,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 20260831,
    bootstrap_confidence: float = BOOTSTRAP_CONFIDENCE,
    study_label: str = "ScreenQA",
    scientific_status: str | None = None,
    interpretation_boundary: str | None = None,
    code_revision: str,
) -> dict[str, Any]:
    """Run the frozen descriptive proxy-to-outcome audit."""

    import numpy as np  # type: ignore[import-not-found]

    _require(bootstrap_resamples > 0, "bootstrap resamples must be positive")
    _require(0.0 < bootstrap_confidence < 1.0, "invalid bootstrap confidence")
    study_label = study_label.strip()
    _require(bool(study_label) and "\n" not in study_label, "invalid study label")
    if scientific_status is None:
        scientific_status = (
            f"retrospective development-only diagnostic on opened {study_label} "
            "ranker data; not candidate selection or independent validation"
        )
    scientific_status = scientific_status.strip().rstrip(".")
    _require(bool(scientific_status), "empty scientific status")
    if interpretation_boundary is None:
        interpretation_boundary = (
            "This audit measures proxy/outcome alignment on an already opened "
            f"{study_label} development bank. It does not reopen candidate search "
            "or use calibration, formal, reserve, validation, or test inputs. "
            "Thresholds in the table are descriptive only."
        )
    interpretation_boundary = interpretation_boundary.strip()
    _require(bool(interpretation_boundary), "empty interpretation boundary")
    score_path = Path(scores).resolve()
    protocol_path = Path(protocol).resolve()
    implementation_path = Path(implementation_contract).resolve()
    score_sha256 = sha256_file(score_path)
    protocol_sha256 = sha256_file(protocol_path)
    implementation_sha256 = sha256_file(implementation_path)
    if expected_scores_sha256 is not None:
        _require(score_sha256 == expected_scores_sha256, "merged score SHA-256 mismatch")
    if expected_protocol_sha256 is not None:
        _require(protocol_sha256 == expected_protocol_sha256, "audit protocol SHA-256 mismatch")
    if expected_implementation_contract_sha256 is not None:
        _require(
            implementation_sha256 == expected_implementation_contract_sha256,
            "audit implementation contract SHA-256 mismatch",
        )
    provenance = _read_json(_provenance_path(score_path))
    _require(provenance.get("schema") == MERGE_SCHEMA, "merged score provenance mismatch")
    _require(provenance.get("raw_targets_written") is False, "raw target contract failed")
    _require(provenance.get("output_sha256") == score_sha256, "merged score hash mismatch")
    rows = _read_jsonl(score_path)
    grouped = _validate_decisions(rows)
    decision_keys = sorted(grouped)
    sources = sorted({str(grouped[key][0]["source_id"]) for key in decision_keys})
    if expected_decisions is not None:
        _require(len(decision_keys) == expected_decisions, "audit decision count mismatch")
    if expected_sources is not None:
        _require(len(sources) == expected_sources, "audit source count mismatch")
    source_lookup = {source: index for index, source in enumerate(sources)}
    decision_source_indices = np.asarray(
        [source_lookup[str(grouped[key][0]["source_id"])] for key in decision_keys],
        dtype=np.int64,
    )

    gain_rows: list[list[float]] = []
    utility_rows: list[list[float]] = []
    entropy_rows: list[list[float]] = []
    loss_rows: list[list[float]] = []
    action_rows: list[list[str]] = []
    zoom_source_indices: list[int] = []
    disagreement_rows: list[dict[str, Any]] = []
    for decision_index, key in enumerate(decision_keys):
        siblings = grouped[key]
        answer = next(row for row in siblings if row["action_type"] == "ANSWER")
        zooms = sorted(
            (row for row in siblings if row["action_type"] == "ZOOM"),
            key=lambda row: str(row["action_id"]),
        )
        answer_nll = _finite(answer["answer_mean_nll"], "ANSWER NLL")
        gains: list[float] = []
        utilities: list[float] = []
        entropy_proxies: list[float] = []
        loss_proxies: list[float] = []
        actions: list[str] = []
        for zoom in zooms:
            gain = _finite(zoom["correct_after"], "correct_after") - _finite(
                zoom["correct_before"], "correct_before"
            )
            entropy_proxy = _finite(zoom["entropy_before"], "entropy_before") - _finite(
                zoom["entropy_after"], "entropy_after"
            )
            loss_proxy = answer_nll - _finite(zoom["answer_mean_nll"], "ZOOM NLL")
            utility = gain - LAMBDA_COST * _finite(zoom["tool_cost"], "tool cost")
            gains.append(gain)
            utilities.append(utility)
            entropy_proxies.append(entropy_proxy)
            loss_proxies.append(loss_proxy)
            actions.append(str(zoom["action_id"]))
            zoom_source_indices.append(decision_source_indices[decision_index])
            if (loss_proxy > 0.0 and gain < 0.0) or (gain > 0.0 and loss_proxy <= 0.0):
                disagreement_rows.append(
                    {
                        "state_id": key[0],
                        "replicate_id": key[1],
                        "source_id": str(zoom["source_id"]),
                        "action_id": str(zoom["action_id"]),
                        "loss_gap": loss_proxy,
                        "task_gain": gain,
                        "entropy_proxy": entropy_proxy,
                        "utility": utility,
                    }
                )
        gain_rows.append(gains)
        utility_rows.append(utilities)
        entropy_rows.append(entropy_proxies)
        loss_rows.append(loss_proxies)
        action_rows.append(actions)

    gains = np.asarray(gain_rows, dtype=np.float64)
    utilities = np.asarray(utility_rows, dtype=np.float64)
    entropy_proxy = np.asarray(entropy_rows, dtype=np.float64)
    loss_proxy = np.asarray(loss_rows, dtype=np.float64)
    _require(gains.shape[1] == 4, "audit requires four ZOOM actions")
    helpful = (gains.max(axis=1) > 0.0).astype(np.float64)
    oracle_indices = gains.argmax(axis=1)
    decision_indices = np.arange(len(decision_keys), dtype=np.int64)
    oracle_gain = gains[decision_indices, oracle_indices]
    oracle_utility = utilities[decision_indices, oracle_indices]

    rng = np.random.default_rng(bootstrap_seed)
    source_weights = rng.multinomial(
        len(sources),
        np.full(len(sources), 1.0 / len(sources), dtype=np.float64),
        size=bootstrap_resamples,
    ).astype(np.float64)
    zoom_source_array = np.asarray(zoom_source_indices, dtype=np.int64)
    flat_gain = gains.reshape(-1)
    correlations = {
        "answer_loss_gap": _correlation_report(
            x=loss_proxy.reshape(-1),
            y=flat_gain,
            row_source_indices=zoom_source_array,
            source_weights=source_weights,
            source_count=len(sources),
            confidence=bootstrap_confidence,
        ),
        "entropy_reduction": _correlation_report(
            x=entropy_proxy.reshape(-1),
            y=flat_gain,
            row_source_indices=zoom_source_array,
            source_weights=source_weights,
            source_count=len(sources),
            confidence=bootstrap_confidence,
        ),
    }

    selector_indices = {
        "answer_loss_gap": loss_proxy.argmax(axis=1),
        "entropy_reduction": entropy_proxy.argmax(axis=1),
        "oracle": oracle_indices,
    }
    top_one: dict[str, Any] = {}
    for name, indices in selector_indices.items():
        selected_gain = gains[decision_indices, indices]
        selected_utility = utilities[decision_indices, indices]
        top_one[name] = {
            "tie_break": "lexicographically_first_action_id",
            "metrics": _selection_metrics(
                gain=selected_gain,
                utility=selected_utility,
                rescue=(selected_gain > 0.0).astype(np.float64),
                harm=(selected_gain < 0.0).astype(np.float64),
                helpful=helpful,
                oracle_gain=oracle_gain,
                oracle_utility=oracle_utility,
                source_indices=decision_source_indices,
                source_weights=source_weights,
                source_count=len(sources),
                confidence=bootstrap_confidence,
            ),
        }
    random_gain = gains.mean(axis=1)
    random_utility = utilities.mean(axis=1)
    top_one["random_expected"] = {
        "definition": "exact uniform expectation over the four crops; no Monte Carlo seed",
        "metrics": _selection_metrics(
            gain=random_gain,
            utility=random_utility,
            rescue=(gains > 0.0).mean(axis=1),
            harm=(gains < 0.0).mean(axis=1),
            helpful=helpful,
            oracle_gain=oracle_gain,
            oracle_utility=oracle_utility,
            source_indices=decision_source_indices,
            source_weights=source_weights,
            source_count=len(sources),
            confidence=bootstrap_confidence,
        ),
    }

    call_rate_grid: dict[str, list[dict[str, Any]]] = {}
    for proxy_name, proxy_values in (
        ("answer_loss_gap", loss_proxy),
        ("entropy_reduction", entropy_proxy),
    ):
        selected_indices = proxy_values.argmax(axis=1)
        scores_by_decision = proxy_values[decision_indices, selected_indices]
        selected_gain = gains[decision_indices, selected_indices]
        selected_utility = utilities[decision_indices, selected_indices]
        ranking = sorted(
            range(len(decision_keys)),
            key=lambda index: (
                -float(scores_by_decision[index]),
                decision_keys[index],
                action_rows[index][int(selected_indices[index])],
            ),
        )
        grid_rows: list[dict[str, Any]] = []
        for target_rate in CALL_RATES:
            calls = max(1, int(math.floor(target_rate * len(decision_keys) + 0.5)))
            called = np.zeros(len(decision_keys), dtype=np.float64)
            called[np.asarray(ranking[:calls], dtype=np.int64)] = 1.0
            total = np.ones(len(decision_keys), dtype=np.float64)
            gain_when_called = called * selected_gain
            utility_when_called = called * selected_utility
            grid_rows.append(
                {
                    "target_call_rate": target_rate,
                    "calls": calls,
                    "achieved_call_rate": calls / len(decision_keys),
                    "threshold_inclusive": float(scores_by_decision[ranking[calls - 1]]),
                    "threshold_status": "descriptive_development_only",
                    "metrics": {
                        "mean_policy_utility": _bootstrap_ratio(
                            utility_when_called,
                            total,
                            decision_source_indices,
                            source_weights,
                            len(sources),
                            bootstrap_confidence,
                        ),
                        "mean_policy_task_gain": _bootstrap_ratio(
                            gain_when_called,
                            total,
                            decision_source_indices,
                            source_weights,
                            len(sources),
                            bootstrap_confidence,
                        ),
                        "task_gain_per_call": _bootstrap_ratio(
                            gain_when_called,
                            called,
                            decision_source_indices,
                            source_weights,
                            len(sources),
                            bootstrap_confidence,
                        ),
                        "rescue_within_helpful_states": _bootstrap_ratio(
                            called * (selected_gain > 0.0),
                            helpful,
                            decision_source_indices,
                            source_weights,
                            len(sources),
                            bootstrap_confidence,
                        ),
                        "induced_harm_per_call": _bootstrap_ratio(
                            called * (selected_gain < 0.0),
                            called,
                            decision_source_indices,
                            source_weights,
                            len(sources),
                            bootstrap_confidence,
                        ),
                        "unnecessary_calls_per_call": _bootstrap_ratio(
                            called * (selected_utility <= 0.0),
                            called,
                            decision_source_indices,
                            source_weights,
                            len(sources),
                            bootstrap_confidence,
                        ),
                    },
                }
            )
        call_rate_grid[proxy_name] = grid_rows

    loss_falls = [
        row for row in disagreement_rows if row["loss_gap"] > 0.0 and row["task_gain"] < 0.0
    ]
    task_without_loss = [
        row for row in disagreement_rows if row["task_gain"] > 0.0 and row["loss_gap"] <= 0.0
    ]
    output_path = Path(output_dir).resolve()
    report = {
        "schema": AUDIT_SCHEMA,
        "scientific_status": scientific_status,
        "study": {
            "label": study_label,
            "interpretation_boundary": interpretation_boundary,
        },
        "population": {
            "sources": len(sources),
            "decisions": len(decision_keys),
            "score_records": len(rows),
            "zoom_actions": int(gains.size),
            "helpful_decisions": int(helpful.sum()),
        },
        "definitions": {
            "task_gain": "correct_after - correct_before",
            "entropy_reduction": "entropy_before - entropy_after",
            "answer_loss_gap": "answer_now_mean_nll - zoom_mean_nll",
            "utility": f"task_gain - {LAMBDA_COST} * tool_cost",
            "random": "exact uniform expectation over four crops",
            "grid_rounding": "nearest integer with half rounded upward; at least one call",
            "grid_thresholds": "selected on this opened development population; descriptive only",
        },
        "bootstrap": {
            "method": "iid whole-source percentile bootstrap with all decisions/actions retained",
            "n_resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "confidence_level": bootstrap_confidence,
            "spearman": "exact weighted midranks under integer source multiplicities",
            "grid": "outcomes resampled by source conditional on the full-bank descriptive ranking",
        },
        "correlations": correlations,
        "top_one": top_one,
        "call_rate_grid": call_rate_grid,
        "disagreements": {
            "loss_improves_task_falls": {
                "count": len(loss_falls),
                "rate_over_zoom_actions": len(loss_falls) / gains.size,
                "source_count": len({row["source_id"] for row in loss_falls}),
                "examples": sorted(
                    loss_falls, key=lambda row: (-float(row["loss_gap"]), row["state_id"], row["action_id"])
                )[:25],
            },
            "task_improves_without_positive_loss_gap": {
                "count": len(task_without_loss),
                "rate_over_zoom_actions": len(task_without_loss) / gains.size,
                "source_count": len({row["source_id"] for row in task_without_loss}),
                "examples": sorted(
                    task_without_loss,
                    key=lambda row: (float(row["loss_gap"]), row["state_id"], row["action_id"]),
                )[:25],
            },
        },
        "inputs": {
            "scores": str(score_path),
            "scores_sha256": score_sha256,
            "scores_provenance_sha256": sha256_file(_provenance_path(score_path)),
            "protocol": str(protocol_path),
            "protocol_sha256": protocol_sha256,
            "implementation_contract": str(implementation_path),
            "implementation_contract_sha256": implementation_sha256,
            "model": provenance["model"],
            "model_revision": provenance["model_revision"],
            "measurement_config": provenance["measurement_config"],
            "score_code_revision": provenance["code_revision"],
            "analysis_code_revision": code_revision,
            "analysis_module_sha256": sha256_file(Path(__file__)),
        },
        "outcome_use": {
            "opened_ranker_development_used": True,
            "candidate_search_reopened": False,
            "calibration_or_formal_inputs_used": False,
            "reserve_validation_or_test_inputs_used": False,
            "protected_role_inputs_used": False,
        },
    }
    report_path = output_path / "report.json"
    markdown_path = output_path / "report.md"
    _atomic_write_json(report_path, report)
    _atomic_write_text(markdown_path, render_proxy_audit_markdown(report))
    completion = {
        "schema": "visual_action_proxy_outcome_audit_completion_v1",
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "markdown": str(markdown_path),
        "markdown_sha256": sha256_file(markdown_path),
        "scores_sha256": score_sha256,
        "protocol_sha256": protocol_sha256,
        "implementation_contract_sha256": implementation_sha256,
        "analysis_code_revision": code_revision,
        "study_label": study_label,
    }
    _atomic_write_json(output_path / "audit.complete.json", completion)
    return report


def compare_proxy_nll_hardware(
    *,
    first_scores: str | Path,
    first_benchmark: str | Path,
    second_scores: str | Path,
    second_benchmark: str | Path,
    protocol: str | Path,
    output_dir: str | Path,
    expected_first_scores_sha256: str | None = None,
    expected_second_scores_sha256: str | None = None,
    expected_protocol_sha256: str | None = None,
    expected_decisions: int = 64,
    remaining_gpu_minutes: float,
    code_revision: str,
) -> dict[str, Any]:
    """Compare matched engineering scores and apply the frozen hardware rule."""

    import numpy as np  # type: ignore[import-not-found]

    _require(remaining_gpu_minutes > 0.0, "remaining GPU minutes must be positive")
    protocol_path = Path(protocol).resolve()
    protocol_sha256 = sha256_file(protocol_path)
    if expected_protocol_sha256 is not None:
        _require(protocol_sha256 == expected_protocol_sha256, "hardware protocol hash mismatch")

    runs: dict[str, dict[str, Any]] = {}
    for score_value, benchmark_value, expected_hash in (
        (first_scores, first_benchmark, expected_first_scores_sha256),
        (second_scores, second_benchmark, expected_second_scores_sha256),
    ):
        score_path = Path(score_value).resolve()
        benchmark_path = Path(benchmark_value).resolve()
        score_sha256 = sha256_file(score_path)
        if expected_hash is not None:
            _require(score_sha256 == expected_hash, "hardware score SHA-256 mismatch")
        provenance = _read_json(_provenance_path(score_path))
        benchmark = _read_json(benchmark_path)
        _require(provenance.get("schema") == SCORE_SCHEMA, "hardware score schema mismatch")
        _require(provenance.get("output_sha256") == score_sha256, "hardware score hash mismatch")
        _require(provenance.get("raw_targets_written") is False, "hardware raw-target contract")
        _require(benchmark.get("output_sha256") == score_sha256, "benchmark/score hash mismatch")
        gpu_type = str(benchmark.get("gpu_type", ""))
        _require(gpu_type in {"h800", "rtx_4090"}, "unsupported benchmark GPU type")
        _require(gpu_type not in runs, "duplicate benchmark GPU type")
        rows = _read_jsonl(score_path)
        grouped = _validate_decisions(rows)
        _require(len(grouped) == expected_decisions, "hardware decision count mismatch")
        _require(len(rows) == expected_decisions * 5, "hardware score record count mismatch")
        measurement = provenance.get("measurement_config")
        _require(isinstance(measurement, Mapping), "hardware measurement config is absent")
        accelerator = str(measurement.get("accelerator_name", ""))
        if gpu_type == "h800":
            _require("H800" in accelerator, "H800 provenance accelerator mismatch")
        else:
            _require("4090" in accelerator, "4090 provenance accelerator mismatch")
        runs[gpu_type] = {
            "score_path": score_path,
            "score_sha256": score_sha256,
            "provenance": provenance,
            "benchmark_path": benchmark_path,
            "benchmark": benchmark,
            "grouped": grouped,
            "measurement": dict(measurement),
        }
    _require(set(runs) == {"h800", "rtx_4090"}, "matched hardware pair is incomplete")

    first_run = runs["h800"]
    second_run = runs["rtx_4090"]
    first_provenance = first_run["provenance"]
    second_provenance = second_run["provenance"]
    for name in (
        "manifest_sha256",
        "rollouts_sha256",
        "model",
        "model_revision",
        "target_rule",
        "code_revision",
        "shard_count",
        "shard_index",
    ):
        _require(
            first_provenance.get(name) == second_provenance.get(name),
            f"hardware provenance differs for {name}",
        )
    ignored_measurement = {"accelerator_name", "compute_capability"}
    first_measurement = {
        key: value
        for key, value in first_run["measurement"].items()
        if key not in ignored_measurement
    }
    second_measurement = {
        key: value
        for key, value in second_run["measurement"].items()
        if key not in ignored_measurement
    }
    _require(first_measurement == second_measurement, "non-hardware numerical contract differs")

    keys = sorted(first_run["grouped"])
    _require(keys == sorted(second_run["grouped"]), "hardware decision identities differ")
    gaps: dict[str, list[float]] = {"h800": [], "rtx_4090": []}
    selected: dict[str, list[str]] = {"h800": [], "rtx_4090": []}
    for key in keys:
        reference_rows: dict[str, Mapping[str, Any]] | None = None
        for gpu_type in ("h800", "rtx_4090"):
            siblings = runs[gpu_type]["grouped"][key]
            by_action = {str(row["action_id"]): row for row in siblings}
            answer = next(row for row in siblings if row["action_type"] == "ANSWER")
            zooms = sorted(
                (row for row in siblings if row["action_type"] == "ZOOM"),
                key=lambda row: str(row["action_id"]),
            )
            if reference_rows is None:
                reference_rows = by_action
            else:
                _require(set(by_action) == set(reference_rows), "hardware action identities differ")
                for action_id, row in by_action.items():
                    reference = reference_rows[action_id]
                    for name in (
                        "state_id",
                        "replicate_id",
                        "source_id",
                        "image_id",
                        "action_type",
                        "target_answer_sha256",
                        "correct_before",
                        "correct_after",
                        "entropy_before",
                        "entropy_after",
                        "tool_cost",
                    ):
                        _require(row[name] == reference[name], "hardware score populations differ")
            answer_nll = _finite(answer["answer_mean_nll"], "hardware ANSWER NLL")
            decision_gaps = [
                answer_nll - _finite(row["answer_mean_nll"], "hardware ZOOM NLL")
                for row in zooms
            ]
            gaps[gpu_type].extend(decision_gaps)
            best = max(range(len(zooms)), key=lambda index: decision_gaps[index])
            selected[gpu_type].append(str(zooms[best]["action_id"]))

    h800 = np.asarray(gaps["h800"], dtype=np.float64)
    rtx = np.asarray(gaps["rtx_4090"], dtype=np.float64)
    ones = np.ones_like(h800)
    pearson = _weighted_correlation(h800, rtx, ones)
    spearman = _weighted_correlation(
        _weighted_midranks(ones, _rank_plan(h800)),
        _weighted_midranks(ones, _rank_plan(rtx)),
        ones,
    )
    absolute = np.abs(h800 - rtx)
    sign_agreement = float(np.mean((h800 > 0.0) == (rtx > 0.0)))
    top_one_agreement = float(
        np.mean(
            np.asarray(selected["h800"], dtype=object)
            == np.asarray(selected["rtx_4090"], dtype=object)
        )
    )
    rtx_benchmark = runs["rtx_4090"]["benchmark"]
    h800_benchmark = runs["h800"]["benchmark"]
    rtx_wall = _finite(
        rtx_benchmark["projected_four_gpu_full_wall_seconds"], "4090 wall projection"
    )
    rtx_gpu_minutes = _finite(
        rtx_benchmark["projected_four_gpu_gpu_minutes"], "4090 quota projection"
    )
    h800_eligible = spearman >= 0.99 and sign_agreement >= 0.95 and top_one_agreement >= 0.95
    if rtx_wall <= 4.0 * 60.0 * 60.0 and rtx_gpu_minutes <= remaining_gpu_minutes:
        selected_hardware = "rtx_4090"
        decision_reason = "4090 projection fits four hours and live quota; matches rollout hardware"
    elif h800_eligible:
        selected_hardware = "h800"
        decision_reason = "4090 fit condition failed and all frozen H800 stability gates passed"
    else:
        selected_hardware = "rtx_4090_resumable"
        decision_reason = "4090 fit condition failed and H800 stability gates did not all pass"

    report = {
        "schema": "proxy_nll_hardware_consistency_audit_v1",
        "scientific_status": "engineering numerical-stability audit; not a task result",
        "population": {
            "decisions": len(keys),
            "zoom_actions": len(h800),
        },
        "loss_gap_consistency": {
            "pearson": pearson,
            "spearman": spearman,
            "sign_agreement": sign_agreement,
            "top_one_crop_agreement": top_one_agreement,
            "absolute_difference": {
                "median": float(np.median(absolute)),
                "p95": float(np.quantile(absolute, 0.95)),
                "maximum": float(absolute.max()),
            },
        },
        "benchmarks": {
            gpu_type: {
                "accelerator_name": runs[gpu_type]["measurement"]["accelerator_name"],
                "elapsed_seconds": runs[gpu_type]["benchmark"]["elapsed_seconds"],
                "projected_four_gpu_full_wall_seconds": runs[gpu_type]["benchmark"][
                    "projected_four_gpu_full_wall_seconds"
                ],
                "projected_four_gpu_gpu_minutes": runs[gpu_type]["benchmark"][
                    "projected_four_gpu_gpu_minutes"
                ],
                "scores_sha256": runs[gpu_type]["score_sha256"],
                "measurement_config": runs[gpu_type]["measurement"],
            }
            for gpu_type in ("rtx_4090", "h800")
        },
        "hardware_decision": {
            "selected": selected_hardware,
            "reason": decision_reason,
            "remaining_gpu_minutes_at_decision": remaining_gpu_minutes,
            "gates": {
                "rtx_4090_projected_wall_at_most_four_hours": rtx_wall <= 14400.0,
                "rtx_4090_projected_gpu_minutes_fit_live_quota": (
                    rtx_gpu_minutes <= remaining_gpu_minutes
                ),
                "h800_spearman_at_least_0_99": spearman >= 0.99,
                "h800_sign_agreement_at_least_0_95": sign_agreement >= 0.95,
                "h800_top_one_agreement_at_least_0_95": top_one_agreement >= 0.95,
                "h800_all_stability_gates": h800_eligible,
            },
        },
        "inputs": {
            "protocol": str(protocol_path),
            "protocol_sha256": protocol_sha256,
            "analysis_code_revision": code_revision,
            "analysis_module_sha256": sha256_file(Path(__file__)),
            "h800_scores": str(runs["h800"]["score_path"]),
            "rtx_4090_scores": str(runs["rtx_4090"]["score_path"]),
        },
    }
    output_path = Path(output_dir).resolve()
    report_path = output_path / "report.json"
    markdown_path = output_path / "report.md"
    _atomic_write_json(report_path, report)
    lines = [
        "# Proxy-NLL hardware consistency audit",
        "",
        "Status: engineering numerical-stability audit; not a task result.",
        "",
        f"- Matched decisions: {len(keys)}",
        f"- Loss-gap Pearson: {pearson:.8f}",
        f"- Loss-gap Spearman: {spearman:.8f}",
        f"- Sign agreement: {sign_agreement:.6f}",
        f"- Top-one crop agreement: {top_one_agreement:.6f}",
        f"- Median / p95 / max absolute gap difference: {np.median(absolute):.8f} / {np.quantile(absolute, 0.95):.8f} / {absolute.max():.8f}",
        f"- Frozen hardware decision: {selected_hardware}",
        f"- Reason: {decision_reason}",
        "",
    ]
    _atomic_write_text(markdown_path, "\n".join(lines))
    completion = {
        "schema": "proxy_nll_hardware_consistency_completion_v1",
        "report_sha256": sha256_file(report_path),
        "markdown_sha256": sha256_file(markdown_path),
        "protocol_sha256": protocol_sha256,
        "analysis_code_revision": code_revision,
    }
    _atomic_write_json(output_path / "audit.complete.json", completion)
    return report
