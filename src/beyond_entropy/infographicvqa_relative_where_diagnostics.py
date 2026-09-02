from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .infographicvqa_decar import DECAR_ACTION_IDS
from .infographicvqa_relative_where import (
    RELATIVE_WHERE_SCHEMA,
    RELATIVE_WHERE_VARIANTS,
)


ACTION_GENERALIZATION_SCHEMA = (
    "infographicvqa_relative_where_action_generalization_audit_v1"
)
NLL_TOLERANCES = (0.0, 1e-4, 1e-3, 1e-2)


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"relative-where diagnostic {label} is not finite")
    return result


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    import numpy as np  # type: ignore[import-not-found]

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("relative-where diagnostic quantile input is invalid")
    return {
        name: float(np.quantile(array, quantile))
        for name, quantile in (
            ("q00", 0.00),
            ("q10", 0.10),
            ("q25", 0.25),
            ("q50", 0.50),
            ("q75", 0.75),
            ("q90", 0.90),
            ("q100", 1.00),
        )
    }


def _rank_bins(
    rows: Sequence[Mapping[str, Any]], field: str, *, bins: int = 10
) -> list[list[Mapping[str, Any]]]:
    if not rows or bins <= 0:
        raise ValueError("relative-where diagnostic rank-bin input is invalid")
    ordered = sorted(rows, key=lambda row: (float(row[field]), tuple(row["key"])))
    bin_count = min(bins, len(ordered))
    result: list[list[Mapping[str, Any]]] = [[] for _ in range(bin_count)]
    for rank, row in enumerate(ordered):
        result[min(bin_count - 1, rank * bin_count // len(ordered))].append(row)
    return result


def _balanced_mean(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, float | None]:
    available = [row for row in rows if row[field] is not None]
    if not available:
        return {"question_balanced": None, "source_balanced": None}
    question = sum(float(row[field]) for row in available) / len(available)
    by_source: dict[str, list[float]] = defaultdict(list)
    for row in available:
        by_source[str(row["source_id"])].append(float(row[field]))
    source = sum(sum(values) / len(values) for values in by_source.values()) / len(
        by_source
    )
    return {"question_balanced": question, "source_balanced": source}


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("relative-where diagnostic stratum is empty")
    metric_names = (
        "exact_agreement",
        "top2_hit",
        "row_agreement",
        "column_agreement",
        "nll_regret",
        "probability_weighted_nll_regret",
        "pairwise_concordance",
        "max_probability",
        "predicted_margin",
        "teacher_best_second_gap",
        "teacher_crop_nll_range",
        "teacher_best_crop_gain_vs_answer_now",
    )
    metrics = {name: _balanced_mean(rows, name) for name in metric_names}
    tie_aware = {
        f"atol_{tolerance:g}": _balanced_mean(rows, f"tie_aware_hit_{tolerance:g}")
        for tolerance in NLL_TOLERANCES
    }
    regrets = [float(row["nll_regret"]) for row in rows]
    return {
        "decisions": len(rows),
        "sources": len({str(row["source_id"]) for row in rows}),
        "metrics": metrics,
        "tie_aware_hit": tie_aware,
        "nll_regret_quantiles": _quantiles(regrets),
    }


def _strata(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    prefix: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    buckets = _rank_bins(rows, field)
    for index, bucket in enumerate(buckets):
        values = [float(row[field]) for row in bucket]
        result.append(
            {
                "name": f"{prefix}-{index:02d}",
                "rank_low_inclusive": index / len(buckets),
                "rank_high_exclusive": (index + 1) / len(buckets),
                "field_min": min(values),
                "field_max": max(values),
                **_summarize(bucket),
            }
        )
    return result


def _parse_inputs(
    prediction_rows: Sequence[Mapping[str, Any]],
    nll_rows: Sequence[Mapping[str, Any]],
    answer_rows: Sequence[Mapping[str, Any]],
    *,
    expected_decisions: int | None,
    expected_sources: int | None,
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    dict[tuple[str, str], dict[str, float]],
    dict[tuple[str, str], Mapping[str, Any]],
]:
    predictions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in prediction_rows:
        key = (str(row.get("state_id")), str(row.get("replicate_id")))
        variants = row.get("variants")
        if (
            row.get("schema") != RELATIVE_WHERE_SCHEMA
            or key in predictions
            or not isinstance(variants, Mapping)
            or set(variants) != set(RELATIVE_WHERE_VARIANTS)
            or int(row.get("outer_fold", -1)) not in range(5)
        ):
            raise ValueError("relative-where diagnostic prediction contract changed")
        predictions[key] = row

    nll: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    nll_identity: dict[tuple[str, str], tuple[str, str]] = {}
    for row in nll_rows:
        if row.get("schema") != "visual_action_answer_nll_v1":
            raise ValueError("relative-where diagnostic NLL schema changed")
        key = (str(row.get("state_id")), str(row.get("replicate_id")))
        action_id = str(row.get("action_id"))
        if action_id in nll[key]:
            raise ValueError("relative-where diagnostic NLL row is duplicated")
        nll[key][action_id] = _finite(row.get("answer_mean_nll"), "NLL")
        identity = (str(row.get("source_id")), str(row.get("image_id")))
        if key in nll_identity and nll_identity[key] != identity:
            raise ValueError("relative-where diagnostic NLL identity changed")
        nll_identity[key] = identity

    answers: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in answer_rows:
        key = (str(row.get("state_id")), str(row.get("replicate_id")))
        if (
            key in answers
            or str(row.get("action_id")) != "answer-now"
            or str(row.get("action_type")) != "ANSWER"
        ):
            raise ValueError("relative-where diagnostic answer row changed")
        _finite(row.get("entropy_before"), "entropy")
        answers[key] = row

    keys = set(predictions)
    if (
        keys != set(nll)
        or keys != set(answers)
        or (expected_decisions is not None and len(keys) != expected_decisions)
    ):
        raise ValueError("relative-where diagnostic decision coverage changed")
    sources = {str(row.get("source_id")) for row in predictions.values()}
    if expected_sources is not None and len(sources) != expected_sources:
        raise ValueError("relative-where diagnostic source coverage changed")
    expected_actions = {"answer-now", *DECAR_ACTION_IDS}
    for key in keys:
        prediction = predictions[key]
        answer = answers[key]
        identity = (str(prediction.get("source_id")), str(prediction.get("image_id")))
        if (
            set(nll[key]) != expected_actions
            or nll_identity[key] != identity
            or (str(answer.get("source_id")), str(answer.get("image_id"))) != identity
        ):
            raise ValueError("relative-where diagnostic joined identity changed")
    return predictions, dict(nll), answers


def audit_relative_where_action_generalization(
    prediction_rows: Sequence[Mapping[str, Any]],
    nll_rows: Sequence[Mapping[str, Any]],
    answer_rows: Sequence[Mapping[str, Any]],
    *,
    expected_decisions: int | None = 23_946,
    expected_sources: int | None = 2_204,
) -> dict[str, Any]:
    predictions, nll, answers = _parse_inputs(
        prediction_rows,
        nll_rows,
        answer_rows,
        expected_decisions=expected_decisions,
        expected_sources=expected_sources,
    )
    keys = sorted(predictions)
    source_frequencies = Counter(str(predictions[key].get("source_id")) for key in keys)
    teacher_rows: list[dict[str, Any]] = []
    variant_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in RELATIVE_WHERE_VARIANTS
    }
    for key in keys:
        prediction = predictions[key]
        source_id = str(prediction.get("source_id"))
        crop_nll = [nll[key][action_id] for action_id in DECAR_ACTION_IDS]
        best_nll = min(crop_nll)
        ordered_teacher = sorted(
            range(len(DECAR_ACTION_IDS)),
            key=lambda index: (crop_nll[index], DECAR_ACTION_IDS[index]),
        )
        teacher_index = ordered_teacher[0]
        best_second_gap = crop_nll[ordered_teacher[1]] - best_nll
        crop_range = max(crop_nll) - best_nll
        baseline_nll = nll[key]["answer-now"]
        entropy = _finite(answers[key].get("entropy_before"), "entropy")
        common = {
            "key": key,
            "source_id": source_id,
            "outer_fold": int(prediction["outer_fold"]),
            "entropy_before": entropy,
            "teacher_index": teacher_index,
            "teacher_action_id": DECAR_ACTION_IDS[teacher_index],
            "teacher_best_second_gap": best_second_gap,
            "teacher_crop_nll_range": crop_range,
            "teacher_best_crop_gain_vs_answer_now": baseline_nll - best_nll,
            "teacher_exact_best_set_size": sum(value == best_nll for value in crop_nll),
            "source_frequency": source_frequencies[source_id],
        }
        teacher_rows.append(common)
        for name in RELATIVE_WHERE_VARIANTS:
            raw = prediction["variants"][name]
            scores = [_finite(value, "score") for value in raw["action_scores"]]
            probabilities = [
                _finite(value, "probability") for value in raw["action_probabilities"]
            ]
            if (
                len(scores) != len(DECAR_ACTION_IDS)
                or len(probabilities) != len(DECAR_ACTION_IDS)
                or any(value < 0.0 for value in probabilities)
                or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)
            ):
                raise ValueError("relative-where diagnostic prediction vector changed")
            predicted_index = max(range(len(scores)), key=lambda index: scores[index])
            if str(raw.get("selected_action_id")) != DECAR_ACTION_IDS[predicted_index]:
                raise ValueError("relative-where diagnostic selected action changed")
            top2 = sorted(
                range(len(scores)),
                key=lambda index: (-scores[index], DECAR_ACTION_IDS[index]),
            )[:2]
            eligible_pairs = 0
            concordant_pairs = 0.0
            for left in range(len(scores)):
                for right in range(left + 1, len(scores)):
                    truth = crop_nll[right] - crop_nll[left]
                    if truth == 0.0:
                        continue
                    eligible_pairs += 1
                    prediction_difference = scores[left] - scores[right]
                    if prediction_difference * truth > 0.0:
                        concordant_pairs += 1.0
                    elif prediction_difference == 0.0:
                        concordant_pairs += 0.5
            ordered_scores = sorted(scores, reverse=True)
            row: dict[str, Any] = {
                **common,
                "predicted_index": predicted_index,
                "predicted_action_id": DECAR_ACTION_IDS[predicted_index],
                "exact_agreement": float(predicted_index == teacher_index),
                "top2_hit": float(teacher_index in top2),
                "row_agreement": float(predicted_index // 2 == teacher_index // 2),
                "column_agreement": float(predicted_index % 2 == teacher_index % 2),
                "nll_regret": crop_nll[predicted_index] - best_nll,
                "probability_weighted_nll_regret": sum(
                    probabilities[index] * (crop_nll[index] - best_nll)
                    for index in range(len(DECAR_ACTION_IDS))
                ),
                "pairwise_concordance": (
                    concordant_pairs / eligible_pairs if eligible_pairs else None
                ),
                "max_probability": max(probabilities),
                "predicted_margin": ordered_scores[0] - ordered_scores[1],
            }
            for tolerance in NLL_TOLERANCES:
                row[f"tie_aware_hit_{tolerance:g}"] = float(
                    crop_nll[predicted_index] <= best_nll + tolerance
                )
            variant_rows[name].append(row)

    teacher_action_counts = Counter(
        str(row["teacher_action_id"]) for row in teacher_rows
    )
    gaps = [float(row["teacher_best_second_gap"]) for row in teacher_rows]
    ranges = [float(row["teacher_crop_nll_range"]) for row in teacher_rows]
    gains = [float(row["teacher_best_crop_gain_vs_answer_now"]) for row in teacher_rows]
    teacher_audit = {
        "action_counts": {
            action_id: teacher_action_counts[action_id]
            for action_id in DECAR_ACTION_IDS
        },
        "action_rates": {
            action_id: teacher_action_counts[action_id] / len(teacher_rows)
            for action_id in DECAR_ACTION_IDS
        },
        "exact_tie_rate": sum(
            int(row["teacher_exact_best_set_size"]) > 1 for row in teacher_rows
        )
        / len(teacher_rows),
        "near_tie_rate": {
            f"atol_{tolerance:g}": sum(gap <= tolerance for gap in gaps) / len(gaps)
            for tolerance in NLL_TOLERANCES
        },
        "best_second_gap_quantiles": _quantiles(gaps),
        "crop_nll_range_quantiles": _quantiles(ranges),
        "best_crop_gain_vs_answer_now_quantiles": _quantiles(gains),
        "best_crop_beats_answer_now_rate": sum(gain > 0.0 for gain in gains)
        / len(gains),
        "source_frequency_quantiles": _quantiles(
            [float(row["source_frequency"]) for row in teacher_rows]
        ),
    }

    variants: dict[str, Any] = {}
    for name, rows in variant_rows.items():
        predicted_counts = Counter(str(row["predicted_action_id"]) for row in rows)
        confusion = {
            teacher_action: {
                predicted_action: sum(
                    row["teacher_action_id"] == teacher_action
                    and row["predicted_action_id"] == predicted_action
                    for row in rows
                )
                for predicted_action in DECAR_ACTION_IDS
            }
            for teacher_action in DECAR_ACTION_IDS
        }
        by_fold = []
        for fold in range(5):
            fold_rows = [row for row in rows if int(row["outer_fold"]) == fold]
            by_fold.append({"outer_fold": fold, **_summarize(fold_rows)})
        frequency_groups = (
            ("one", lambda count: count == 1),
            ("two_to_four", lambda count: 2 <= count <= 4),
            ("five_to_nine", lambda count: 5 <= count <= 9),
            ("ten_or_more", lambda count: count >= 10),
        )
        by_source_frequency = []
        for label, predicate in frequency_groups:
            current = [row for row in rows if predicate(int(row["source_frequency"]))]
            if current:
                by_source_frequency.append({"name": label, **_summarize(current)})
        variants[name] = {
            "overall": _summarize(rows),
            "predicted_action_counts": {
                action_id: predicted_counts[action_id] for action_id in DECAR_ACTION_IDS
            },
            "predicted_action_rates": {
                action_id: predicted_counts[action_id] / len(rows)
                for action_id in DECAR_ACTION_IDS
            },
            "teacher_by_predicted_confusion": confusion,
            "by_outer_fold": by_fold,
            "by_confidence_decile": _strata(
                rows, field="max_probability", prefix="confidence"
            ),
            "by_teacher_stability_decile": _strata(
                rows, field="teacher_best_second_gap", prefix="teacher-gap"
            ),
            "by_entropy_decile": _strata(
                rows, field="entropy_before", prefix="entropy"
            ),
            "by_source_frequency": by_source_frequency,
        }

    return {
        "schema": ACTION_GENERALIZATION_SCHEMA,
        "scientific_status": "post-gate official-train source-OOF diagnostic",
        "population": {
            "decisions": len(keys),
            "sources": len(source_frequencies),
            "images": len({str(predictions[key].get("image_id")) for key in keys}),
        },
        "action_ids": list(DECAR_ACTION_IDS),
        "nll_tolerances": list(NLL_TOLERANCES),
        "teacher_label_audit": teacher_audit,
        "variants": variants,
        "prediction_outcomes_included": False,
        "validation_or_test_inputs_used": False,
        "changes_parent_train_gate": False,
    }
