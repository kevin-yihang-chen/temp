from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPORT_SCHEMA = "visual_action_proxy_outcome_audit_v1"
PROXY_KEYS = ("answer_loss_gap", "entropy_reduction")
TOP_ONE_KEYS = ("answer_loss_gap", "entropy_reduction", "random_expected")
PROXY_LABELS = {
    "answer_loss_gap": "Answer-loss gap",
    "entropy_reduction": "Entropy reduction",
    "random_expected": "Uniform random",
}
PROXY_COLORS = {
    "answer_loss_gap": "#0072B2",
    "entropy_reduction": "#D55E00",
    "random_expected": "#7F7F7F",
}


@dataclass(frozen=True)
class Interval:
    point: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class AuditFigureData:
    label: str
    path: Path
    sha256: str
    decisions: int
    sources: int
    correlation: Mapping[str, Interval]
    top_one_gain: Mapping[str, Interval]
    sparse_utility: Mapping[str, tuple[tuple[float, float, Interval], ...]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interval(value: Mapping[str, Any], *, context: str) -> Interval:
    required = ("point", "ci_low", "ci_high", "valid_resamples")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"{context} is missing interval fields: {missing}")
    if int(value["valid_resamples"]) != 2000:
        raise ValueError(f"{context} must have 2,000 valid bootstrap resamples")
    result = Interval(
        point=float(value["point"]),
        ci_low=float(value["ci_low"]),
        ci_high=float(value["ci_high"]),
    )
    if not result.ci_low <= result.point <= result.ci_high:
        raise ValueError(f"{context} point must lie inside its confidence interval")
    return result


def _validate_opened_development_report(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported proxy-outcome audit schema")
    bootstrap = payload.get("bootstrap", {})
    if (
        int(bootstrap.get("n_resamples", -1)) != 2000
        or float(bootstrap.get("confidence_level", -1)) != 0.95
    ):
        raise ValueError("paper figure requires the frozen 2,000-draw 95% bootstrap")
    outcome_use = payload.get("outcome_use", {})
    if outcome_use.get("opened_ranker_development_used") is not True:
        raise ValueError("paper figure accepts only opened ranker-development audits")
    forbidden_true = (
        "calibration_opened",
        "formal_opened",
        "reserve_opened",
        "validation_or_test_opened",
        "candidate_search_reopened",
        "screenqa_candidate_search_reopened",
        "calibration_or_formal_inputs_used",
        "reserve_validation_or_test_inputs_used",
        "protected_role_inputs_used",
    )
    used = [key for key in forbidden_true if outcome_use.get(key) is True]
    if used:
        raise ValueError(f"paper figure report used forbidden outcomes: {used}")


def load_audit_figure_data(
    *, label: str, report: Path, expected_sha256: str
) -> AuditFigureData:
    if not label.strip():
        raise ValueError("report label must be non-empty")
    actual_sha256 = sha256_file(report)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label} report SHA-256 mismatch")
    payload = json.loads(report.read_text(encoding="utf-8"))
    _validate_opened_development_report(payload)

    population = payload["population"]
    correlations = {
        key: _interval(
            payload["correlations"][key]["spearman"],
            context=f"{label} {key} Spearman",
        )
        for key in PROXY_KEYS
    }
    top_one = {
        key: _interval(
            payload["top_one"][key]["metrics"]["mean_task_gain"],
            context=f"{label} {key} top-one task gain",
        )
        for key in TOP_ONE_KEYS
    }
    sparse: dict[str, tuple[tuple[float, float, Interval], ...]] = {}
    for key in PROXY_KEYS:
        rows = []
        for index, row in enumerate(payload["call_rate_grid"][key]):
            rows.append(
                (
                    float(row["target_call_rate"]),
                    float(row["achieved_call_rate"]),
                    _interval(
                        row["metrics"]["mean_policy_utility"],
                        context=f"{label} {key} call-rate row {index}",
                    ),
                )
            )
        target_rates = [row[0] for row in rows]
        if target_rates != sorted(target_rates) or len(set(target_rates)) != len(rows):
            raise ValueError(f"{label} {key} call-rate grid is not strictly ordered")
        sparse[key] = tuple(rows)

    return AuditFigureData(
        label=label,
        path=report,
        sha256=actual_sha256,
        decisions=int(population["decisions"]),
        sources=int(population["sources"]),
        correlation=correlations,
        top_one_gain=top_one,
        sparse_utility=sparse,
    )


def metric_rows(data: Sequence[AuditFigureData]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for audit in data:
        for key, interval in audit.correlation.items():
            rows.append(
                {
                    "domain": audit.label,
                    "panel": "spearman",
                    "series": key,
                    "target_call_rate": "",
                    "achieved_call_rate": "",
                    "point": interval.point,
                    "ci_low": interval.ci_low,
                    "ci_high": interval.ci_high,
                }
            )
        for key, interval in audit.top_one_gain.items():
            rows.append(
                {
                    "domain": audit.label,
                    "panel": "top_one_task_gain",
                    "series": key,
                    "target_call_rate": "",
                    "achieved_call_rate": "",
                    "point": interval.point,
                    "ci_low": interval.ci_low,
                    "ci_high": interval.ci_high,
                }
            )
        for key, rate_rows in audit.sparse_utility.items():
            for target_rate, achieved_rate, interval in rate_rows:
                rows.append(
                    {
                        "domain": audit.label,
                        "panel": "sparse_policy_utility",
                        "series": key,
                        "target_call_rate": target_rate,
                        "achieved_call_rate": achieved_rate,
                        "point": interval.point,
                        "ci_low": interval.ci_low,
                        "ci_high": interval.ci_high,
                    }
                )
    return rows


def write_metric_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    fieldnames = (
        "domain",
        "panel",
        "series",
        "target_call_rate",
        "achieved_call_rate",
        "point",
        "ci_low",
        "ci_high",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_proxy_alignment_figure(
    data: Sequence[AuditFigureData], *, output_pdf: Path, output_png: Path
) -> None:
    if not data:
        raise ValueError("at least one audit report is required")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    domains = [audit.label for audit in data]
    centers = np.arange(len(data), dtype=float)

    width = 0.34
    for proxy_index, key in enumerate(PROXY_KEYS):
        values = np.asarray([audit.correlation[key].point for audit in data])
        lows = np.asarray([audit.correlation[key].ci_low for audit in data])
        highs = np.asarray([audit.correlation[key].ci_high for audit in data])
        x = centers + (proxy_index - 0.5) * width
        axes[0].bar(
            x,
            values,
            width=width,
            color=PROXY_COLORS[key],
            label=PROXY_LABELS[key],
            alpha=0.9,
        )
        axes[0].errorbar(
            x,
            values,
            yerr=np.vstack((values - lows, highs - values)),
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2,
        )
    axes[0].set_title("(a) Proxy alignment")
    axes[0].set_ylabel("Spearman with signed task gain")
    axes[0].set_xticks(centers, domains)
    axes[0].axhline(0, color="#444444", linewidth=0.7)
    axes[0].legend(frameon=False)

    width = 0.25
    for proxy_index, key in enumerate(TOP_ONE_KEYS):
        values = np.asarray([audit.top_one_gain[key].point for audit in data])
        lows = np.asarray([audit.top_one_gain[key].ci_low for audit in data])
        highs = np.asarray([audit.top_one_gain[key].ci_high for audit in data])
        x = centers + (proxy_index - 1) * width
        axes[1].bar(
            x,
            100 * values,
            width=width,
            color=PROXY_COLORS[key],
            label=PROXY_LABELS[key],
            alpha=0.9,
        )
        axes[1].errorbar(
            x,
            100 * values,
            yerr=np.vstack((100 * (values - lows), 100 * (highs - values))),
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2,
        )
    axes[1].set_title("(b) Best predicted crop")
    axes[1].set_ylabel("Mean task gain (percentage points)")
    axes[1].set_xticks(centers, domains)
    axes[1].axhline(0, color="#444444", linewidth=0.7)
    axes[1].legend(frameon=False)

    linestyles = ("-", "--", ":", "-.")
    for domain_index, audit in enumerate(data):
        for key in PROXY_KEYS:
            rows = [row for row in audit.sparse_utility[key] if row[0] <= 0.25]
            x = np.asarray([100 * row[1] for row in rows])
            values = np.asarray([100 * row[2].point for row in rows])
            lows = np.asarray([100 * row[2].ci_low for row in rows])
            highs = np.asarray([100 * row[2].ci_high for row in rows])
            label = f"{audit.label}: {PROXY_LABELS[key]}"
            axes[2].plot(
                x,
                values,
                marker="o" if key == "answer_loss_gap" else "s",
                markersize=3.5,
                linewidth=1.3,
                color=PROXY_COLORS[key],
                linestyle=linestyles[domain_index % len(linestyles)],
                label=label,
            )
            axes[2].fill_between(x, lows, highs, color=PROXY_COLORS[key], alpha=0.10)
    axes[2].set_xscale("log")
    axes[2].set_title("(c) Sparse selection frontier")
    axes[2].set_xlabel("Achieved call rate (%)")
    axes[2].set_ylabel("Mean policy utility (percentage points)")
    axes[2].axhline(0, color="#444444", linewidth=0.7)
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(figure)


def write_provenance(
    path: Path,
    *,
    data: Sequence[AuditFigureData],
    output_pdf: Path,
    output_png: Path,
    metric_csv: Path,
) -> None:
    payload = {
        "schema": "proxy_alignment_paper_figure_v1",
        "scientific_status": (
            "descriptive opened-development mechanism evidence; not independent "
            "validation or candidate selection"
        ),
        "reports": [
            {
                "label": audit.label,
                "path": str(audit.path.resolve()),
                "sha256": audit.sha256,
                "decisions": audit.decisions,
                "sources": audit.sources,
            }
            for audit in data
        ],
        "outputs": {
            "pdf": {"path": str(output_pdf.resolve()), "sha256": sha256_file(output_pdf)},
            "png": {"path": str(output_png.resolve()), "sha256": sha256_file(output_png)},
            "csv": {"path": str(metric_csv.resolve()), "sha256": sha256_file(metric_csv)},
        },
        "selection": {
            "threshold_selected": False,
            "call_rate_selected": False,
            "protected_outcome_used": False,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
