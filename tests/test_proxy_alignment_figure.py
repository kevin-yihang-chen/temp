from __future__ import annotations

import json
from pathlib import Path

import pytest

from beyond_entropy.proxy_alignment_figure import (
    load_audit_figure_data,
    metric_rows,
    render_proxy_alignment_figure,
    sha256_file,
    write_metric_csv,
    write_provenance,
)


def _interval(point: float) -> dict[str, float | int]:
    return {
        "point": point,
        "ci_low": point - 0.01,
        "ci_high": point + 0.01,
        "valid_resamples": 2000,
    }


def _report() -> dict[str, object]:
    return {
        "schema": "visual_action_proxy_outcome_audit_v1",
        "bootstrap": {"n_resamples": 2000, "confidence_level": 0.95},
        "population": {"decisions": 100, "sources": 50},
        "outcome_use": {
            "opened_ranker_development_used": True,
            "calibration_or_formal_inputs_used": False,
            "reserve_validation_or_test_inputs_used": False,
            "protected_role_inputs_used": False,
        },
        "correlations": {
            "answer_loss_gap": {"spearman": _interval(0.2)},
            "entropy_reduction": {"spearman": _interval(0.1)},
        },
        "top_one": {
            key: {"metrics": {"mean_task_gain": _interval(point)}}
            for key, point in (
                ("answer_loss_gap", 0.03),
                ("entropy_reduction", 0.01),
                ("random_expected", -0.01),
            )
        },
        "call_rate_grid": {
            key: [
                {
                    "target_call_rate": rate,
                    "achieved_call_rate": rate + 0.0001,
                    "metrics": {"mean_policy_utility": _interval(point)},
                }
                for rate, point in ((0.005, 0.002), (0.01, 0.003), (0.25, -0.01))
            ]
            for key in ("answer_loss_gap", "entropy_reduction")
        },
    }


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_load_and_render_hash_bound_proxy_figure(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    _write_report(report, _report())
    audit = load_audit_figure_data(
        label="Fixture", report=report, expected_sha256=sha256_file(report)
    )
    assert audit.decisions == 100
    assert audit.correlation["answer_loss_gap"].point == pytest.approx(0.2)
    assert len(metric_rows([audit])) == 11

    pdf = tmp_path / "figure.pdf"
    png = tmp_path / "figure.png"
    csv = tmp_path / "figure.csv"
    provenance = tmp_path / "figure.provenance.json"
    write_metric_csv(csv, metric_rows([audit]))
    render_proxy_alignment_figure([audit], output_pdf=pdf, output_png=png)
    write_provenance(
        provenance,
        data=[audit],
        output_pdf=pdf,
        output_png=png,
        metric_csv=csv,
    )
    assert pdf.stat().st_size > 1_000
    assert png.stat().st_size > 1_000
    output = json.loads(provenance.read_text(encoding="utf-8"))
    assert output["selection"]["protected_outcome_used"] is False
    assert output["outputs"]["pdf"]["sha256"] == sha256_file(pdf)


def test_proxy_figure_rejects_hash_and_outcome_leakage(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    payload = _report()
    _write_report(report, payload)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_audit_figure_data(label="Fixture", report=report, expected_sha256="0" * 64)

    payload["outcome_use"]["protected_role_inputs_used"] = True
    _write_report(report, payload)
    with pytest.raises(ValueError, match="forbidden outcomes"):
        load_audit_figure_data(
            label="Fixture", report=report, expected_sha256=sha256_file(report)
        )


def test_proxy_figure_accepts_percentile_interval_excluding_point(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    payload = _report()
    payload["correlations"]["answer_loss_gap"]["spearman"] = {
        "point": 0.01,
        "ci_low": 0.02,
        "ci_high": 0.04,
        "valid_resamples": 2000,
    }
    _write_report(report, payload)
    audit = load_audit_figure_data(
        label="Fixture", report=report, expected_sha256=sha256_file(report)
    )
    assert audit.correlation["answer_loss_gap"].point == pytest.approx(0.01)
