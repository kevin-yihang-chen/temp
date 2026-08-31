from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from beyond_entropy.answer_likelihood import sha256_file
from beyond_entropy.backbone_replication_decision import decide_backbone_replication
from beyond_entropy.proxy_outcome_audit import AUDIT_SCHEMA


def _interval(point: float, low: float, high: float) -> dict[str, float | int]:
    return {
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "valid_resamples": 5000,
    }


def _report(protocol_sha256: str) -> dict[str, object]:
    def selector(gain: float, harm: float) -> dict[str, object]:
        return {
            "metrics": {
                "mean_task_gain": _interval(gain, gain - 0.01, gain + 0.01),
                "induced_harm_rate": _interval(harm, harm - 0.001, harm + 0.001),
            }
        }

    return {
        "schema": AUDIT_SCHEMA,
        "study": {"label": "ScreenQA Qwen2.5-VL-7B opened development"},
        "inputs": {"protocol_sha256": protocol_sha256},
        "population": {
            "decisions": 512,
            "sources": 512,
            "zoom_actions": 2048,
            "score_records": 2560,
        },
        "bootstrap": {
            "n_resamples": 5000,
            "seed": 20260903,
            "confidence_level": 0.95,
        },
        "outcome_use": {
            "opened_ranker_development_used": True,
            "candidate_search_reopened": False,
            "calibration_or_formal_inputs_used": False,
            "reserve_validation_or_test_inputs_used": False,
            "protected_role_inputs_used": False,
        },
        "correlations": {
            "answer_loss_gap": {"spearman": _interval(0.2, 0.1, 0.3)}
        },
        "top_one": {
            "answer_loss_gap": selector(0.03, 0.01),
            "entropy_reduction": selector(0.01, 0.03),
            "random_expected": selector(-0.01, 0.04),
        },
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_backbone_decision_has_three_frozen_outcomes(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# frozen protocol\n", encoding="utf-8")
    base = _report(sha256_file(protocol))

    report = tmp_path / "pass.json"
    _write(report, base)
    passed = decide_backbone_replication(
        report=report,
        protocol=protocol,
        output_dir=tmp_path / "pass",
        expected_report_sha256=sha256_file(report),
        expected_protocol_sha256=sha256_file(protocol),
        code_revision="test-code",
    )
    assert passed["decision"] == "strong_backbone_replication"
    assert all(row["passed"] for row in passed["conditions"].values())

    partial_payload = copy.deepcopy(base)
    partial_payload["top_one"]["entropy_reduction"]["metrics"][
        "mean_task_gain"
    ] = _interval(0.04, 0.03, 0.05)
    partial_report = tmp_path / "partial.json"
    _write(partial_report, partial_payload)
    partial = decide_backbone_replication(
        report=partial_report,
        protocol=protocol,
        output_dir=tmp_path / "partial",
        code_revision="test-code",
    )
    assert partial["decision"] == "partial_backbone_replication"

    failed_payload = copy.deepcopy(base)
    failed_payload["correlations"]["answer_loss_gap"]["spearman"] = _interval(
        0.01, -0.01, 0.03
    )
    failed_report = tmp_path / "failed.json"
    _write(failed_report, failed_payload)
    failed = decide_backbone_replication(
        report=failed_report,
        protocol=protocol,
        output_dir=tmp_path / "failed",
        code_revision="test-code",
    )
    assert failed["decision"] == "backbone_non_replication"


def test_backbone_decision_rejects_population_and_protected_use(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# frozen protocol\n", encoding="utf-8")
    payload = _report(sha256_file(protocol))
    payload["population"]["sources"] = 511
    report = tmp_path / "bad-population.json"
    _write(report, payload)
    with pytest.raises(ValueError, match="population mismatch"):
        decide_backbone_replication(
            report=report,
            protocol=protocol,
            output_dir=tmp_path / "bad-population",
            code_revision="test-code",
        )

    payload = _report(sha256_file(protocol))
    payload["outcome_use"]["protected_role_inputs_used"] = True
    report = tmp_path / "protected.json"
    _write(report, payload)
    with pytest.raises(ValueError, match="forbidden outcome use"):
        decide_backbone_replication(
            report=report,
            protocol=protocol,
            output_dir=tmp_path / "protected",
            code_revision="test-code",
        )
