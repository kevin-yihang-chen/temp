from __future__ import annotations

import copy
import json
from pathlib import Path

from beyond_entropy.answer_likelihood import sha256_file
from beyond_entropy.proxy_outcome_audit import AUDIT_SCHEMA
from beyond_entropy.proxy_replication_decision import decide_proxy_replication


def _interval(point: float, low: float, high: float) -> dict[str, float | int]:
    return {
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "valid_resamples": 2000,
    }


def _report(protocol_sha256: str) -> dict[str, object]:
    selector = lambda gain, harm: {  # noqa: E731
        "metrics": {
            "mean_task_gain": _interval(gain, gain - 0.01, gain + 0.01),
            "induced_harm_rate": _interval(harm, harm - 0.001, harm + 0.001),
        }
    }
    return {
        "schema": AUDIT_SCHEMA,
        "study": {"label": "DocVQA ranker development"},
        "inputs": {"protocol_sha256": protocol_sha256},
        "bootstrap": {
            "n_resamples": 2000,
            "seed": 20260901,
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
            "answer_loss_gap": {
                "spearman": _interval(0.2, 0.1, 0.3),
            }
        },
        "top_one": {
            "answer_loss_gap": selector(0.03, 0.01),
            "entropy_reduction": selector(0.01, 0.03),
            "random_expected": selector(-0.01, 0.04),
        },
        "call_rate_grid": {
            "answer_loss_gap": [
                {
                    "target_call_rate": rate,
                    "metrics": {
                        "mean_policy_utility": _interval(0.002, 0.001, 0.003)
                    },
                }
                for rate in (0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0)
            ]
        },
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_replication_decision_has_three_frozen_outcomes(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# frozen protocol\n", encoding="utf-8")
    base = _report(sha256_file(protocol))

    report = tmp_path / "pass.json"
    _write(report, base)
    passed = decide_proxy_replication(
        report=report,
        protocol=protocol,
        output_dir=tmp_path / "pass",
        expected_report_sha256=sha256_file(report),
        expected_protocol_sha256=sha256_file(protocol),
        code_revision="test-code",
    )
    assert passed["decision"] == "replicated_alignment"
    assert all(row["passed"] for row in passed["conditions"].values())
    assert (tmp_path / "pass/decision.complete.json").is_file()

    partial_payload = copy.deepcopy(base)
    partial_payload["top_one"]["entropy_reduction"]["metrics"][
        "mean_task_gain"
    ] = _interval(0.04, 0.03, 0.05)
    partial_report = tmp_path / "partial.json"
    _write(partial_report, partial_payload)
    partial = decide_proxy_replication(
        report=partial_report,
        protocol=protocol,
        output_dir=tmp_path / "partial",
        code_revision="test-code",
    )
    assert partial["decision"] == "partial_alignment"

    failed_payload = copy.deepcopy(base)
    failed_payload["correlations"]["answer_loss_gap"]["spearman"] = _interval(
        0.01, -0.01, 0.03
    )
    failed_report = tmp_path / "failed.json"
    _write(failed_report, failed_payload)
    failed = decide_proxy_replication(
        report=failed_report,
        protocol=protocol,
        output_dir=tmp_path / "failed",
        code_revision="test-code",
    )
    assert failed["decision"] == "non_replication"


def test_replication_decision_rejects_protected_outcome_use(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# frozen protocol\n", encoding="utf-8")
    payload = _report(sha256_file(protocol))
    payload["outcome_use"]["protected_role_inputs_used"] = True
    report = tmp_path / "report.json"
    _write(report, payload)
    try:
        decide_proxy_replication(
            report=report,
            protocol=protocol,
            output_dir=tmp_path / "decision",
            code_revision="test-code",
        )
    except ValueError as exc:
        assert "forbidden outcome use" in str(exc)
    else:
        raise AssertionError("protected outcome use was accepted")


def test_docvqa_replication_decision_runner_is_hash_locked() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/run_docvqa_proxy_replication_decision.sh").read_text()
    assert "analysis/audit.complete.json" in runner
    assert "proxy-to-outcome-cross-domain-protocol-v1.md" in runner
    assert "proxy-replication-decision-implementation-v1.md" in runner
    assert "f800edfdb516caf128e0036d824130dc" in runner
    assert "61bbcd5392eceb65837d95ffc25c23f8" in runner
    assert "--expected-report-sha256" in runner
    assert "--expected-protocol-sha256" in runner
    assert "selection.score_threshold_selected" in runner
    assert "selection.call_rate_selected" in runner
    assert "selection.protected_outcome_used" in runner
