from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_screenqa_semantic_activation import verify_activation, write_audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, object]:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_bytes(b"{}\n" * 72_555)
    rollouts_sha256 = _sha256(rollouts)
    input_audit = tmp_path / "ranker-rollouts.audit.json"
    input_audit.write_text(
        json.dumps(
            {
                "passed": True,
                "rollouts_sha256": rollouts_sha256,
                "records": 72_555,
                "states": 14_511,
                "source_components": 1_510,
                "calibration_outcomes_opened": False,
                "formal_outcomes_opened": False,
                "reserve_outcomes_opened": False,
            }
        ),
        encoding="utf-8",
    )
    v1_protocol = tmp_path / "v1.md"
    v1_protocol.write_text("v1 frozen\n", encoding="utf-8")
    v2_protocol = tmp_path / "v2.md"
    v2_protocol.write_text("v2 frozen\n", encoding="utf-8")
    v1_sha256 = _sha256(v1_protocol)
    candidate_dir = tmp_path / "candidate-v1"
    candidate_dir.mkdir()
    candidate_audit = candidate_dir / "candidate.audit.json"
    candidate_audit.write_text(
        json.dumps(
            {
                "protocol_applied": True,
                "protocol_sha256": v1_sha256,
                "registered_feature_modes": [
                    "context-geometry",
                    "spatial-context-geometry",
                ],
                "selection_reason": "no_registered_candidate_is_eligible",
                "candidate_frozen": False,
                "semantic_escalation_required": True,
                "calibration_outcomes_opened": False,
                "formal_outcomes_opened": False,
                "reserve_outcomes_opened": False,
                "candidates": [
                    {
                        "feature_mode": mode,
                        "eligible": False,
                        "tail_selection_status": "no_non_degenerate_safe_threshold",
                        "code_revision": "1" * 40,
                        "rollouts_sha256": rollouts_sha256,
                    }
                    for mode in ["context-geometry", "spatial-context-geometry"]
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    sums = candidate_dir / "SHA256SUMS"
    sums.write_text(
        f"{_sha256(candidate_audit)}  candidate.audit.json\n", encoding="utf-8"
    )
    return {
        "candidate_dir": candidate_dir,
        "expected_candidate_audit_sha256": _sha256(candidate_audit),
        "ranker_rollouts": rollouts,
        "expected_ranker_rollouts_sha256": rollouts_sha256,
        "ranker_input_audit": input_audit,
        "expected_ranker_input_audit_sha256": _sha256(input_audit),
        "v1_protocol": v1_protocol,
        "expected_v1_protocol_sha256": v1_sha256,
        "v2_protocol": v2_protocol,
        "expected_v2_protocol_sha256": _sha256(v2_protocol),
        "sealed_output_dirs": [tmp_path / "calibration", tmp_path / "formal"],
        "expected_code_revision": "2" * 40,
    }


def test_screenqa_semantic_activation_accepts_only_double_failure(tmp_path):
    args = _fixture(tmp_path)
    audit = verify_activation(**args)
    assert audit["passed"] is True
    assert audit["semantic_escalation_activated"] is True
    assert audit["failed_feature_modes"] == [
        "context-geometry",
        "spatial-context-geometry",
    ]
    assert audit["calibration_outcomes_opened"] is False


def test_screenqa_semantic_activation_rejects_an_eligible_v1_candidate(tmp_path):
    args = _fixture(tmp_path)
    path = Path(args["candidate_dir"]) / "candidate.audit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidates"][1]["eligible"] = True
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    sums = path.parent / "SHA256SUMS"
    sums.write_text(f"{_sha256(path)}  candidate.audit.json\n", encoding="utf-8")
    args["expected_candidate_audit_sha256"] = _sha256(path)
    with pytest.raises(ValueError, match="remains eligible"):
        verify_activation(**args)


def test_screenqa_semantic_activation_rejects_materialized_protected_output(tmp_path):
    args = _fixture(tmp_path)
    formal = Path(args["sealed_output_dirs"][1])
    formal.mkdir()
    (formal / "opened.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already materialized"):
        verify_activation(**args)


def test_screenqa_semantic_activation_rejects_input_hash_drift(tmp_path):
    args = _fixture(tmp_path)
    Path(args["ranker_input_audit"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input audit SHA-256 mismatch"):
        verify_activation(**args)


def test_screenqa_semantic_activation_audit_resume_is_exact(tmp_path):
    audit = verify_activation(**_fixture(tmp_path))
    output = tmp_path / "activation" / "audit.json"
    write_audit(output, audit, resume=False)
    write_audit(output, audit, resume=True)
    changed = dict(audit)
    changed["passed"] = False
    with pytest.raises(FileExistsError, match="already exists or drifted"):
        write_audit(output, changed, resume=True)
