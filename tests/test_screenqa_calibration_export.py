from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from beyond_entropy.risk_control import AcquisitionCalibrationRow
from scripts.calibrate_screenqa_fixed_sequence import (
    FAILURE,
    SUCCESS,
    build_result_audit,
    calibrate_rows,
)
from scripts.export_screenqa_calibration_manifest import (
    PROTOCOL_SHA256,
    verify_candidate,
)
from scripts.verify_screenqa_calibration_manifest import verify_manifest
from scripts.verify_screenqa_calibration_result import verify_result
from scripts.export_screenqa_formal_manifest import verify_formal_gate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_bundle(root: Path, *, threshold=None, frozen: bool = True) -> Path:
    root.mkdir()
    report_path = root / "selected-ranker-report.json"
    report_path.write_text("{}\n", encoding="utf-8")
    report_sha256 = _sha256(report_path)
    contract = {
        "method": "fixed_sequence_bounded_mean_kl_ltt_v1",
        "threshold_order": "strict_to_permissive_descending",
        "threshold_rate_weighting": "pooled_development_decisions",
        "target_pooled_development_call_rates": [
            0.005,
            0.01,
            0.015,
            0.02,
            0.03,
            0.05,
        ],
        "constraints": [
            {"kind": "induced_harm", "limit": 0.005},
            {"kind": "net_negative_call_mass", "limit": 0.02},
        ],
        "family_error": 0.05,
        "per_step_p_cutoff": 0.025,
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "calibration_sources": 1016,
        "calibration_decisions": 9951,
        "formal_sources": 1471,
        "formal_decisions": 14672,
    }
    model = {
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "feature_mode": "context-geometry",
        "seed": 20260831,
        "n_folds": 5,
        "lambda_cost": 0.05,
        "domains": ["screenqa"],
        "threshold": threshold,
        "development_oof_threshold": 0.25,
        "threshold_grid": [1.0, 0.5],
        "calibration_contract": contract,
        "candidate_selection": {
            "protocol_sha256": PROTOCOL_SHA256,
            "selected_feature_mode": "context-geometry",
            "ranker_report_sha256": report_sha256,
            "ranker_rollouts_sha256": "1" * 64,
            "ranker_fit_code_revision": "2" * 40,
            "raw_model_sha256": "3" * 64,
            "ranker_training_outcomes_used": True,
            "calibration_outcomes_used": False,
            "formal_outcomes_used": False,
            "reserve_outcomes_used": False,
        },
    }
    model_path = root / "model.json"
    model_path.write_text(json.dumps(model, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "candidate_frozen": frozen,
        "semantic_escalation_required": False,
        "calibration_outcomes_opened": False,
        "formal_outcomes_opened": False,
        "reserve_outcomes_opened": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "selected_feature_mode": "context-geometry",
        "selected_model_sha256": _sha256(model_path),
        "selected_ranker_report_sha256": report_sha256,
        "threshold_grid": [1.0, 0.5],
    }
    audit_path = root / "candidate.audit.json"
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")
    with (root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted((model_path, report_path, audit_path)):
            handle.write(f"{_sha256(path)}  {path.name}\n")
    return root


def test_screenqa_calibration_export_accepts_only_frozen_candidate(tmp_path):
    candidate = verify_candidate(_candidate_bundle(tmp_path / "candidate"))
    assert candidate["feature_mode"] == "context-geometry"
    assert candidate["threshold_grid"] == [1.0, 0.5]


def test_screenqa_calibration_export_rejects_development_execution_threshold(tmp_path):
    with pytest.raises(ValueError, match="execution threshold must be unset"):
        verify_candidate(_candidate_bundle(tmp_path / "candidate", threshold=0.25))


def test_screenqa_calibration_export_rejects_unfrozen_candidate(tmp_path):
    with pytest.raises(ValueError, match="not safely frozen"):
        verify_candidate(_candidate_bundle(tmp_path / "candidate", frozen=False))


def test_screenqa_calibration_export_slurm_contract_is_sealed_and_notified():
    root = Path(__file__).resolve().parents[1]
    worker = (root / "scripts/slurm_screenqa_calibration_manifest.sh").read_text()
    submit = (root / "scripts/submit_screenqa_calibration_manifest.sh").read_text()
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "BE_SCREENQA_CANDIDATE_BUNDLE_SHA256" in worker
    assert "--formal-output-dir" in worker
    assert "--reserve-output-dir" in worker
    assert "--untouched-output-dir" in worker
    assert "verify_candidate" in submit
    assert "tracked worktree must be clean" in submit
    assert "--mail-type=ALL" in submit


def _calibration_manifest_bundle(root: Path, candidate_dir: Path) -> tuple[str, str]:
    root.mkdir()
    candidate = verify_candidate(candidate_dir)
    manifest = root / "manifest.jsonl"
    manifest.write_text('{"state_id":"one"}\n', encoding="utf-8")
    manifest_sha256 = _sha256(manifest)
    audit = {
        "passed": True,
        "scientific_status": (
            "only frozen risk-calibration labels opened after sole candidate freeze"
        ),
        "ranker_training_outcomes_previously_used": True,
        "risk_calibration_opened": True,
        "formal_test_opened": False,
        "reserve_opened": False,
        "untouched_opened": False,
        "official_validation_test_opened": False,
        "annotation_objects_deserialized": 9951,
        "selected_rico_images": 4001,
        "selected_source_components": 1016,
        "unselected_annotation_objects_deserialized": 0,
        "candidate": candidate,
        "manifest": {
            "manifest_sha256": manifest_sha256,
            "count": 9951,
            "scorer": "screenqa",
        },
        "export_provenance": {
            "selection_metadata": {
                "role": "risk_calibration",
                "candidate": candidate,
                "candidate_frozen_before_annotation_deserialization": True,
                "formal_outcomes_opened": False,
                "reserve_outcomes_opened": False,
                "untouched_outcomes_opened": False,
            }
        },
    }
    audit_path = root / "manifest.audit.json"
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")
    with (root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in (manifest, audit_path):
            handle.write(f"{_sha256(path)}  {path.name}\n")
    return manifest_sha256, _sha256(audit_path)


def test_screenqa_calibration_manifest_verifier_binds_candidate_and_seals_formal(
    tmp_path,
):
    candidate_dir = _candidate_bundle(tmp_path / "candidate")
    manifest_sha256, audit_sha256 = _calibration_manifest_bundle(
        tmp_path / "manifest", candidate_dir
    )
    candidate = verify_candidate(candidate_dir)
    result = verify_manifest(
        tmp_path / "manifest",
        candidate_dir=candidate_dir,
        expected_candidate_bundle_sha256=candidate["bundle_sha256"],
        expected_manifest_sha256=manifest_sha256,
        expected_audit_sha256=audit_sha256,
    )
    assert result["states"] == 9951
    assert "formal_test" in result["sealed_roles"]


def test_screenqa_calibration_bank_contract_is_sharded_resumable_and_notified():
    root = Path(__file__).resolve().parents[1]
    worker = (root / "scripts/slurm_screenqa_calibration_bank.sh").read_text()
    merge = (root / "scripts/slurm_screenqa_calibration_bank_merge.sh").read_text()
    submit = (root / "scripts/submit_screenqa_calibration_bank.sh").read_text()
    assert "#SBATCH --array=0-3" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "--resume" in worker
    assert "resume changed completed calibration rollout bytes" in worker
    assert "outcomes may calibrate only the frozen threshold sequence" in worker
    assert "BE_SCREENQA_CANDIDATE_BUNDLE_SHA256" in worker
    assert "--require-resume-audit" in merge
    assert "--bootstrap-resamples 5000" in merge
    assert "--dependency=\"afterok:${array_job_id}\"" in submit
    assert "--mail-type=ALL" in submit
    assert "tracked worktree must be clean" in submit


def test_screenqa_calibration_rollout_verifier_contract_is_complete():
    root = Path(__file__).resolve().parents[1]
    content = (root / "scripts/verify_screenqa_calibration_rollouts.py").read_text()
    assert "EXPECTED_STATES = 9951" in content
    assert "EXPECTED_RECORDS = 49755" in content
    assert "EXPECTED_SOURCES = 1016" in content
    assert "resume_audit_required" in content
    assert "outcomes may calibrate only" in content
    assert '"formal_outcomes_opened": False' in content


def _risk_rows(*, gain: float) -> list[AcquisitionCalibrationRow]:
    return [
        AcquisitionCalibrationRow(
            source_id=f"screenqa:source-{index}",
            score=0.6,
            gain=gain,
            tool_cost=1.0,
        )
        for index in range(1016)
    ]


def test_screenqa_fixed_sequence_success_alone_allows_formal(tmp_path):
    candidate_dir = _candidate_bundle(tmp_path / "candidate")
    candidate = json.loads((candidate_dir / "model.json").read_text())
    calibration, model = calibrate_rows(
        candidate,
        _risk_rows(gain=1.0),
        expected_decisions=1016,
    )
    assert calibration["selection_status"] == SUCCESS
    assert model["threshold"] == 0.5
    audit = build_result_audit(
        calibration,
        input_hashes={"candidate": "1" * 64},
        code_revision="2" * 40,
    )
    assert audit["formal_allowed"] is True
    assert audit["formal_stop_required"] is False


def test_screenqa_fixed_sequence_failure_keeps_formal_sealed(tmp_path):
    candidate_dir = _candidate_bundle(tmp_path / "candidate")
    candidate = json.loads((candidate_dir / "model.json").read_text())
    calibration, model = calibrate_rows(
        candidate,
        _risk_rows(gain=-1.0),
        expected_decisions=1016,
    )
    assert calibration["selection_status"] == FAILURE
    assert model["threshold"] is None
    audit = build_result_audit(
        calibration,
        input_hashes={"candidate": "1" * 64},
        code_revision="2" * 40,
    )
    assert audit["formal_allowed"] is False
    assert audit["formal_stop_required"] is True


def _calibration_result_bundle(
    root: Path,
    candidate_dir: Path,
    *,
    gain: float,
) -> Path:
    root.mkdir()
    candidate = json.loads((candidate_dir / "model.json").read_text())
    candidate_info = verify_candidate(candidate_dir)
    input_hashes = {
        "candidate_model_sha256": candidate_info["model_sha256"],
        "candidate_audit_sha256": candidate_info["audit_sha256"],
        "candidate_bundle_sha256": candidate_info["bundle_sha256"],
        "candidate_report_sha256": candidate_info["report_sha256"],
        "manifest_sha256": "4" * 64,
        "manifest_audit_sha256": "5" * 64,
        "rollouts_sha256": "6" * 64,
        "merge_audit_sha256": "7" * 64,
        "rollout_input_audit_sha256": "8" * 64,
    }
    calibration, model = calibrate_rows(
        candidate,
        _risk_rows(gain=gain),
        expected_decisions=1016,
        run_provenance={
            "code_revision": "9" * 40,
            **input_hashes,
        },
    )
    calibration["n_decisions"] = 9951
    audit = build_result_audit(
        calibration,
        input_hashes=input_hashes,
        code_revision="9" * 40,
    )
    for name, payload in (
        ("calibration.json", calibration),
        ("model.json", model),
        ("calibration.audit.json", audit),
    ):
        (root / name).write_text(
            json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    with (root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(root.iterdir()):
            if path.name != "SHA256SUMS":
                handle.write(f"{_sha256(path)}  {path.name}\n")
    return root


@pytest.mark.parametrize(
    ("gain", "formal_allowed"),
    [(1.0, True), (-1.0, False)],
)
def test_screenqa_calibration_result_verifier_recomputes_formal_gate(
    tmp_path,
    gain,
    formal_allowed,
):
    candidate_dir = _candidate_bundle(tmp_path / "candidate")
    result_dir = _calibration_result_bundle(
        tmp_path / "result", candidate_dir, gain=gain
    )
    result = verify_result(result_dir, candidate_dir)
    assert result["formal_allowed"] is formal_allowed


def test_screenqa_calibration_slurm_contract_is_bound_and_notified():
    root = Path(__file__).resolve().parents[1]
    worker = (root / "scripts/slurm_screenqa_calibrate.sh").read_text()
    submit = (root / "scripts/submit_screenqa_calibrate.sh").read_text()
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "BE_SCREENQA_CANDIDATE_BUNDLE_SHA256" in worker
    assert "BE_SCREENQA_CALIBRATION_MERGE_AUDIT_SHA256" in worker
    assert "calibrate_screenqa_fixed_sequence" in worker
    assert "verify_screenqa_calibration_result" in worker
    assert "tracked worktree must be clean" in submit
    assert "sealed ScreenQA role is already materialized" in submit
    assert "--mail-type=ALL" in submit


def test_screenqa_formal_gate_accepts_only_successful_recomputed_calibration(tmp_path):
    candidate_dir = _candidate_bundle(tmp_path / "candidate")
    success_dir = _calibration_result_bundle(
        tmp_path / "success", candidate_dir, gain=1.0
    )
    assert verify_formal_gate(candidate_dir, success_dir)["formal_allowed"] is True

    failed_candidate = _candidate_bundle(tmp_path / "failed-candidate")
    failure_dir = _calibration_result_bundle(
        tmp_path / "failure", failed_candidate, gain=-1.0
    )
    with pytest.raises(ValueError, match="blocked by risk calibration"):
        verify_formal_gate(failed_candidate, failure_dir)


def test_screenqa_formal_export_contract_preserves_sealed_roles():
    root = Path(__file__).resolve().parents[1]
    content = (root / "scripts/export_screenqa_formal_manifest.py").read_text()
    assert "EXPECTED_IMAGES = 6000" in content
    assert "EXPECTED_QA_ROWS = 14672" in content
    assert "EXPECTED_SOURCES = 1471" in content
    assert "verify_formal_gate" in content
    assert "This gate must run before the annotation file is opened or hashed" in content
    assert '"reserve_opened": False' in content
    assert '"untouched_opened": False' in content
    assert '"official_validation_test_opened": False' in content


def test_screenqa_formal_export_slurm_gate_is_double_checked_and_notified():
    root = Path(__file__).resolve().parents[1]
    worker = (root / "scripts/slurm_screenqa_formal_manifest.sh").read_text()
    submit = (root / "scripts/submit_screenqa_formal_manifest.sh").read_text()
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "BE_SCREENQA_CANDIDATE_BUNDLE_SHA256" in worker
    assert "BE_SCREENQA_CALIBRATION_BUNDLE_SHA256" in worker
    assert "verify_screenqa_calibration_result" in worker
    assert "export_screenqa_formal_manifest" in worker
    assert "verify_formal_gate" in submit
    assert "tracked worktree must be clean" in submit
    assert "--mail-type=ALL" in submit
