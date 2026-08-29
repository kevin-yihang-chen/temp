from __future__ import annotations

import hashlib
import json

import pytest

from scripts.calibrate_docvqa_train_factorized_v2 import _write_bundle_exclusive


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_docvqa_calibration_bundle_is_exclusive_and_self_auditing(tmp_path):
    output = tmp_path / "calibration"
    audit = _write_bundle_exclusive(
        output,
        {"selection_status": "no_non_degenerate_safe_threshold"},
        {"threshold": None},
        {"passed": True, "formal_outcomes_used": False},
    )
    assert sorted(path.name for path in output.iterdir()) == [
        "calibration.audit.json",
        "calibration.json",
        "model.json",
    ]
    assert audit["calibration_sha256"] == _sha256(output / "calibration.json")
    assert audit["model_sha256"] == _sha256(output / "model.json")
    stored = json.loads((output / "calibration.audit.json").read_text())
    assert stored == audit
    with pytest.raises(FileExistsError, match="already exists"):
        _write_bundle_exclusive(output, {}, {}, {})
