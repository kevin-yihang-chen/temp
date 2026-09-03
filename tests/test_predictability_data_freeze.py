from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "freeze_predictability_data.py"
    spec = importlib.util.spec_from_file_location("freeze_predictability_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, *, role: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "state_id": f"state-{role}",
        "image_id": f"image-{role}",
        "source_id": f"source-{role}",
        "image_path": "image.png",
        "question": "what?",
        "target": "x",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_staging_resume_requires_all_three_roles(tmp_path: Path) -> None:
    module = _module()
    _write_manifest(tmp_path / "chartqa" / "train" / "manifest.jsonl", role="train")
    with pytest.raises(ValueError, match="partial staged benchmark"):
        module.summarize_existing_benchmark(tmp_path, "chartqa")


def test_staging_resume_reconstructs_complete_summary(tmp_path: Path) -> None:
    module = _module()
    for role in ("train", "validation", "test"):
        _write_manifest(tmp_path / "docvqa" / role / "manifest.jsonl", role=role)
    report = module.summarize_existing_benchmark(tmp_path, "docvqa")
    assert report is not None
    assert report["train"]["states"] == 1
    assert report["test"]["historically_opened"] is False
    assert all(report[role]["resumed_from_complete_staging"] for role in report)
