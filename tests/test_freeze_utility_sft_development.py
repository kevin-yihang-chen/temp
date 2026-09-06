from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from test_utility_sft import make_samples


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/freeze_utility_sft_development.py"
SPEC = importlib.util.spec_from_file_location("freeze_utility_sft_development", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_dataset(tmp_path, benchmark, role, index, *, source_id=None, rgb_hash=None):
    state_id = f"{benchmark}-{role}-{index}"
    sample = make_samples(
        state_id=state_id, image_id=f"image-{state_id}",
        source_id=source_id or f"source-{state_id}", benchmark=benchmark, role=role,
        rgb_hash=rgb_hash or f"{index+1:064x}",
    )[0]
    path = tmp_path / f"{benchmark}-{role}.json"
    path.write_text(json.dumps({
        "schema": "utility_sft_dataset_v1", "role": role, "benchmark": benchmark,
        "aggregation": "single", "samples": [sample.to_dict()],
    }))
    return str(path)


def test_freeze_six_development_roles_and_refuse_overwrite(tmp_path):
    paths = [write_dataset(tmp_path, b, r, i) for i, (b, r) in enumerate(
        (b, r) for b in MODULE.BENCHMARKS for r in MODULE.ROLES
    )]
    output = tmp_path / "bundle.json"
    report = MODULE.freeze_development_dataset(paths, str(output))
    assert report["states"] == 6
    assert report["test_data_present"] is False
    assert report["formal_test_eligible"] is False
    assert report["split_audit"]["passed"]
    assert len(report["inventory"]) == 6
    with pytest.raises(FileExistsError):
        MODULE.freeze_development_dataset(paths, str(output))


def test_freeze_rejects_missing_role_and_global_rgb_overlap(tmp_path):
    pairs = [(b, r) for b in MODULE.BENCHMARKS for r in MODULE.ROLES]
    paths = [write_dataset(tmp_path, b, r, i) for i, (b, r) in enumerate(pairs)]
    with pytest.raises(ValueError, match="exactly six"):
        MODULE.freeze_development_dataset(paths[:-1], str(tmp_path / "missing.json"))

    # Identical RGB across benchmarks and different roles is leakage even when
    # opaque image/source IDs differ.
    paths[0] = write_dataset(tmp_path, pairs[0][0], pairs[0][1], 100, rgb_hash="f"*64)
    paths[-1] = write_dataset(tmp_path, pairs[-1][0], pairs[-1][1], 101, rgb_hash="f"*64)
    with pytest.raises(ValueError, match="RGB split leakage"):
        MODULE.freeze_development_dataset(paths, str(tmp_path / "overlap.json"))
