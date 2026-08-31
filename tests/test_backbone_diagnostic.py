from __future__ import annotations

import json
from pathlib import Path

import pytest

from beyond_entropy.backbone_diagnostic import (
    select_source_disjoint_manifest,
    sha256_file,
)


def _write_manifest(path: Path, *, target_suffix: str = "") -> None:
    rows = []
    for source_index in range(8):
        for state_index in range(3):
            rows.append(
                {
                    "source_id": f"source-{source_index:02d}",
                    "state_id": f"state-{source_index:02d}-{state_index:02d}",
                    "image_path": f"images/{source_index:02d}.png",
                    "question": f"question {source_index} {state_index}",
                    "target": {"answers": [f"answer-{target_suffix}-{state_index}"]},
                }
            )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _selected_ids(path: Path) -> list[tuple[str, str]]:
    return [
        (row["source_id"], row["state_id"])
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    ]


def test_source_disjoint_selection_is_deterministic_and_target_blind(
    tmp_path: Path,
) -> None:
    manifest_a = tmp_path / "manifest-a.jsonl"
    manifest_b = tmp_path / "manifest-b.jsonl"
    _write_manifest(manifest_a, target_suffix="a")
    _write_manifest(manifest_b, target_suffix="different")
    output_a = tmp_path / "selected-a.jsonl"
    output_b = tmp_path / "selected-b.jsonl"
    result_a = select_source_disjoint_manifest(
        manifest=manifest_a,
        output=output_a,
        report=tmp_path / "report-a.json",
        expected_manifest_sha256=sha256_file(manifest_a),
        source_count=5,
        namespace="fixture",
        seed=17,
        code_revision="test-code",
    )
    result_b = select_source_disjoint_manifest(
        manifest=manifest_b,
        output=output_b,
        report=tmp_path / "report-b.json",
        expected_manifest_sha256=sha256_file(manifest_b),
        source_count=5,
        namespace="fixture",
        seed=17,
        code_revision="test-code",
    )
    assert _selected_ids(output_a) == _selected_ids(output_b)
    assert len(_selected_ids(output_a)) == 5
    assert len({source_id for source_id, _ in _selected_ids(output_a)}) == 5
    assert result_a["selection"]["outcomes_used_for_ranking"] is False
    assert result_b["selection"]["labels_used_for_ranking"] is False


def test_source_disjoint_selection_rejects_hash_and_overwrite(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    output = tmp_path / "selected.jsonl"
    report = tmp_path / "report.json"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        select_source_disjoint_manifest(
            manifest=manifest,
            output=output,
            report=report,
            expected_manifest_sha256="0" * 64,
            source_count=2,
            namespace="fixture",
            seed=1,
            code_revision="test-code",
        )
    select_source_disjoint_manifest(
        manifest=manifest,
        output=output,
        report=report,
        expected_manifest_sha256=sha256_file(manifest),
        source_count=2,
        namespace="fixture",
        seed=1,
        code_revision="test-code",
    )
    with pytest.raises(FileExistsError):
        select_source_disjoint_manifest(
            manifest=manifest,
            output=output,
            report=report,
            expected_manifest_sha256=sha256_file(manifest),
            source_count=2,
            namespace="fixture",
            seed=1,
            code_revision="test-code",
        )
