from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.download_infographicvqa_train import _verify_download


def test_verify_download_hashes_exact_expected_files(tmp_path: Path) -> None:
    first = tmp_path / "InfographicVQA" / "train-a.parquet"
    second = tmp_path / "InfographicVQA" / "train-b.parquet"
    first.parent.mkdir()
    first.write_bytes(b"abc")
    second.write_bytes(b"defg")
    rows = _verify_download(
        tmp_path,
        expected_names={
            "InfographicVQA/train-a.parquet",
            "InfographicVQA/train-b.parquet",
        },
        expected_total_bytes=7,
    )
    assert [row["path"] for row in rows] == [
        "InfographicVQA/train-a.parquet",
        "InfographicVQA/train-b.parquet",
    ]
    assert rows[0]["sha256"] == hashlib.sha256(b"abc").hexdigest()


def test_verify_download_rejects_extra_split(tmp_path: Path) -> None:
    train = tmp_path / "InfographicVQA" / "train-a.parquet"
    validation = tmp_path / "InfographicVQA" / "validation-a.parquet"
    train.parent.mkdir()
    train.write_bytes(b"abc")
    validation.write_bytes(b"def")
    with pytest.raises(ValueError, match="file set differs"):
        _verify_download(
            tmp_path,
            expected_names={"InfographicVQA/train-a.parquet"},
            expected_total_bytes=3,
        )
