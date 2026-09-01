#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ID = "lmms-lab-encoder/DocVQA"
REPO_TYPE = "dataset"
REVISION = "539088ef8a8ada01ac8e2e6d4e372586748a265e"
ALLOW_PATTERN = "InfographicVQA/train-*.parquet"
EXPECTED_FILE_COUNT = 24
EXPECTED_TOTAL_BYTES = 1_981_251_656
EXPECTED_NAMES = {
    f"InfographicVQA/train-{index:05d}-of-00024.parquet"
    for index in range(EXPECTED_FILE_COUNT)
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_download(
    root: Path,
    *,
    expected_names: set[str] = EXPECTED_NAMES,
    expected_total_bytes: int = EXPECTED_TOTAL_BYTES,
) -> list[dict[str, Any]]:
    parquet = sorted(path for path in root.rglob("*.parquet") if path.is_file())
    relative = {path.relative_to(root).as_posix() for path in parquet}
    if relative != expected_names:
        missing = sorted(expected_names - relative)
        extra = sorted(relative - expected_names)
        raise ValueError(
            f"InfographicVQA train file set differs; missing={missing[:3]} extra={extra[:3]}"
        )
    rows: list[dict[str, Any]] = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in parquet
    ]
    if sum(int(row["bytes"]) for row in rows) != expected_total_bytes:
        raise ValueError("InfographicVQA train aggregate byte size differs")
    if any(
        "/validation-" in str(row["path"]) or "/test-" in str(row["path"])
        for row in rows
    ):
        raise ValueError("sealed InfographicVQA split was downloaded")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the pinned InfographicVQA train-only transport mirror"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.expanduser().resolve()
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite InfographicVQA train: {destination}")
    partial.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=REVISION,
        allow_patterns=[ALLOW_PATTERN],
        local_dir=partial,
    )
    files = _verify_download(partial)
    manifest = {
        "schema": "infographicvqa_train_transport_manifest_v1",
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "revision": REVISION,
        "allow_pattern": ALLOW_PATTERN,
        "file_count": len(files),
        "aggregate_bytes": sum(int(row["bytes"]) for row in files),
        "validation_files_downloaded": False,
        "test_files_downloaded": False,
        "questions_or_answers_read": False,
        "files": files,
    }
    manifest_path = partial / "download-manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    final_manifest = destination / "download-manifest.json"
    print(
        json.dumps(
            {
                "output_dir": str(destination),
                "manifest": str(final_manifest),
                "manifest_sha256": _sha256(final_manifest),
                "file_count": len(files),
                "aggregate_bytes": manifest["aggregate_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
