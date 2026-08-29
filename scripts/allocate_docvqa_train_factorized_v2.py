from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.docvqa_train_allocation import (
    DATASET_REVISION,
    PROTOCOL_SHA256,
    build_allocation_audit,
    build_allocation_document,
    discover_prior_manifests,
    load_docvqa_source_images,
    load_prior_identities,
    sha256_file,
)


def _write_frozen_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not resume:
            raise FileExistsError(f"frozen output already exists: {path}")
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"existing frozen output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def _code_revision(repo_dir: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the outcome-independent DocVQA-train factorized-v2 identities"
        )
    )
    parser.add_argument("--parquet-file", type=Path, action="append", required=True)
    parser.add_argument("--prior-manifest-root", type=Path, action="append", default=[])
    parser.add_argument("--prior-manifest", type=Path, action="append", default=[])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--allocation-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    protocol = args.protocol.resolve()
    if not protocol.is_file() or sha256_file(protocol) != PROTOCOL_SHA256:
        raise ValueError("DocVQA preregistration SHA-256 mismatch")
    parquet_paths = [path.resolve() for path in args.parquet_file]
    if any(not path.is_file() for path in parquet_paths):
        raise FileNotFoundError("one or more DocVQA train Parquet shards do not exist")
    prior_manifests = discover_prior_manifests(
        args.prior_manifest_root,
        args.prior_manifest,
    )
    prior_image_ids, prior_docvqa_groups, prior_records = load_prior_identities(
        prior_manifests,
        verify_images=True,
    )

    parquet_sha256: list[str] = []
    for index, path in enumerate(parquet_paths, start=1):
        parquet_sha256.append(sha256_file(path))
        print(f"hashed source Parquet shards: {index}/{len(parquet_paths)}", flush=True)

    source_images, row_count = load_docvqa_source_images(parquet_paths)

    repo_dir = Path(__file__).resolve().parents[1]
    allocation_document = build_allocation_document(
        source_images,
        excluded_image_ids=prior_image_ids,
        excluded_source_group_ids=prior_docvqa_groups,
        prior_banks=prior_records,
        parquet_files=parquet_paths,
        parquet_sha256=parquet_sha256,
        row_count=row_count,
        protocol_path=protocol,
        code_revision=_code_revision(repo_dir),
    )
    allocation_path = args.allocation_output.resolve()
    _write_frozen_json(allocation_path, allocation_document, resume=args.resume)
    allocation_sha256 = sha256_file(allocation_path)
    audit_document = build_allocation_audit(
        allocation_document,
        allocation_path=allocation_path,
        allocation_sha256=allocation_sha256,
        excluded_image_ids=prior_image_ids,
        excluded_source_group_ids=prior_docvqa_groups,
    )
    _write_frozen_json(args.audit_output.resolve(), audit_document, resume=args.resume)
    print(json.dumps(audit_document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
