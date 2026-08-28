from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from beyond_entropy.manifest_export import image_digest
from beyond_entropy.vtool_adapter import vtool_identity_join_key


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicate_rows(keys: Iterable[str]) -> int:
    return sum(count - 1 for count in Counter(keys).values() if count > 1)


def _manifest_keys(path: Path) -> list[str]:
    keys: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                keys.append(
                    vtool_identity_join_key(
                        str(value["image_id"]),
                        str(value["question"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid manifest row {path}:{line_number}: {exc}") from exc
    return keys


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exact VTool parquet overlap with frozen manifests",
    )
    parser.add_argument("--vtool-parquet", type=Path, required=True)
    parser.add_argument("--vtool-dataset-revision", required=True)
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    from PIL import Image  # type: ignore[import-untyped]

    args = parse_args()
    manifest_paths: dict[str, Path] = {}
    for specification in args.manifest:
        name, separator, raw_path = specification.partition("=")
        if not separator or not name or not raw_path or name in manifest_paths:
            raise ValueError(f"invalid or duplicate manifest specification: {specification!r}")
        manifest_paths[name] = Path(raw_path)

    table = parquet.read_table(
        args.vtool_parquet,
        columns=["id", "images", "extra_info"],
    )
    vtool_keys: list[str] = []
    invalid_image_rows: list[str] = []
    for row in table.to_pylist():
        images = row.get("images") or []
        if (
            len(images) != 1
            or not isinstance(images[0], Mapping)
            or not images[0].get("bytes")
        ):
            invalid_image_rows.append(str(row.get("id", "<missing-id>")))
            continue
        extra_info = row.get("extra_info")
        if not isinstance(extra_info, Mapping) or not extra_info.get("question"):
            raise ValueError(f"VTool row {row.get('id')!r} is missing extra_info.question")
        image = Image.open(io.BytesIO(images[0]["bytes"])).convert("RGB")
        vtool_keys.append(
            vtool_identity_join_key(
                image_digest(image),
                str(extra_info["question"]),
            )
        )

    vtool_counts = Counter(vtool_keys)
    manifests: dict[str, Any] = {}
    for name, path in sorted(manifest_paths.items()):
        keys = _manifest_keys(path)
        counts = Counter(keys)
        matching_keys = set(vtool_counts) & set(counts)
        manifests[name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "rows": len(keys),
            "unique_keys": len(counts),
            "duplicate_key_rows": _duplicate_rows(keys),
            "matched_vtool_unique_keys": len(matching_keys),
            "matched_vtool_rows": sum(vtool_counts[key] for key in matching_keys),
        }

    _write_json(
        {
            "scientific_status": "dataset identity and contamination audit",
            "join_key": "decoded RGB SHA-256 plus normalized question SHA-256",
            "vtool": {
                "parquet": str(args.vtool_parquet.resolve()),
                "parquet_sha256": _sha256(args.vtool_parquet),
                "dataset_revision": args.vtool_dataset_revision,
                "rows": table.num_rows,
                "valid_identity_rows": len(vtool_keys),
                "unique_keys": len(vtool_counts),
                "duplicate_key_rows": _duplicate_rows(vtool_keys),
                "invalid_image_rows": invalid_image_rows,
            },
            "manifests": manifests,
        },
        args.output,
    )
    print(json.dumps({"output": str(args.output), "rows": table.num_rows}))


if __name__ == "__main__":
    main()
