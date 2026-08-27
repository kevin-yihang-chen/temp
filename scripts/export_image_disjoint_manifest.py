from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_manifest(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"manifest row {line_number} is not an object")
            rows.append(value)
    state_ids = [str(row["state_id"]) for row in rows]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError(f"manifest contains duplicate state IDs: {path}")
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Exclude every image used by another manifest")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = read_manifest(args.source)
    excluded = read_manifest(args.exclude)
    excluded_images = {str(row["image_id"]) for row in excluded}
    selected = [
        row for row in source if str(row["image_id"]) not in excluded_images
    ]
    if not selected:
        raise ValueError("image-disjoint manifest would be empty")
    selected_images = {str(row["image_id"]) for row in selected}
    if selected_images & excluded_images:
        raise RuntimeError("output manifest retains excluded images")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "source_sha256": sha256(args.source),
                "exclude": str(args.exclude.resolve()),
                "exclude_sha256": sha256(args.exclude),
                "output": str(args.output.resolve()),
                "output_sha256": sha256(args.output),
                "selected_states": len(selected),
                "selected_images": len(selected_images),
                "excluded_images": len(excluded_images),
                "selected_strata": Counter(
                    str(row.get("stratum", "unknown")) for row in selected
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
