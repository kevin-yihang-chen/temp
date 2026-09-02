#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote
from urllib.request import Request, urlopen

from beyond_entropy.refocus_chart_audit import audit_chartqa_train_lineage


TRAIN_TREE_COMPONENTS = ("ChartQA Dataset", "train", "png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match a train-only Refocus_Chart manifest against a pinned original "
            "ChartQA train/png Git tree without traversing validation or test."
        )
    )
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--root-tree-sha", required=True)
    parser.add_argument(
        "--api-base",
        default="https://api.github.com/repos/vis-nlp/ChartQA/git/trees",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def _fetch_tree(
    api_base: str,
    tree_sha: str,
    *,
    recursive: bool,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    suffix = "?recursive=1" if recursive else ""
    url = f"{api_base.rstrip('/')}/{quote(tree_sha, safe='')}{suffix}"
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "beyond-entropy-refocus-lineage-audit/1",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        payload = json.load(response)
        response_metadata = {
            "url": url,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }
    if not isinstance(payload, dict) or payload.get("sha") != tree_sha:
        raise ValueError(
            f"GitHub tree response does not match requested SHA {tree_sha}"
        )
    tree = payload.get("tree")
    if not isinstance(tree, list):
        raise ValueError("GitHub tree response must contain a tree list")
    if payload.get("truncated") is True:
        raise ValueError(f"GitHub tree response is truncated for {tree_sha}")
    return payload, response_metadata


def _child_tree_sha(payload: Mapping[str, Any], component: str) -> str:
    tree = payload.get("tree")
    if not isinstance(tree, list):
        raise ValueError("GitHub tree response must contain a tree list")
    matches = [
        entry
        for entry in tree
        if isinstance(entry, Mapping)
        and entry.get("path") == component
        and entry.get("type") == "tree"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one child tree named {component!r}")
    sha = str(matches[0].get("sha", ""))
    if not sha:
        raise ValueError(f"child tree {component!r} has no SHA")
    return sha


def _load_train_row_ids(path: Path) -> tuple[list[str], str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "refocus_chart_train_metadata_report_v1":
        raise ValueError("train metadata report schema mismatch")
    if report.get("test_accessed") is not False:
        raise ValueError("lineage input must explicitly attest test_accessed=false")
    train = report.get("train")
    if not isinstance(train, Mapping):
        raise ValueError("train metadata report is missing train audit")
    rows = train.get("row_manifests")
    if not isinstance(rows, list):
        raise ValueError("train audit is missing row manifests")
    row_ids = [str(row["row_id"]) for row in rows if isinstance(row, Mapping)]
    if len(row_ids) != len(rows):
        raise ValueError("every train manifest row must be a mapping with row_id")
    manifest_sha256 = str(train.get("manifest_sha256", ""))
    return row_ids, manifest_sha256


def main() -> None:
    args = parse_args()
    row_ids, manifest_sha256 = _load_train_row_ids(args.train_report)

    request_metadata: list[dict[str, Any]] = []
    current_sha = args.root_tree_sha
    resolved_path: list[dict[str, str]] = []
    for component in TRAIN_TREE_COMPONENTS:
        payload, request = _fetch_tree(
            args.api_base,
            current_sha,
            recursive=False,
            timeout_seconds=args.timeout_seconds,
        )
        request_metadata.append(request)
        current_sha = _child_tree_sha(payload, component)
        resolved_path.append({"component": component, "tree_sha": current_sha})

    train_png_payload, request = _fetch_tree(
        args.api_base,
        current_sha,
        recursive=True,
        timeout_seconds=args.timeout_seconds,
    )
    request_metadata.append(request)
    entries = train_png_payload["tree"]
    lineage = audit_chartqa_train_lineage(
        row_ids,
        train_png_entries=entries,
        repository="vis-nlp/ChartQA",
        root_tree_sha=args.root_tree_sha,
        train_png_tree_sha=current_sha,
    )
    report = {
        **lineage,
        "refocus_train_manifest_sha256": manifest_sha256,
        "resolved_train_tree_path": resolved_path,
        "requests": request_metadata,
        "network_scope": (
            "non-recursive ancestor trees plus recursive ChartQA Dataset/train/png; "
            "no validation/test subtree contents"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": report["decision"],
                "matched_unique_row_ids": report["matched_unique_row_ids"],
                "missing_unique_row_ids": report["missing_unique_row_ids"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
