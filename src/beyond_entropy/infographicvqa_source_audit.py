from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any
from urllib.parse import urlparse


INFOVQA_TRAIN_ROWS = 23_946
INFOVQA_TRAIN_FILES = 24
INFOVQA_TRAIN_BYTES = 1_981_251_656
INFOVQA_PILOT_SOURCES = 512
SOURCE_NAMESPACE = "beyond-entropy-infographicvqa-train-source-v1"
PILOT_NAMESPACE = "beyond-entropy-infographicvqa-train-pilot-v1"
ALLOWED_COLUMNS = ("questionId", "image", "image_url", "data_split")
FORBIDDEN_COLUMNS = (
    "question",
    "answers",
    "answer_type",
    "operation/reasoning",
    "ocr",
)
EXPECTED_SCHEMA = {
    "questionId": "string",
    "question": "string",
    "answers": "list<item: string>",
    "answer_type": "list<item: string>",
    "image": "struct<bytes: binary, path: string>",
    "image_url": "string",
    "operation/reasoning": "list<item: string>",
    "ocr": "string",
    "data_split": "string",
}


class _DisjointSet:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def normalize_hostname(raw_url: str) -> str | None:
    value = str(raw_url).strip()
    if not value:
        return None
    parsed = urlparse(value if "://" in value else f"//{value}")
    try:
        hostname = parsed.hostname
    except ValueError:
        return None
    if hostname is None:
        return None
    normalized = hostname.rstrip(".").lower()
    if normalized.startswith("www."):
        normalized = normalized[4:]
    if not normalized or any(character.isspace() for character in normalized):
        return None
    return normalized


def build_source_components(
    hosts_by_rgb: Mapping[str, set[str]],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    if not hosts_by_rgb or any(not key for key in hosts_by_rgb):
        raise ValueError("InfographicVQA source nodes must be non-empty")
    nodes = sorted(hosts_by_rgb)
    dsu = _DisjointSet(nodes)
    first_by_host: dict[str, str] = {}
    for rgb_hash in nodes:
        for host in sorted(hosts_by_rgb[rgb_hash]):
            if not host:
                raise ValueError("InfographicVQA source host must be non-empty")
            if host in first_by_host:
                dsu.union(rgb_hash, first_by_host[host])
            else:
                first_by_host[host] = rgb_hash
    members_by_root: dict[str, list[str]] = {}
    for rgb_hash in nodes:
        members_by_root.setdefault(dsu.find(rgb_hash), []).append(rgb_hash)
    source_by_rgb: dict[str, str] = {}
    members_by_source: dict[str, tuple[str, ...]] = {}
    for members in members_by_root.values():
        ordered = tuple(sorted(members))
        payload = (SOURCE_NAMESPACE + "\0" + "\0".join(ordered)).encode()
        source_id = "infovqa-source:" + hashlib.sha256(payload).hexdigest()
        if source_id in members_by_source:
            raise RuntimeError("InfographicVQA source hash collision")
        members_by_source[source_id] = ordered
        for rgb_hash in ordered:
            source_by_rgb[rgb_hash] = source_id
    if set(source_by_rgb) != set(hosts_by_rgb):
        raise RuntimeError("InfographicVQA component coverage differs")
    return source_by_rgb, members_by_source


def _rgb_hash(raw: bytes) -> tuple[str, int, int]:
    from PIL import Image, ImageFile  # type: ignore[import-not-found]

    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise RuntimeError("InfographicVQA audit forbids truncated-image recovery")
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width <= 0 or height <= 0:
            raise ValueError("InfographicVQA image has invalid dimensions")
        digest = hashlib.sha256()
        digest.update(width.to_bytes(8, "big"))
        digest.update(height.to_bytes(8, "big"))
        digest.update(rgb.tobytes())
        return digest.hexdigest(), width, height


def _summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("InfographicVQA summary values must be non-empty")
    ordered = sorted(int(value) for value in values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "min": ordered[0],
        "median": float(median(ordered)),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def audit_infographicvqa_train_sources(
    parquet_paths: Sequence[Any],
    download_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    import pyarrow.parquet as pq  # type: ignore[import-not-found,import-untyped]

    paths = sorted(parquet_paths, key=lambda path: path.name)
    if len(paths) != INFOVQA_TRAIN_FILES:
        raise ValueError("InfographicVQA audit requires exactly 24 train files")
    manifest_files = {
        str(row["path"]): row for row in download_manifest.get("files", [])
    }
    if (
        download_manifest.get("revision")
        != "539088ef8a8ada01ac8e2e6d4e372586748a265e"
        or int(download_manifest.get("file_count", -1)) != INFOVQA_TRAIN_FILES
        or int(download_manifest.get("aggregate_bytes", -1)) != INFOVQA_TRAIN_BYTES
        or bool(download_manifest.get("validation_files_downloaded", True))
        or bool(download_manifest.get("test_files_downloaded", True))
        or bool(download_manifest.get("questions_or_answers_read", True))
    ):
        raise ValueError("InfographicVQA download manifest contract changed")

    total_metadata_rows = 0
    for path in paths:
        relative = f"InfographicVQA/{path.name}"
        row = manifest_files.get(relative)
        if (
            row is None
            or int(row.get("bytes", -1)) != path.stat().st_size
            or str(row.get("sha256", "")) != _file_sha256(path)
        ):
            raise ValueError("InfographicVQA parquet hash or byte size differs")
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        observed_schema = {
            field.name: str(field.type) for field in schema
        }
        if list(observed_schema) != list(EXPECTED_SCHEMA) or observed_schema != EXPECTED_SCHEMA:
            raise ValueError("InfographicVQA parquet schema changed")
        total_metadata_rows += parquet.metadata.num_rows
    if total_metadata_rows != INFOVQA_TRAIN_ROWS:
        raise ValueError("InfographicVQA footer row count changed")

    seen_questions: set[str] = set()
    encoded_cache: dict[str, tuple[str, int, int]] = {}
    encoded_by_rgb: dict[str, set[str]] = {}
    hosts_by_rgb: dict[str, set[str]] = {}
    paths_by_rgb: dict[str, set[str]] = {}
    rows: list[dict[str, Any]] = []
    missing_host_rows = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        file_row = 0
        for batch in parquet.iter_batches(batch_size=64, columns=list(ALLOWED_COLUMNS)):
            data = batch.to_pydict()
            if set(data) != set(ALLOWED_COLUMNS):
                raise RuntimeError("InfographicVQA allowed-column scan changed")
            for question_id, image_value, raw_url, split in zip(
                data["questionId"],
                data["image"],
                data["image_url"],
                data["data_split"],
            ):
                question_identity = str(question_id).strip()
                if (
                    not question_identity
                    or question_identity in seen_questions
                    or str(split) != "train"
                    or not isinstance(image_value, dict)
                ):
                    raise ValueError("InfographicVQA row identity or split is invalid")
                raw = image_value.get("bytes")
                image_path = str(image_value.get("path") or "").strip()
                if not isinstance(raw, bytes) or not raw or not image_path:
                    raise ValueError("InfographicVQA image payload is invalid")
                encoded_hash = hashlib.sha256(raw).hexdigest()
                if encoded_hash not in encoded_cache:
                    encoded_cache[encoded_hash] = _rgb_hash(raw)
                rgb_hash, width, height = encoded_cache[encoded_hash]
                encoded_by_rgb.setdefault(rgb_hash, set()).add(encoded_hash)
                paths_by_rgb.setdefault(rgb_hash, set()).add(image_path)
                host = normalize_hostname(str(raw_url or ""))
                if host is None:
                    missing_host_rows += 1
                else:
                    hosts_by_rgb.setdefault(rgb_hash, set()).add(host)
                hosts_by_rgb.setdefault(rgb_hash, set())
                seen_questions.add(question_identity)
                rows.append(
                    {
                        "question_id": question_identity,
                        "transport_file": path.name,
                        "transport_row": file_row,
                        "image_path": image_path,
                        "encoded_sha256": encoded_hash,
                        "decoded_rgb_sha256": rgb_hash,
                        "width": width,
                        "height": height,
                        "normalized_hostname": host,
                    }
                )
                file_row += 1
        if file_row != parquet.metadata.num_rows:
            raise RuntimeError("InfographicVQA scanned row count differs from footer")
    if len(rows) != INFOVQA_TRAIN_ROWS or len(seen_questions) != INFOVQA_TRAIN_ROWS:
        raise RuntimeError("InfographicVQA complete row coverage differs")

    source_by_rgb, members_by_source = build_source_components(hosts_by_rgb)
    questions_by_source: dict[str, int] = {source: 0 for source in members_by_source}
    for row in rows:
        source_id = source_by_rgb[str(row["decoded_rgb_sha256"])]
        row["source_id"] = source_id
        questions_by_source[source_id] += 1
    source_rows = sorted(rows, key=lambda row: str(row["question_id"]))
    source_rank = sorted(
        members_by_source,
        key=lambda source: (
            hashlib.sha256((PILOT_NAMESPACE + "\0" + source).encode()).hexdigest(),
            source,
        ),
    )
    if len(source_rank) < INFOVQA_PILOT_SOURCES:
        raise ValueError("InfographicVQA has too few source components for pilot")
    pilot_rows = [
        {
            "selection_rank": rank,
            "source_id": source_id,
            "image_count": len(members_by_source[source_id]),
            "question_count": questions_by_source[source_id],
        }
        for rank, source_id in enumerate(source_rank[:INFOVQA_PILOT_SOURCES])
    ]

    widths = [values[1] for values in encoded_cache.values()]
    heights = [values[2] for values in encoded_cache.values()]
    image_counts = [len(members) for members in members_by_source.values()]
    question_counts = list(questions_by_source.values())
    hosts = {host for values in hosts_by_rgb.values() for host in values}
    largest_tuples = sorted(
        (
            (len(members), questions_by_source[source], source)
            for source, members in members_by_source.items()
        ),
        key=lambda row: (-row[0], -row[1], row[2]),
    )[:10]
    largest = [
        {
            "source_id": source,
            "image_count": image_count,
            "question_count": question_count,
        }
        for image_count, question_count, source in largest_tuples
    ]
    report = {
        "scientific_status": "outcome-blind InfographicVQA train transport and source-component audit",
        "n_rows": len(rows),
        "n_unique_question_ids": len(seen_questions),
        "n_encoded_images": len(encoded_cache),
        "n_decoded_rgb_images": len(hosts_by_rgb),
        "n_source_components": len(members_by_source),
        "n_normalized_hostnames": len(hosts),
        "missing_or_malformed_hostname_rows": missing_host_rows,
        "decoded_images_with_multiple_encoded_variants": sum(
            len(values) > 1 for values in encoded_by_rgb.values()
        ),
        "questions_per_source": _summary(question_counts),
        "images_per_source": _summary(image_counts),
        "encoded_image_width": _summary(widths),
        "encoded_image_height": _summary(heights),
        "largest_source_components": largest,
        "pilot_source_count": len(pilot_rows),
        "source_grouping": "connected components of normalized hostname or identical decoded RGB",
        "audits": {
            "download_manifest_contract_exact": True,
            "all_file_hashes_verified": True,
            "aggregate_bytes_exact": True,
            "schema_exact": True,
            "footer_and_scanned_rows_exact": True,
            "question_ids_unique_and_nonempty": True,
            "all_split_markers_train": True,
            "all_images_decode_without_truncated_recovery": True,
            "source_component_coverage_exact": True,
            "columns_read": list(ALLOWED_COLUMNS),
            "columns_forbidden_and_not_read": list(FORBIDDEN_COLUMNS),
            "question_text_read": False,
            "answers_read": False,
            "task_outcomes_read": False,
            "validation_or_test_rows_read": False,
        },
    }
    return report, source_rows, pilot_rows


def _file_sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
