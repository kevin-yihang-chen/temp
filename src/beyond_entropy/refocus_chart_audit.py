from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AUDIT_SCHEMA = "refocus_chart_metadata_audit_v1"
LINEAGE_SCHEMA = "refocus_chart_train_lineage_audit_v1"
ROW_SCHEMA = "refocus_chart_row_manifest_v1"
STRUCTURAL_METADATA_FIELDS = (
    "source",
    "x_values",
    "y_values",
    "x_values_bbox",
    "y_values_bbox",
    "figure_bbox",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_pinned_shard(
    shard_root: Path, shard: Mapping[str, Any]
) -> tuple[Path, str]:
    root = shard_root.resolve(strict=True)
    path = (root / str(shard["path"])).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"official train shard escapes local root: {path}")
    expected_size = int(shard["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"official train shard size mismatch for {path}: "
            f"{actual_size} != {expected_size}"
        )
    actual_sha256 = sha256_file(path)
    expected_sha256 = str(shard["lfs_sha256"])
    _require_hex_digest("official train shard lfs_sha256", expected_sha256, length=64)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"official train shard SHA-256 mismatch for {path}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return path, actual_sha256


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_hex_digest(name: str, value: str, *, length: int) -> None:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a {length}-character lowercase hex digest")


def parse_tools_metadata(value: object) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("tools metadata must be valid JSON") from exc
    else:
        parsed = value
    metadata = _require_mapping("tools metadata", parsed)
    missing = [field for field in STRUCTURAL_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"tools metadata is missing structural fields: {missing}")
    return metadata


def structural_chart_signature(value: object) -> str:
    """Hash a conservative non-pixel proxy for chart identity.

    The question-dependent focus area is intentionally excluded. Equal signatures
    are treated as a potential shared chart, but unequal signatures do not prove
    that underlying image pixels differ.
    """

    metadata = parse_tools_metadata(value)
    structural = {field: metadata[field] for field in STRUCTURAL_METADATA_FIELDS}
    return canonical_sha256(structural)


@dataclass(frozen=True)
class RefocusRowManifest:
    row_id: str
    row_index: int
    split: str
    source: str
    structural_chart_sha256: str
    question_sha256: str
    answer_sha256: str
    question_answer_sha256: str
    prompt_sha256: str
    tools_name: str

    def __post_init__(self) -> None:
        if not self.row_id:
            raise ValueError("row_id must be non-empty")
        if self.row_index < 0:
            raise ValueError("row_index must be non-negative")
        if not self.split or not self.source or not self.tools_name:
            raise ValueError("split, source, and tools_name must be non-empty")
        for field in (
            "structural_chart_sha256",
            "question_sha256",
            "answer_sha256",
            "question_answer_sha256",
            "prompt_sha256",
        ):
            value = getattr(self, field)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": ROW_SCHEMA, **asdict(self)}


def normalize_official_bbox_columns(
    labels: object,
    bbox_columns: object,
    *,
    field_name: str,
) -> dict[str, dict[str, Any]]:
    """Convert the official Arrow columnar bbox representation to VTool JSON.

    Hugging Face stores each bbox mapping as four parallel coordinate arrays,
    whereas VTool serializes the same values under their axis-label keys.
    """

    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        raise ValueError(f"{field_name} labels must be a sequence")
    normalized_labels = [str(label) for label in labels]
    if len(normalized_labels) != len(set(normalized_labels)):
        raise ValueError(f"{field_name} labels must be unique")
    columns = _require_mapping(field_name, bbox_columns)
    coordinate_names = ("x1", "y1", "x2", "y2")
    if set(columns) != set(coordinate_names):
        raise ValueError(f"{field_name} must contain exactly {list(coordinate_names)}")
    coordinate_columns: dict[str, Sequence[Any]] = {}
    for coordinate_name in coordinate_names:
        values = columns[coordinate_name]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"{field_name}.{coordinate_name} must be a sequence")
        if len(values) != len(normalized_labels):
            raise ValueError(
                f"{field_name}.{coordinate_name} length {len(values)} does not "
                f"match labels length {len(normalized_labels)}"
            )
        coordinate_columns[coordinate_name] = values
    return {
        label: {
            coordinate_name: coordinate_columns[coordinate_name][index]
            for coordinate_name in coordinate_names
        }
        for index, label in enumerate(normalized_labels)
    }


def build_row_manifest(
    row: Mapping[str, Any], *, expected_split: str
) -> RefocusRowManifest:
    raw_row_id = row.get("id")
    if not isinstance(raw_row_id, str) or not raw_row_id.strip():
        raise ValueError("row id must be a non-empty string")
    row_id = raw_row_id.strip()
    split = str(row.get("split", "")).strip()
    source = str(row.get("source", "")).strip()
    if split != expected_split:
        raise ValueError(
            f"row {row_id!r} split {split!r} != expected {expected_split!r}"
        )
    if str(row.get("agent_name", "")) != "vtool_agent":
        raise ValueError(f"row {row_id!r} has unexpected agent_name")
    if str(row.get("data_source", "")) != "ReFocus/ReFocus_Data":
        raise ValueError(f"row {row_id!r} has unexpected data_source")

    extra = _require_mapping("extra_info", row.get("extra_info"))
    if str(extra.get("split", "")) != expected_split:
        raise ValueError(f"row {row_id!r} extra_info split mismatch")
    if extra.get("need_tools_kwargs") is not True:
        raise ValueError(f"row {row_id!r} does not require tools_kwargs")
    row_index = extra.get("index")
    if type(row_index) is not int or row_index < 0:
        raise ValueError(f"row {row_id!r} index must be a non-negative integer")

    tools = _require_mapping("tools_kwargs", extra.get("tools_kwargs"))
    tools_name = str(tools.get("name", "")).strip()
    metadata = parse_tools_metadata(tools.get("metadata"))
    if str(metadata["source"]) != source:
        raise ValueError(f"row {row_id!r} metadata source mismatch")

    question = str(extra.get("question", "")).strip()
    answer = str(extra.get("answer", "")).strip()
    if not question or not answer:
        raise ValueError(f"row {row_id!r} question and answer must be non-empty")
    reward_model = _require_mapping("reward_model", row.get("reward_model"))
    if str(reward_model.get("ground_truth", "")).strip() != answer:
        raise ValueError(
            f"row {row_id!r} reward target disagrees with extra_info answer"
        )
    if str(reward_model.get("style", "")) != "rule":
        raise ValueError(f"row {row_id!r} has unexpected reward style")

    prompt = row.get("prompt")
    if (
        not isinstance(prompt, Sequence)
        or isinstance(prompt, (str, bytes))
        or not prompt
    ):
        raise ValueError(f"row {row_id!r} prompt must be a non-empty sequence")

    return RefocusRowManifest(
        row_id=row_id,
        row_index=row_index,
        split=split,
        source=source,
        structural_chart_sha256=structural_chart_signature(metadata),
        question_sha256=canonical_sha256(question),
        answer_sha256=canonical_sha256(answer),
        question_answer_sha256=canonical_sha256([question, answer]),
        prompt_sha256=canonical_sha256(prompt),
        tools_name=tools_name,
    )


def audit_split_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
    dataset_revision: str,
    parquet_sha256: str,
) -> dict[str, Any]:
    _require_hex_digest("dataset_revision", dataset_revision, length=40)
    _require_hex_digest("parquet_sha256", parquet_sha256, length=64)
    manifests = [build_row_manifest(row, expected_split=split) for row in rows]
    manifests.sort(key=lambda item: item.row_index)
    if not manifests:
        raise ValueError(f"{split} rows must be non-empty")
    indices = [item.row_index for item in manifests]
    if indices != list(range(len(manifests))):
        raise ValueError(f"{split} row indices must be contiguous from zero")

    def counts(field: str) -> Counter[Any]:
        return Counter(getattr(item, field) for item in manifests)

    row_ids = counts("row_id")
    chart_signatures = counts("structural_chart_sha256")
    questions = counts("question_sha256")
    question_answers = counts("question_answer_sha256")
    prompts = counts("prompt_sha256")
    manifest_rows = [item.to_dict() for item in manifests]
    return {
        "schema": AUDIT_SCHEMA,
        "split": split,
        "dataset_revision": dataset_revision,
        "parquet_sha256": parquet_sha256,
        "parquet_sha256_verification": (
            "declared LFS digest supplied to metadata-only range audit; "
            "full Parquet bytes were not hashed"
        ),
        "rows": len(manifests),
        "unique_row_ids": len(row_ids),
        "duplicate_row_id_rows": len(manifests) - len(row_ids),
        "unique_structural_chart_signatures": len(chart_signatures),
        "structural_chart_duplicate_rows": len(manifests) - len(chart_signatures),
        "max_rows_per_structural_chart": max(chart_signatures.values()),
        "duplicate_exact_question_rows": len(manifests) - len(questions),
        "duplicate_question_answer_rows": len(manifests) - len(question_answers),
        "duplicate_prompt_rows": len(manifests) - len(prompts),
        "source_counts": dict(sorted(counts("source").items())),
        "tools_name_counts": dict(sorted(counts("tools_name").items())),
        "manifest_sha256": canonical_sha256(manifest_rows),
        "row_manifests": manifest_rows,
    }


def audit_chartqa_train_lineage(
    row_ids: Iterable[str],
    *,
    train_png_entries: Iterable[Mapping[str, Any]],
    repository: str,
    root_tree_sha: str,
    train_png_tree_sha: str,
) -> dict[str, Any]:
    """Match Refocus row IDs against a pinned ChartQA train/png tree only.

    The caller must supply entries from ``ChartQA Dataset/train/png``. This
    function deliberately has no validation/test split input so the lineage
    check cannot turn into another protected-split overlap audit.
    """

    _require_hex_digest("root_tree_sha", root_tree_sha, length=40)
    _require_hex_digest("train_png_tree_sha", train_png_tree_sha, length=40)
    if not repository.strip():
        raise ValueError("repository must be non-empty")
    supplied_row_ids = list(row_ids)
    if any(not isinstance(row_id, str) for row_id in supplied_row_ids):
        raise ValueError("row_ids must be strings")
    normalized_row_ids = [row_id.strip() for row_id in supplied_row_ids]
    if not normalized_row_ids or any(not row_id for row_id in normalized_row_ids):
        raise ValueError("row_ids must be non-empty strings")
    duplicate_row_ids = len(normalized_row_ids) - len(set(normalized_row_ids))

    png_stems: list[str] = []
    for entry in train_png_entries:
        if entry.get("type") != "blob":
            continue
        path = str(entry.get("path", ""))
        if not path.lower().endswith(".png"):
            continue
        filename = path.rsplit("/", 1)[-1]
        stem = filename[:-4]
        if not stem:
            raise ValueError("ChartQA train PNG entry has an empty stem")
        png_stems.append(stem)
    if not png_stems:
        raise ValueError("ChartQA train PNG tree must contain PNG blobs")
    duplicate_png_stems = len(png_stems) - len(set(png_stems))
    if duplicate_png_stems:
        raise ValueError("ChartQA train PNG stems must be unique")

    row_id_set = set(normalized_row_ids)
    png_stem_set = set(png_stems)
    matched = row_id_set & png_stem_set
    missing = sorted(row_id_set - png_stem_set)
    decision = (
        "all_refocus_train_row_ids_match_pinned_chartqa_train_png_stems"
        if not missing and duplicate_row_ids == 0
        else "refocus_train_lineage_not_fully_resolved"
    )
    return {
        "schema": LINEAGE_SCHEMA,
        "repository": repository,
        "root_tree_sha": root_tree_sha,
        "train_png_tree_sha": train_png_tree_sha,
        "refocus_train_rows": len(normalized_row_ids),
        "refocus_train_unique_row_ids": len(row_id_set),
        "refocus_train_duplicate_row_id_rows": duplicate_row_ids,
        "chartqa_train_png_blobs": len(png_stems),
        "chartqa_train_unique_png_stems": len(png_stem_set),
        "matched_unique_row_ids": len(matched),
        "missing_unique_row_ids": len(missing),
        "missing_row_id_sha256": [canonical_sha256(row_id) for row_id in missing],
        "refocus_row_ids_sha256": canonical_sha256(sorted(normalized_row_ids)),
        "chartqa_train_png_stems_sha256": canonical_sha256(sorted(png_stems)),
        "pixel_identity_checked": False,
        "protected_split_contents_accessed": False,
        "decision": decision,
        "interpretation": (
            "row-ID membership supports original ChartQA train lineage; it does "
            "not prove pixel equality or establish a license for the derivative"
        ),
    }
