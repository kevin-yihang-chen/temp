from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from beyond_entropy.chartqapro import (
    CHARTQAPRO_PILOT_NAMESPACE,
    CHARTQAPRO_PROMPT_ADAPTER,
    build_chartqapro_direct_prompt,
    chartqapro_final_question,
    chartqapro_target,
    select_chartqapro_pilot_images,
)
from beyond_entropy.manifest_export import image_digest
from beyond_entropy.vtool_adapter import (
    normalize_question_for_vtool_join,
    vtool_identity_join_key,
)


DATASET_ID = "ahmed-masry/ChartQAPro"
DATASET_REVISION = "e27c2874825874d6767d2bbc538ed4f0dc2c64c2"
EXPECTED_PARQUET_SHA256 = (
    "feb03a5579e49114b45350b75aed7f674e285f795a425c83e22f400095c91489"
)
EXPECTED_ROWS = 1948
OFFICIAL_CODE_REVISION = "4b422c658270aff1d3105fd0fb39b1dd5de9f08c"
OFFICIAL_EVALUATOR_SHA256 = (
    "fe21d33f076a394765935b6231c2b918c321d4d197d618374ab2ab6b2cc96a71"
)
VLMEVALKIT_REVISION = "3cd4332def8c1bf224b0c171cd73f292acc4482c"
VTOOL_DATASET_REVISION = "00f10ecc5b25d94fd66e14c3671af9fb0f088989"
EXPECTED_VTOOL_PARQUET_SHA256 = (
    "f2055cd5dd667cfb3c313f22905adb1f536e41e0433839e832f638478ba0c1c3"
)
EXPECTED_VTOOL_AUDIT_SHA256 = (
    "015a2bc18a9175bc121370fdfa082de314c2dcd339aca84624ce6cb97b5e803a"
)
EXPECTED_BLOCKED_MANIFEST_HASHES = {
    "chartqa_development": (
        "3c485aa5c09cc9491f866ba5737a78c2b79c3539c6de2663c964b2cff90d814a"
    ),
    "chartqa_validation": (
        "d3178218853b10447228963e839716f0eac768b51bdc0f5b4a83268d3819b58b"
    ),
    "chartqa_train_replication": (
        "72db6feaa4bc042e98741a48dd55421c5246c1b48c84b1fd75740d1d072ca621"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _question_digest(question: str) -> str:
    return hashlib.sha256(
        normalize_question_for_vtool_join(question).encode()
    ).hexdigest()


def _duplicate_rows(values: Iterable[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def _read_manifest_identity(
    name: str,
    path: Path,
) -> dict[str, Any]:
    expected_hash = EXPECTED_BLOCKED_MANIFEST_HASHES[name]
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"blocked manifest {name} SHA-256 mismatch: {actual_hash}"
        )
    image_ids: list[str] = []
    question_digests: list[str] = []
    joint_keys: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                image_id = str(value["image_id"])
                question = str(value["question"])
                image_ids.append(image_id)
                question_digests.append(_question_digest(question))
                joint_keys.append(vtool_identity_join_key(image_id, question))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid blocked manifest row {path}:{line_number}: {exc}"
                ) from exc
    return {
        "name": name,
        "path": str(path.resolve()),
        "sha256": actual_hash,
        "rows": len(image_ids),
        "image_ids": set(image_ids),
        "question_digests": set(question_digests),
        "joint_keys": set(joint_keys),
        "unique_images": len(set(image_ids)),
        "unique_questions": len(set(question_digests)),
        "unique_joint_keys": len(set(joint_keys)),
        "duplicate_joint_rows": _duplicate_rows(joint_keys),
    }


def _read_vtool_identity(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    from PIL import Image  # type: ignore[import-untyped]

    actual_hash = _sha256(path)
    if actual_hash != EXPECTED_VTOOL_PARQUET_SHA256:
        raise ValueError(f"VTool parquet SHA-256 mismatch: {actual_hash}")
    table = parquet.read_table(path, columns=["id", "images", "extra_info"])
    image_ids: list[str] = []
    question_digests: list[str] = []
    joint_keys: list[str] = []
    for row in table.to_pylist():
        images = row.get("images") or []
        extra_info = row.get("extra_info")
        if (
            len(images) != 1
            or not isinstance(images[0], Mapping)
            or not images[0].get("bytes")
            or not isinstance(extra_info, Mapping)
            or not extra_info.get("question")
        ):
            raise ValueError(f"invalid VTool identity row: {row.get('id')!r}")
        with Image.open(io.BytesIO(images[0]["bytes"])) as loaded:
            image_id = image_digest(loaded.convert("RGB"))
        question = str(extra_info["question"])
        image_ids.append(image_id)
        question_digests.append(_question_digest(question))
        joint_keys.append(vtool_identity_join_key(image_id, question))
    return {
        "name": "vtool_test",
        "path": str(path.resolve()),
        "sha256": actual_hash,
        "dataset_revision": VTOOL_DATASET_REVISION,
        "rows": len(image_ids),
        "image_ids": set(image_ids),
        "question_digests": set(question_digests),
        "joint_keys": set(joint_keys),
        "unique_images": len(set(image_ids)),
        "unique_questions": len(set(question_digests)),
        "unique_joint_keys": len(set(joint_keys)),
        "duplicate_joint_rows": _duplicate_rows(joint_keys),
    }


def _public_source_summary(source: Mapping[str, Any]) -> dict[str, Any]:
    omitted = {"image_ids", "question_digests", "joint_keys", "name"}
    return {key: value for key, value in source.items() if key not in omitted}


def _manifest_bytes(payloads: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
        for payload in payloads
    ]
    return ("\n".join(lines) + "\n").encode()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("pyarrow", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze identity-audited ChartQAPro Gate 3 pilot/formal manifests"
    )
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-image-count", type=int, default=200)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--train-replication-manifest", type=Path, required=True)
    parser.add_argument("--vtool-parquet", type=Path, required=True)
    parser.add_argument("--vtool-audit", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    from PIL import Image  # type: ignore[import-untyped]

    args = parse_args()
    source_parquet = args.parquet.resolve()
    source_hash = _sha256(source_parquet)
    if source_hash != EXPECTED_PARQUET_SHA256:
        raise ValueError(f"ChartQAPro parquet SHA-256 mismatch: {source_hash}")
    vtool_audit_hash = _sha256(args.vtool_audit)
    if vtool_audit_hash != EXPECTED_VTOOL_AUDIT_SHA256:
        raise ValueError(f"VTool audit SHA-256 mismatch: {vtool_audit_hash}")

    blocked_sources = [
        _read_manifest_identity(
            "chartqa_development", args.development_manifest.resolve()
        ),
        _read_manifest_identity(
            "chartqa_validation", args.validation_manifest.resolve()
        ),
        _read_manifest_identity(
            "chartqa_train_replication", args.train_replication_manifest.resolve()
        ),
        _read_vtool_identity(args.vtool_parquet.resolve()),
    ]
    blocked_image_ids = set().union(
        *(source["image_ids"] for source in blocked_sources)
    )

    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    table = parquet.ParquetFile(source_parquet)
    if table.metadata.num_rows != EXPECTED_ROWS:
        raise ValueError(
            f"expected {EXPECTED_ROWS} ChartQAPro rows, found {table.metadata.num_rows}"
        )

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        image_dir = temporary / "images"
        image_dir.mkdir()
        payloads: list[dict[str, Any]] = []
        target_image_ids: list[str] = []
        target_question_digests: list[str] = []
        target_joint_keys: list[str] = []
        excluded_source_indices: list[int] = []
        excluded_by_source: dict[str, list[int]] = {
            str(source["name"]): [] for source in blocked_sources
        }
        question_type_counts: Counter[str] = Counter()
        paragraph_presence: Counter[str] = Counter()
        year_flag_length_anomalies: list[dict[str, Any]] = []
        saved_images: set[str] = set()
        source_index = 0

        columns = [
            "Question",
            "Answer",
            "Question Type",
            "image",
            "Year",
            "Paragraph",
        ]
        for batch in table.iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                questions = [str(value) for value in (row.get("Question") or [])]
                answers = [str(value) for value in (row.get("Answer") or [])]
                year_flags = [str(value) for value in (row.get("Year") or [])]
                question_type = str(row.get("Question Type", ""))
                paragraph_value = row.get("Paragraph")
                paragraph = None if paragraph_value is None else str(paragraph_value)
                image_bytes = row.get("image")
                if not isinstance(image_bytes, bytes) or not image_bytes:
                    raise ValueError(f"invalid image bytes at source index {source_index}")
                with Image.open(io.BytesIO(image_bytes)) as loaded:
                    image = loaded.convert("RGB")
                image_id = image_digest(image)
                final_question = chartqapro_final_question(questions, question_type)
                prompt = build_chartqapro_direct_prompt(
                    questions,
                    answers,
                    question_type,
                    paragraph,
                )
                target = chartqapro_target(answers, year_flags, question_type)
                if len(answers) != len(year_flags):
                    year_flag_length_anomalies.append(
                        {
                            "source_index": source_index,
                            "question_type": question_type,
                            "answer_turns": len(answers),
                            "year_flags": len(year_flags),
                        }
                    )
                question_digest = _question_digest(final_question)
                joint_key = vtool_identity_join_key(image_id, final_question)
                target_image_ids.append(image_id)
                target_question_digests.append(question_digest)
                target_joint_keys.append(joint_key)
                question_type_counts[question_type] += 1
                paragraph_presence[
                    "present" if paragraph and paragraph.strip() else "absent"
                ] += 1

                matching_sources = [
                    str(source["name"])
                    for source in blocked_sources
                    if image_id in source["image_ids"]
                ]
                if matching_sources:
                    excluded_source_indices.append(source_index)
                    for name in matching_sources:
                        excluded_by_source[name].append(source_index)
                    image.close()
                    source_index += 1
                    continue

                image_name = f"{image_id}.png"
                if image_id not in saved_images:
                    image.save(image_dir / image_name, format="PNG")
                    saved_images.add(image_id)
                image.close()
                payloads.append(
                    {
                        "state_id": f"chartqapro:{source_index:04d}",
                        "image_id": image_id,
                        "source_id": image_id,
                        "image_path": f"../images/{image_name}",
                        "question": prompt,
                        "target": target,
                        "benchmark": "chartqapro",
                        "stratum": question_type,
                        "source_index": source_index,
                        "dataset_revision": DATASET_REVISION,
                        "prompt_adapter": CHARTQAPRO_PROMPT_ADAPTER,
                        "identity_question_sha256": question_digest,
                        "identity_join_key": joint_key,
                        "paragraph_present": bool(paragraph and paragraph.strip()),
                    }
                )
                source_index += 1

        if source_index != EXPECTED_ROWS:
            raise ValueError(f"read {source_index} rows instead of {EXPECTED_ROWS}")
        duplicate_joint_rows = _duplicate_rows(target_joint_keys)
        if duplicate_joint_rows:
            raise ValueError(
                f"target contains {duplicate_joint_rows} ambiguous duplicate identity rows"
            )
        if not payloads:
            raise ValueError("identity audit excluded every target row")
        eligible_image_ids = [str(payload["image_id"]) for payload in payloads]
        pilot_image_ids = set(
            select_chartqapro_pilot_images(
                eligible_image_ids,
                count=args.pilot_image_count,
            )
        )
        pilot_payloads = [
            payload for payload in payloads if payload["image_id"] in pilot_image_ids
        ]
        formal_payloads = [
            payload for payload in payloads if payload["image_id"] not in pilot_image_ids
        ]
        pilot_images = {str(payload["image_id"]) for payload in pilot_payloads}
        formal_images = {str(payload["image_id"]) for payload in formal_payloads}
        if pilot_images & formal_images:
            raise AssertionError("pilot and formal image groups overlap")
        if pilot_images | formal_images != set(eligible_image_ids):
            raise AssertionError("pilot/formal split lost image groups")

        split_results: dict[str, Any] = {}
        for split_name, split_payloads in (
            ("pilot", pilot_payloads),
            ("formal", formal_payloads),
        ):
            split_dir = temporary / split_name
            split_dir.mkdir()
            manifest_bytes = _manifest_bytes(split_payloads)
            manifest_path = split_dir / "manifest.jsonl"
            manifest_path.write_bytes(manifest_bytes)
            strata = Counter(str(payload["stratum"]) for payload in split_payloads)
            unique_images = len(
                {str(payload["image_id"]) for payload in split_payloads}
            )
            provenance = {
                "scientific_status": (
                    "compatibility-only pilot"
                    if split_name == "pilot"
                    else "untouched formal target; do not inspect before freeze"
                ),
                "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "source_parquet": str(source_parquet),
                "source_parquet_sha256": source_hash,
                "split": split_name,
                "selection": (
                    f"first {args.pilot_image_count} image groups by salted SHA-256 rank"
                    if split_name == "pilot"
                    else "all eligible image groups outside deterministic pilot"
                ),
                "pilot_namespace": CHARTQAPRO_PILOT_NAMESPACE,
                "rows": len(split_payloads),
                "unique_images": unique_images,
                "source_indices": [
                    int(payload["source_index"]) for payload in split_payloads
                ],
                "stratum_counts": dict(sorted(strata.items())),
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "prompt_adapter": CHARTQAPRO_PROMPT_ADAPTER,
                "primary_scorer": "chartqapro released-code parity",
                "sensitivity_scorer": (
                    "chartqapro-spec paper-specified exact category matching"
                ),
                "official_code_revision": OFFICIAL_CODE_REVISION,
                "official_evaluator_sha256": OFFICIAL_EVALUATOR_SHA256,
                "vlmevalkit_revision": VLMEVALKIT_REVISION,
                "code_revision": os.environ.get("BE_CODE_REVISION"),
                "packages": _package_versions(),
            }
            _write_json(split_dir / "manifest.provenance.json", provenance)
            split_results[split_name] = {
                "manifest": str((destination / split_name / "manifest.jsonl")),
                "manifest_sha256": provenance["manifest_sha256"],
                "rows": len(split_payloads),
                "unique_images": unique_images,
                "stratum_counts": provenance["stratum_counts"],
            }

        target_image_set = set(target_image_ids)
        target_question_set = set(target_question_digests)
        target_joint_set = set(target_joint_keys)
        blocked_audit: dict[str, Any] = {}
        for source in blocked_sources:
            name = str(source["name"])
            image_overlap = target_image_set & source["image_ids"]
            question_overlap = target_question_set & source["question_digests"]
            joint_overlap = target_joint_set & source["joint_keys"]
            blocked_audit[name] = {
                **_public_source_summary(source),
                "target_unique_image_overlap": len(image_overlap),
                "target_unique_question_overlap": len(question_overlap),
                "target_unique_joint_overlap": len(joint_overlap),
                "excluded_target_rows_by_image": len(excluded_by_source[name]),
            }
        identity_audit = {
            "scientific_status": "pre-outcome exact identity audit",
            "join_key": "decoded RGB SHA-256 plus normalized final-question SHA-256",
            "exclusion_rule": "exclude an entire target image group on any prior RGB overlap",
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "source_parquet": str(source_parquet),
            "source_parquet_sha256": source_hash,
            "rows": len(target_image_ids),
            "unique_images": len(target_image_set),
            "duplicate_image_rows": _duplicate_rows(target_image_ids),
            "unique_questions": len(target_question_set),
            "unique_joint_keys": len(target_joint_set),
            "duplicate_joint_rows": duplicate_joint_rows,
            "question_type_counts": dict(sorted(question_type_counts.items())),
            "paragraph_presence": dict(sorted(paragraph_presence.items())),
            "year_flag_length_anomalies": year_flag_length_anomalies,
            "year_flag_anomaly_policy": (
                "preserve source flags; released scorer uses only the final flag "
                "for Conversational rows"
            ),
            "excluded_source_indices": excluded_source_indices,
            "excluded_rows": len(excluded_source_indices),
            "eligible_rows": len(payloads),
            "eligible_unique_images": len(set(eligible_image_ids)),
            "blocked_sources": blocked_audit,
            "vtool_prior_audit": {
                "path": str(args.vtool_audit.resolve()),
                "sha256": vtool_audit_hash,
            },
            "pilot": split_results["pilot"],
            "formal": split_results["formal"],
        }
        audit_path = temporary / "identity-audit.json"
        _write_json(audit_path, identity_audit)
        freeze_provenance = {
            "scientific_status": "frozen Gate 3 target export",
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "source_parquet_sha256": source_hash,
            "identity_audit_sha256": _sha256(audit_path),
            "pilot_manifest_sha256": split_results["pilot"]["manifest_sha256"],
            "formal_manifest_sha256": split_results["formal"]["manifest_sha256"],
            "prompt_adapter": CHARTQAPRO_PROMPT_ADAPTER,
            "primary_scorer": "chartqapro released-code parity",
            "sensitivity_scorer": (
                "chartqapro-spec paper-specified exact category matching"
            ),
            "official_code_revision": OFFICIAL_CODE_REVISION,
            "official_evaluator_sha256": OFFICIAL_EVALUATOR_SHA256,
            "vlmevalkit_revision": VLMEVALKIT_REVISION,
            "code_revision": os.environ.get("BE_CODE_REVISION"),
        }
        _write_json(temporary / "freeze.provenance.json", freeze_provenance)
        temporary.replace(destination)

    print(
        json.dumps(
            {
                "output_dir": str(destination),
                "identity_audit_sha256": freeze_provenance["identity_audit_sha256"],
                "pilot": split_results["pilot"],
                "formal": split_results["formal"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
