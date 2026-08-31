#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.dataset import read_jsonl


_manifest_module = importlib.import_module(
    "scripts.verify_screenqa_formal_manifest"
    if __package__
    else "verify_screenqa_formal_manifest"
)
_one_shot_module = importlib.import_module(
    "scripts.screenqa_formal_one_shot"
    if __package__
    else "screenqa_formal_one_shot"
)
verify_manifest = _manifest_module.verify_manifest
build_contract = _one_shot_module.build_contract
verify_shard_completion = _one_shot_module.verify_shard_completion
SCIENTIFIC_STATUS = _one_shot_module.SCIENTIFIC_STATUS

EXPECTED_STATES = 14672
EXPECTED_RECORDS = 73360
EXPECTED_SOURCES = 1471
EXPECTED_SHARDS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def verify_rollouts(
    *,
    formal_manifest_dir: Path,
    candidate_dir: Path,
    expected_candidate_bundle_sha256: str,
    calibration_dir: Path,
    expected_calibration_bundle_sha256: str,
    expected_manifest_sha256: str,
    expected_manifest_audit_sha256: str,
    run_root: Path,
    rollouts: Path,
    expected_rollouts_sha256: str,
    merge_audit: Path,
    expected_merge_audit_sha256: str,
    expected_bank_code_revision: str,
    output: Path,
    resume: bool = False,
) -> dict[str, Any]:
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite formal rollout audit: {output}")
    manifest_info = verify_manifest(
        formal_manifest_dir,
        candidate_dir=candidate_dir,
        expected_candidate_bundle_sha256=expected_candidate_bundle_sha256,
        calibration_dir=calibration_dir,
        expected_calibration_bundle_sha256=expected_calibration_bundle_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_audit_sha256=expected_manifest_audit_sha256,
    )
    actual_rollouts_sha256 = sha256_file(rollouts)
    if actual_rollouts_sha256 != expected_rollouts_sha256:
        raise ValueError("ScreenQA formal rollout SHA-256 mismatch")
    if sha256_file(merge_audit) != expected_merge_audit_sha256:
        raise ValueError("ScreenQA formal merge-audit SHA-256 mismatch")
    merge = _load_json(merge_audit)
    expected_merge = {
        "passed": True,
        "manifest_sha256": expected_manifest_sha256,
        "manifest_limit": None,
        "selected_states": EXPECTED_STATES,
        "merged_records": EXPECTED_RECORDS,
        "merged_rollouts_sha256": expected_rollouts_sha256,
        "shard_count": EXPECTED_SHARDS,
        "resume_audit_required": True,
    }
    for key, expected in expected_merge.items():
        if merge.get(key) != expected:
            raise ValueError(
                f"ScreenQA formal merge audit {key} mismatch: "
                f"expected {expected!r}, got {merge.get(key)!r}"
            )
    invariants = merge.get("invariant_provenance")
    if not isinstance(invariants, Mapping):
        raise ValueError("ScreenQA formal merge invariant provenance is missing")
    expected_invariants = {
        "code_revision": expected_bank_code_revision,
        "manifest_sha256": expected_manifest_sha256,
        "manifest_limit": None,
        "manifest_examples_before_sharding": EXPECTED_STATES,
        "shard_algorithm": "sha256-state-id-v1",
        "shard_count": EXPECTED_SHARDS,
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "scorer": "screenqa",
        "candidate_count": 4,
        "proposer": "ug-grid",
        "visual_crop_ratio": 2.0,
        "visual_cost": 1.0,
        "generation_seeds": [0],
        "max_new_tokens": 32,
        "min_pixels": 200704,
        "max_pixels": 602112,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "system_prompt": "You are a helpful assistant.",
        "local_files_only": True,
    }
    for key, expected in expected_invariants.items():
        if invariants.get(key) != expected:
            raise ValueError(f"ScreenQA formal rollout provenance {key} mismatch")

    source_shards = merge.get("source_shards")
    if not isinstance(source_shards, list) or len(source_shards) != EXPECTED_SHARDS:
        raise ValueError("ScreenQA formal source-shard audit is incomplete")
    marker_hashes: list[str] = []
    shard_states = 0
    shard_records = 0
    manifest = formal_manifest_dir / "manifest.jsonl"
    for expected_index, shard in enumerate(source_shards):
        if not isinstance(shard, Mapping):
            raise ValueError("ScreenQA formal source-shard audit is malformed")
        expected_dir = run_root / f"shard-{expected_index:05d}-of-00004"
        if Path(str(shard.get("directory", ""))).resolve() != expected_dir.resolve():
            raise ValueError(f"ScreenQA formal shard {expected_index} path mismatch")
        if (
            shard.get("shard_index") != expected_index
            or shard.get("resume_audit_required") is not True
            or not shard.get("resume_audit_sha256")
        ):
            raise ValueError(f"ScreenQA formal shard {expected_index} proof mismatch")
        contract = build_contract(
            shard_dir=expected_dir,
            manifest=manifest,
            expected_manifest_sha256=expected_manifest_sha256,
            manifest_audit_sha256=expected_manifest_audit_sha256,
            candidate_bundle_sha256=expected_candidate_bundle_sha256,
            calibration_bundle_sha256=expected_calibration_bundle_sha256,
            code_revision=expected_bank_code_revision,
            shard_index=expected_index,
            shard_count=EXPECTED_SHARDS,
            expected_total_states=EXPECTED_STATES,
        )
        completion = verify_shard_completion(expected_dir, contract)
        marker_hashes.append(sha256_file(expected_dir / "formal-complete.json"))
        if completion["rollouts_sha256"] != shard.get("rollouts_sha256"):
            raise ValueError(f"ScreenQA formal shard {expected_index} rollout mismatch")
        if completion["provenance_sha256"] != shard.get("provenance_sha256"):
            raise ValueError(
                f"ScreenQA formal shard {expected_index} provenance mismatch"
            )
        if completion["resume_audit_sha256"] != shard.get("resume_audit_sha256"):
            raise ValueError(
                f"ScreenQA formal shard {expected_index} resume audit mismatch"
            )
        shard_states += int(completion["states"])
        shard_records += int(completion["records"])
    if shard_states != EXPECTED_STATES or shard_records != EXPECTED_RECORDS:
        raise ValueError("ScreenQA formal one-shot shard totals mismatch")

    records = read_jsonl(rollouts)
    state_counts = Counter(record.state_id for record in records)
    source_ids = {record.source_id for record in records}
    if len(records) != EXPECTED_RECORDS or len(state_counts) != EXPECTED_STATES:
        raise ValueError("ScreenQA formal merged rollout dimensions mismatch")
    if set(state_counts.values()) != {5}:
        raise ValueError("ScreenQA formal states lack five sibling records")
    if len(source_ids) != EXPECTED_SOURCES or any(
        not source_id.startswith("screenqa:") for source_id in source_ids
    ):
        raise ValueError("ScreenQA formal source-component coverage mismatch")
    answer_records = sum(record.action_type == "ANSWER" for record in records)
    zoom_records = sum(record.action_type == "ZOOM" for record in records)
    if answer_records != EXPECTED_STATES or zoom_records != EXPECTED_STATES * 4:
        raise ValueError("ScreenQA formal ANSWER/ZOOM sibling counts mismatch")
    audit = {
        "passed": True,
        "scientific_status": SCIENTIFIC_STATUS,
        "one_shot_formal_bank_complete": True,
        "rollouts": str(rollouts.resolve()),
        "rollouts_sha256": actual_rollouts_sha256,
        "merge_audit": str(merge_audit.resolve()),
        "merge_audit_sha256": expected_merge_audit_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "manifest_audit_sha256": expected_manifest_audit_sha256,
        "candidate_bundle_sha256": expected_candidate_bundle_sha256,
        "calibration_bundle_sha256": expected_calibration_bundle_sha256,
        "selected_threshold": manifest_info["selected_threshold"],
        "bank_code_revision": expected_bank_code_revision,
        "states": len(state_counts),
        "records": len(records),
        "answer_records": answer_records,
        "zoom_records": zoom_records,
        "source_components": len(source_ids),
        "one_shot_completed_shards": len(marker_hashes),
        "completion_marker_sha256s": marker_hashes,
        "ranker_training_outcomes_used": True,
        "calibration_outcomes_used": True,
        "formal_outcomes_collected": True,
        "formal_outcomes_used_for_tuning": False,
        "reserve_outcomes_opened": False,
        "untouched_outcomes_opened": False,
        "official_validation_test_opened": False,
    }
    serialized = json.dumps(audit, allow_nan=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text(encoding="utf-8") != serialized:
            raise ValueError("existing ScreenQA formal rollout audit changed")
    else:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the complete one-shot ScreenQA formal sibling bank"
    )
    parser.add_argument("--formal-manifest-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--expected-candidate-bundle-sha256", required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--expected-calibration-bundle-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-manifest-audit-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--merge-audit", type=Path, required=True)
    parser.add_argument("--expected-merge-audit-sha256", required=True)
    parser.add_argument("--expected-bank-code-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = verify_rollouts(
        formal_manifest_dir=args.formal_manifest_dir.resolve(),
        candidate_dir=args.candidate_dir.resolve(),
        expected_candidate_bundle_sha256=args.expected_candidate_bundle_sha256,
        calibration_dir=args.calibration_dir.resolve(),
        expected_calibration_bundle_sha256=(
            args.expected_calibration_bundle_sha256
        ),
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_manifest_audit_sha256=args.expected_manifest_audit_sha256,
        run_root=args.run_root.resolve(),
        rollouts=args.rollouts.resolve(),
        expected_rollouts_sha256=args.expected_rollouts_sha256,
        merge_audit=args.merge_audit.resolve(),
        expected_merge_audit_sha256=args.expected_merge_audit_sha256,
        expected_bank_code_revision=args.expected_bank_code_revision,
        output=args.output.resolve(),
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
