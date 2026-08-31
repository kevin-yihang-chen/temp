#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from beyond_entropy.dataset import read_jsonl


EXPECTED_SCIENTIFIC_STATUS = (
    "ScreenQA frozen risk-calibration sibling bank; outcomes may calibrate only "
    "the frozen threshold sequence"
)
EXPECTED_STATES = 9951
EXPECTED_RECORDS = 49755
EXPECTED_SOURCES = 1016


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
    rollouts: Path,
    expected_rollouts_sha256: str,
    merge_audit: Path,
    expected_merge_audit_sha256: str,
    expected_manifest_sha256: str,
    expected_bank_code_revision: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite rollout audit: {output}")
    actual_rollouts_sha256 = sha256_file(rollouts)
    if actual_rollouts_sha256 != expected_rollouts_sha256:
        raise ValueError("ScreenQA calibration rollout SHA-256 mismatch")
    if sha256_file(merge_audit) != expected_merge_audit_sha256:
        raise ValueError("ScreenQA calibration merge-audit SHA-256 mismatch")
    merge = _load_json(merge_audit)
    expected_merge = {
        "passed": True,
        "manifest_sha256": expected_manifest_sha256,
        "manifest_limit": None,
        "selected_states": EXPECTED_STATES,
        "merged_records": EXPECTED_RECORDS,
        "merged_rollouts_sha256": expected_rollouts_sha256,
        "shard_count": 4,
        "resume_audit_required": True,
    }
    for key, expected in expected_merge.items():
        if merge.get(key) != expected:
            raise ValueError(
                f"ScreenQA calibration merge audit {key} mismatch: "
                f"expected {expected!r}, got {merge.get(key)!r}"
            )
    invariants = merge.get("invariant_provenance")
    if not isinstance(invariants, Mapping):
        raise ValueError("ScreenQA calibration merge invariant provenance is missing")
    expected_invariants = {
        "code_revision": expected_bank_code_revision,
        "manifest_sha256": expected_manifest_sha256,
        "manifest_limit": None,
        "manifest_examples_before_sharding": EXPECTED_STATES,
        "shard_algorithm": "sha256-state-id-v1",
        "shard_count": 4,
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
            raise ValueError(f"ScreenQA calibration rollout provenance {key} mismatch")

    source_shards = merge.get("source_shards")
    if not isinstance(source_shards, list) or len(source_shards) != 4:
        raise ValueError("ScreenQA calibration source-shard audit is incomplete")
    shard_states = 0
    shard_records = 0
    for expected_index, shard in enumerate(source_shards):
        if not isinstance(shard, Mapping):
            raise ValueError("ScreenQA calibration source-shard audit is malformed")
        if (
            shard.get("shard_index") != expected_index
            or shard.get("resume_audit_required") is not True
            or not shard.get("resume_audit_sha256")
        ):
            raise ValueError(
                f"ScreenQA calibration shard {expected_index} resume proof mismatch"
            )
        shard_dir = Path(str(shard["directory"]))
        provenance_path = shard_dir / "rollouts.provenance.json"
        if sha256_file(provenance_path) != shard.get("provenance_sha256"):
            raise ValueError(
                f"ScreenQA calibration shard {expected_index} provenance hash mismatch"
            )
        provenance = _load_json(provenance_path)
        if provenance.get("scientific_status") != EXPECTED_SCIENTIFIC_STATUS:
            raise ValueError(
                f"ScreenQA calibration shard {expected_index} scientific status mismatch"
            )
        shard_states += int(shard["states"])
        shard_records += int(shard["records"])
    if shard_states != EXPECTED_STATES or shard_records != EXPECTED_RECORDS:
        raise ValueError("ScreenQA calibration source-shard totals mismatch")

    records = read_jsonl(rollouts)
    state_counts = Counter(record.state_id for record in records)
    source_ids = {record.source_id for record in records}
    if len(records) != EXPECTED_RECORDS or len(state_counts) != EXPECTED_STATES:
        raise ValueError("ScreenQA calibration merged rollout dimensions mismatch")
    if set(state_counts.values()) != {5}:
        raise ValueError("ScreenQA calibration states lack five sibling records")
    if len(source_ids) != EXPECTED_SOURCES or any(
        not source_id.startswith("screenqa:") for source_id in source_ids
    ):
        raise ValueError("ScreenQA calibration source-component coverage mismatch")
    answer_records = sum(record.action_type == "ANSWER" for record in records)
    zoom_records = sum(record.action_type == "ZOOM" for record in records)
    if answer_records != EXPECTED_STATES or zoom_records != EXPECTED_STATES * 4:
        raise ValueError("ScreenQA calibration ANSWER/ZOOM sibling counts mismatch")
    audit = {
        "passed": True,
        "scientific_status": EXPECTED_SCIENTIFIC_STATUS,
        "rollouts": str(rollouts.resolve()),
        "rollouts_sha256": actual_rollouts_sha256,
        "merge_audit": str(merge_audit.resolve()),
        "merge_audit_sha256": expected_merge_audit_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "bank_code_revision": expected_bank_code_revision,
        "states": len(state_counts),
        "records": len(records),
        "answer_records": answer_records,
        "zoom_records": zoom_records,
        "source_components": len(source_ids),
        "resume_audited_shards": len(source_shards),
        "ranker_training_outcomes_used": True,
        "calibration_outcomes_used": True,
        "formal_outcomes_opened": False,
        "reserve_outcomes_opened": False,
        "untouched_outcomes_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the complete ScreenQA risk-calibration sibling bank"
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--merge-audit", type=Path, required=True)
    parser.add_argument("--expected-merge-audit-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-bank-code-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = verify_rollouts(
        rollouts=args.rollouts,
        expected_rollouts_sha256=args.expected_rollouts_sha256,
        merge_audit=args.merge_audit,
        expected_merge_audit_sha256=args.expected_merge_audit_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_bank_code_revision=args.expected_bank_code_revision,
        output=args.output,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
