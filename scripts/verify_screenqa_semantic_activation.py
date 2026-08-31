#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}")
EXPECTED_FEATURE_MODES = ["context-geometry", "spatial-context-geometry"]
EXPECTED_RECORDS = 72_555
EXPECTED_DECISIONS = 14_511
EXPECTED_SOURCES = 1_510


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def require_hash(path: Path, expected: str, name: str) -> str:
    if SHA256.fullmatch(expected) is None:
        raise ValueError(f"malformed expected SHA-256 for {name}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"ScreenQA semantic activation {name} SHA-256 mismatch")
    return actual


def verify_checksum_bundle(candidate_dir: Path) -> str:
    checksum_path = candidate_dir / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError("ScreenQA low-capacity candidate checksum bundle missing")
    entries = checksum_path.read_text(encoding="utf-8").splitlines()
    if not entries:
        raise ValueError("ScreenQA low-capacity candidate checksum bundle is empty")
    seen: set[str] = set()
    for line in entries:
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or SHA256.fullmatch(parts[0]) is None:
            raise ValueError("malformed ScreenQA candidate checksum entry")
        filename = parts[1].strip()
        if filename in seen or Path(filename).name != filename:
            raise ValueError("unsafe or duplicate ScreenQA candidate checksum entry")
        seen.add(filename)
        path = candidate_dir / filename
        if not path.is_file() or sha256_file(path) != parts[0]:
            raise ValueError(f"ScreenQA candidate checksum mismatch: {filename}")
    if seen != {"candidate.audit.json"}:
        raise ValueError("failed low-capacity candidate bundle must contain only its audit")
    return sha256_file(checksum_path)


def require_sealed(paths: Sequence[Path]) -> list[str]:
    sealed: list[str] = []
    for path in paths:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"ScreenQA protected output is already materialized: {path}")
        sealed.append(str(path.resolve()))
    return sealed


def verify_activation(
    *,
    candidate_dir: Path,
    expected_candidate_audit_sha256: str,
    ranker_rollouts: Path,
    expected_ranker_rollouts_sha256: str,
    ranker_input_audit: Path,
    expected_ranker_input_audit_sha256: str,
    v1_protocol: Path,
    expected_v1_protocol_sha256: str,
    v2_protocol: Path,
    expected_v2_protocol_sha256: str,
    sealed_output_dirs: Sequence[Path],
    expected_code_revision: str,
) -> dict[str, Any]:
    candidate_audit_path = candidate_dir / "candidate.audit.json"
    candidate_audit_sha256 = require_hash(
        candidate_audit_path,
        expected_candidate_audit_sha256,
        "candidate audit",
    )
    candidate_bundle_sha256 = verify_checksum_bundle(candidate_dir)
    ranker_rollouts_sha256 = require_hash(
        ranker_rollouts,
        expected_ranker_rollouts_sha256,
        "ranker rollouts",
    )
    ranker_input_audit_sha256 = require_hash(
        ranker_input_audit,
        expected_ranker_input_audit_sha256,
        "ranker input audit",
    )
    v1_protocol_sha256 = require_hash(
        v1_protocol, expected_v1_protocol_sha256, "v1 protocol"
    )
    v2_protocol_sha256 = require_hash(
        v2_protocol, expected_v2_protocol_sha256, "v2 protocol"
    )

    candidate = load_json(candidate_audit_path)
    expected_gate = {
        "protocol_applied": True,
        "protocol_sha256": v1_protocol_sha256,
        "registered_feature_modes": EXPECTED_FEATURE_MODES,
        "selection_reason": "no_registered_candidate_is_eligible",
        "candidate_frozen": False,
        "semantic_escalation_required": True,
        "calibration_outcomes_opened": False,
        "formal_outcomes_opened": False,
        "reserve_outcomes_opened": False,
    }
    for key, expected in expected_gate.items():
        if candidate.get(key) != expected:
            raise ValueError(f"ScreenQA semantic activation candidate {key} mismatch")
    candidates = candidate.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("ScreenQA semantic activation requires exactly two v1 candidates")
    by_mode: dict[str, Mapping[str, Any]] = {}
    for row in candidates:
        if not isinstance(row, Mapping):
            raise ValueError("malformed ScreenQA v1 candidate row")
        mode = str(row.get("feature_mode", ""))
        if mode in by_mode:
            raise ValueError("duplicate ScreenQA v1 candidate feature mode")
        by_mode[mode] = row
    if list(sorted(by_mode)) != sorted(EXPECTED_FEATURE_MODES):
        raise ValueError("ScreenQA v1 candidate feature modes mismatch")
    code_revisions: set[str] = set()
    rollout_hashes: set[str] = set()
    for mode in EXPECTED_FEATURE_MODES:
        row = by_mode[mode]
        if row.get("eligible") is not False:
            raise ValueError(f"ScreenQA v1 candidate remains eligible: {mode}")
        if row.get("tail_selection_status") == "selected_non_degenerate_safe_threshold":
            raise ValueError(f"ScreenQA v1 candidate has a selected safe tail: {mode}")
        code_revisions.add(str(row.get("code_revision", "")))
        rollout_hashes.add(str(row.get("rollouts_sha256", "")))
    if len(code_revisions) != 1 or "" in code_revisions:
        raise ValueError("ScreenQA v1 candidate code revisions differ or are missing")
    if rollout_hashes != {ranker_rollouts_sha256}:
        raise ValueError("ScreenQA v1 candidates are not bound to the ranker bank")

    input_audit = load_json(ranker_input_audit)
    expected_input = {
        "passed": True,
        "rollouts_sha256": ranker_rollouts_sha256,
        "records": EXPECTED_RECORDS,
        "states": EXPECTED_DECISIONS,
        "source_components": EXPECTED_SOURCES,
        "calibration_outcomes_opened": False,
        "formal_outcomes_opened": False,
        "reserve_outcomes_opened": False,
    }
    for key, expected in expected_input.items():
        if input_audit.get(key) != expected:
            raise ValueError(f"ScreenQA semantic activation input audit {key} mismatch")
    with ranker_rollouts.open("rb") as handle:
        records = sum(1 for _ in handle)
    if records != EXPECTED_RECORDS:
        raise ValueError("ScreenQA semantic activation rollout line count mismatch")

    sealed = require_sealed(sealed_output_dirs)
    if re.fullmatch(r"[0-9a-f]{40,64}", expected_code_revision) is None:
        raise ValueError("ScreenQA semantic activation code revision is malformed")
    return {
        "passed": True,
        "semantic_escalation_activated": True,
        "scientific_status": (
            "conditional v2 semantic ranker activated only after both frozen v1 "
            "low-capacity candidates failed; protected outcomes remain sealed"
        ),
        "candidate_audit": str(candidate_audit_path.resolve()),
        "candidate_audit_sha256": candidate_audit_sha256,
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "failed_feature_modes": EXPECTED_FEATURE_MODES,
        "v1_candidate_code_revision": next(iter(code_revisions)),
        "semantic_code_revision": expected_code_revision,
        "ranker_rollouts": str(ranker_rollouts.resolve()),
        "ranker_rollouts_sha256": ranker_rollouts_sha256,
        "ranker_input_audit": str(ranker_input_audit.resolve()),
        "ranker_input_audit_sha256": ranker_input_audit_sha256,
        "records": EXPECTED_RECORDS,
        "decisions": EXPECTED_DECISIONS,
        "sources": EXPECTED_SOURCES,
        "v1_protocol": str(v1_protocol.resolve()),
        "v1_protocol_sha256": v1_protocol_sha256,
        "v2_protocol": str(v2_protocol.resolve()),
        "v2_protocol_sha256": v2_protocol_sha256,
        "calibration_outcomes_opened": False,
        "formal_outcomes_opened": False,
        "reserve_outcomes_opened": False,
        "protected_output_dirs": sealed,
    }


def write_audit(path: Path, payload: Mapping[str, Any], *, resume: bool) -> None:
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not resume or path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError("ScreenQA semantic activation audit already exists or drifted")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError("ScreenQA semantic activation staging file exists")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the frozen conditional ScreenQA semantic activation gate"
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--expected-candidate-audit-sha256", required=True)
    parser.add_argument("--ranker-rollouts", type=Path, required=True)
    parser.add_argument("--expected-ranker-rollouts-sha256", required=True)
    parser.add_argument("--ranker-input-audit", type=Path, required=True)
    parser.add_argument("--expected-ranker-input-audit-sha256", required=True)
    parser.add_argument("--v1-protocol", type=Path, required=True)
    parser.add_argument("--expected-v1-protocol-sha256", required=True)
    parser.add_argument("--v2-protocol", type=Path, required=True)
    parser.add_argument("--expected-v2-protocol-sha256", required=True)
    parser.add_argument("--sealed-output-dir", type=Path, action="append", default=[])
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    audit = verify_activation(
        candidate_dir=args.candidate_dir,
        expected_candidate_audit_sha256=args.expected_candidate_audit_sha256,
        ranker_rollouts=args.ranker_rollouts,
        expected_ranker_rollouts_sha256=args.expected_ranker_rollouts_sha256,
        ranker_input_audit=args.ranker_input_audit,
        expected_ranker_input_audit_sha256=args.expected_ranker_input_audit_sha256,
        v1_protocol=args.v1_protocol,
        expected_v1_protocol_sha256=args.expected_v1_protocol_sha256,
        v2_protocol=args.v2_protocol,
        expected_v2_protocol_sha256=args.expected_v2_protocol_sha256,
        sealed_output_dirs=args.sealed_output_dir,
        expected_code_revision=args.expected_code_revision,
    )
    write_audit(args.output, audit, resume=args.resume)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
