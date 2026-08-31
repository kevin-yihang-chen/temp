from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256sums(directory: Path) -> None:
    sums_path = directory / "SHA256SUMS"
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = directory / relative.strip()
        if sha256_file(path) != expected:
            raise RuntimeError(f"SHA256SUMS mismatch for {path}")


def verify_manifest(
    manifest_dir: Path,
    *,
    expected_manifest_sha256: str,
    expected_audit_sha256: str,
    expected_states: int,
) -> dict[str, object]:
    verify_sha256sums(manifest_dir)
    manifest_path = manifest_dir / "manifest.jsonl"
    audit_path = manifest_dir / "manifest.audit.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("ScreenQA manifest SHA-256 mismatch")
    if sha256_file(audit_path) != expected_audit_sha256:
        raise RuntimeError("ScreenQA manifest audit SHA-256 mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_sealed = {
        "risk_calibration_opened": False,
        "formal_test_opened": False,
        "reserve_opened": False,
        "untouched_opened": False,
        "official_validation_test_opened": False,
    }
    if any(audit.get(key) != value for key, value in expected_sealed.items()):
        raise RuntimeError("ScreenQA non-ranker role is no longer sealed")
    manifest_audit = audit.get("manifest")
    if not isinstance(manifest_audit, dict):
        raise RuntimeError("ScreenQA manifest audit payload is malformed")
    if audit.get("passed") is not True:
        raise RuntimeError("ScreenQA manifest audit did not pass")
    if manifest_audit.get("manifest_sha256") != expected_manifest_sha256:
        raise RuntimeError("ScreenQA audit-to-manifest hash binding mismatch")
    if manifest_audit.get("count") != expected_states:
        raise RuntimeError("ScreenQA manifest state count mismatch")
    if manifest_audit.get("scorer") != "screenqa":
        raise RuntimeError("ScreenQA manifest scorer mismatch")
    if audit.get("selected_rico_images") != 6007:
        raise RuntimeError("ScreenQA selected RICO identity count mismatch")
    if audit.get("unique_decoded_rgb_images") != 5993:
        raise RuntimeError("ScreenQA unique RGB identity count mismatch")
    return {
        "passed": True,
        "manifest_sha256": expected_manifest_sha256,
        "audit_sha256": expected_audit_sha256,
        "states": expected_states,
        "selected_rico_images": audit["selected_rico_images"],
        "unique_decoded_rgb_images": audit["unique_decoded_rgb_images"],
        "selected_source_components": audit["selected_source_components"],
        "sealed_roles": sorted(expected_sealed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the frozen ScreenQA ranker manifest and sealed-role audit"
    )
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--expected-states", type=int, required=True)
    args = parser.parse_args()
    result = verify_manifest(
        args.manifest_dir,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_audit_sha256=args.expected_audit_sha256,
        expected_states=args.expected_states,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
