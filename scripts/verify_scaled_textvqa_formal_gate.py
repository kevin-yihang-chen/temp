from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from export_textvqa_train_scale_formal import (
    _load_mapping,
    _sha256,
    _verify_freeze_components,
)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that the scaled TextVQA formal gate remains frozen"
    )
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--expected-policy-freeze-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    args = parser.parse_args()

    for path, expected, name in (
        (args.policy_freeze, args.expected_policy_freeze_sha256, "policy freeze"),
        (args.model, args.expected_model_sha256, "model"),
        (args.manifest, args.expected_manifest_sha256, "manifest"),
        (args.audit, args.expected_audit_sha256, "formal audit"),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
    freeze = _load_mapping(args.policy_freeze, "policy freeze")
    _verify_freeze_components(freeze)
    frozen_model = _require_mapping(
        _require_mapping(freeze.get("artifacts"), "freeze artifacts").get(
            "calibrated_model"
        ),
        "frozen calibrated model",
    )
    if (
        Path(str(frozen_model.get("path", ""))).resolve() != args.model.resolve()
        or frozen_model.get("sha256") != args.expected_model_sha256
    ):
        raise ValueError("submitted model differs from the policy freeze")

    provenance_path = args.manifest.parent / "manifest.provenance.json"
    provenance = _load_mapping(provenance_path, "formal manifest provenance")
    if provenance.get("manifest_sha256") != args.expected_manifest_sha256:
        raise ValueError("formal manifest provenance hash mismatch")
    selection = _require_mapping(
        provenance.get("selection_metadata"), "manifest selection metadata"
    )
    if (
        selection.get("policy_freeze_sha256")
        != args.expected_policy_freeze_sha256
        or selection.get("role") != "formal_test"
        or selection.get("selection_uses_targets") is not False
    ):
        raise ValueError("formal manifest is not bound to the frozen policy")
    audit = _load_mapping(args.audit, "formal audit")
    formal_audit = _require_mapping(audit.get("formal"), "formal audit payload")
    if (
        audit.get("passed") is not True
        or audit.get("policy_freeze_sha256") != args.expected_policy_freeze_sha256
        or formal_audit.get("manifest_sha256") != args.expected_manifest_sha256
    ):
        raise ValueError("formal audit is not bound to the frozen policy and manifest")
    print(
        json.dumps(
            {
                "passed": True,
                "policy_freeze_sha256": args.expected_policy_freeze_sha256,
                "model_sha256": args.expected_model_sha256,
                "manifest_sha256": args.expected_manifest_sha256,
                "audit_sha256": args.expected_audit_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
