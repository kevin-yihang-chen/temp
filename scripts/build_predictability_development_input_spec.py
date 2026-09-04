from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from beyond_entropy.predictability_audit import AUDIT_BENCHMARKS
from beyond_entropy.predictability_matrix_artifacts import (
    DEVELOPMENT_INPUT_SCHEMA,
    atomic_json_write_exclusive,
    current_clean_revision,
    sha256_file,
)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a hash-bound development-only matrix input spec"
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    revision = current_clean_revision(args.repo_root)
    protocol = Path(args.protocol).resolve()
    if sha256_file(protocol) != args.expected_protocol_sha256:
        raise ValueError("development spec protocol SHA-256 mismatch")
    run_root = Path(args.run_root).resolve()
    benchmarks: dict[str, Any] = {}
    completion_hashes: dict[str, str] = {}
    for benchmark in AUDIT_BENCHMARKS:
        role_specs: dict[str, Any] = {}
        for role in ("train", "validation"):
            completion = run_root / benchmark / role / "complete.json"
            completion_sha256 = sha256_file(completion)
            report = json.loads(completion.read_text(encoding="utf-8"))
            if (
                not isinstance(report, dict)
                or report.get("schema") != "predictability_formal_development_role_v1"
                or report.get("passed") is not True
                or report.get("benchmark") != benchmark
                or report.get("role") != role
                or report.get("code_revision") != revision
                or report.get("protocol_sha256") != args.expected_protocol_sha256
            ):
                raise ValueError(f"invalid sealed role report: {benchmark}.{role}")
            artifacts = _mapping(
                report.get("artifacts"), name=f"{benchmark}.{role}.artifacts"
            )
            role_spec: dict[str, Any] = {}
            for name in ("manifest", "rollouts", "rollout_provenance", "features"):
                path_key = f"{name}_path"
                path = Path(str(artifacts[path_key])).resolve()
                expected = str(artifacts[name])
                if sha256_file(path) != expected:
                    raise ValueError(
                        f"sealed artifact changed after completion: "
                        f"{benchmark}.{role}.{name}"
                    )
                role_spec[name] = {"path": str(path), "sha256": expected}
            role_specs[role] = role_spec
            completion_hashes[f"{benchmark}.{role}"] = completion_sha256
        benchmarks[benchmark] = role_specs
    spec = {
        "schema": DEVELOPMENT_INPUT_SCHEMA,
        "code_revision": revision,
        "protocol": {
            "path": str(protocol),
            "sha256": args.expected_protocol_sha256,
        },
        "benchmarks": benchmarks,
    }
    atomic_json_write_exclusive(args.output, spec)
    print(
        json.dumps(
            {
                "schema": DEVELOPMENT_INPUT_SCHEMA,
                "output": str(Path(args.output).resolve()),
                "output_sha256": sha256_file(args.output),
                "code_revision": revision,
                "sealed_role_completion_sha256": completion_hashes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
