#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = Path("/userhome/cs3/yihangc/Documents/runtime/vtool-action-credit-g1")
EXPECTED_VERSIONS = {
    "torch": "2.9.0+cu128",
    "transformers": "4.57.6",
    "vllm": "0.12.0",
    "ray": "2.58.0",
    "tensordict": "0.10.0",
}
EXPECTED_UPSTREAM_COMMIT = "d2aa28353ec10c7f91b39f502925003a81d6982d"


def _git(runtime: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=runtime,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _resolved_module_file(module: Any, *, name: str) -> str:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError(f"imported module {name!r} has no filesystem origin")
    return str(Path(module_file).resolve(strict=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed import and version audit for action-credit G1."
    )
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = args.runtime.resolve(strict=True)
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "src"))

    versions: dict[str, str] = {}
    import_paths: dict[str, str] = {}
    import_errors: dict[str, str] = {}
    for name in (*EXPECTED_VERSIONS, "verl"):
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
            import_paths[name] = _resolved_module_file(module, name=name)
        except Exception as exc:  # fail-closed report retains the exact exception
            import_errors[name] = f"{type(exc).__name__}: {exc}"

    integration_error = None
    try:
        integration = importlib.import_module(
            "integrations.vtool_action_credit.paired_vtool"
        )
        integration_path = _resolved_module_file(
            integration, name="integrations.vtool_action_credit.paired_vtool"
        )
    except Exception as exc:
        integration_error = f"{type(exc).__name__}: {exc}"
        integration_path = ""

    upstream_commit = _git(runtime, "rev-parse", "HEAD")
    upstream_status = _git(runtime, "status", "--short")
    expected_modified = {
        "M verl/experimental/agent_loop/agent_loop.py",
        "M verl/trainer/ppo/ray_trainer.py",
    }
    actual_modified = {
        line.strip() for line in upstream_status.splitlines() if line.strip()
    }
    checks: dict[str, Any] = {
        "all_required_imports_succeeded": not import_errors,
        "integration_import_succeeded": integration_error is None,
        "upstream_commit_matches": upstream_commit == EXPECTED_UPSTREAM_COMMIT,
        "runtime_has_only_expected_patch": actual_modified == expected_modified,
        "verl_imports_from_runtime": import_paths.get("verl", "").startswith(
            str(runtime)
        ),
        "versions_match": all(
            versions.get(name) == expected
            for name, expected in EXPECTED_VERSIONS.items()
        ),
    }
    report = {
        "schema": "vtool_action_credit_g1_environment_audit_v1",
        "python": sys.version,
        "runtime": str(runtime),
        "upstream_commit": upstream_commit,
        "upstream_status": upstream_status.splitlines(),
        "versions": versions,
        "import_paths": import_paths,
        "import_errors": import_errors,
        "integration_path": integration_path,
        "integration_error": integration_error,
        "checks": checks,
        "decision": (
            "vtool_action_credit_g1_import_gate_passed"
            if all(checks.values())
            else "vtool_action_credit_g1_import_gate_failed"
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
