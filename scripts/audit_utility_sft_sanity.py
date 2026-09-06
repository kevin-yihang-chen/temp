"""Audit immutable overfit reports with objective-identifiable gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.predictability_matrix_artifacts import atomic_json_write_exclusive, sha256_file
from beyond_entropy.utility_training import required_sanity_checks, sanity_passed


def audit(reports: list[str], output: str) -> dict:
    if len(reports) != 3:
        raise ValueError("exactly the three matched SFT reports are required")
    arms = {}
    for raw in reports:
        path = Path(raw).resolve()
        report = json.loads(path.read_text())
        method = report.get("method")
        if (report.get("schema") != "utility_sft_train_sanity_v1"
                or report.get("test_accessed") is not False
                or report.get("formal_claim_eligible") is not False
                or method not in ("format", "best_action", "utility")
                or method in arms):
            raise ValueError("invalid or duplicate sanity report")
        selector = path.parent / "selector.pt"
        if sha256_file(selector) != report.get("selector_sha256"):
            raise ValueError("selector/report hash mismatch")
        engineering = report.get("scientific_status") == "engineering_only"
        arms[method] = {
            "report": str(path), "report_sha256": sha256_file(path),
            "selector": str(selector), "selector_sha256": sha256_file(selector),
            "original_runner_overfit_passed": report.get("overfit_passed"),
            "objective_required_checks": list(required_sanity_checks(method)),
            "checks": report["checks"],
            "objective_consistent_overfit_passed": sanity_passed(
                method, report["checks"], engineering=engineering
            ),
        }
    if set(arms) != {"format", "best_action", "utility"}:
        raise ValueError("all three methods required")
    result = {
        "schema": "utility_sft_sanity_audit_v1",
        "formal_claim_eligible": False, "test_accessed": False,
        "all_objective_consistent_gates_passed": all(
            row["objective_consistent_overfit_passed"] for row in arms.values()
        ),
        "arms": arms,
        "audit_note": (
            "The original best_action runner incorrectly required signed gain separation. "
            "One-hot CE identifies only the winning class; immutable raw report is retained, "
            "and no rerun or output modification was performed."
        ),
    }
    atomic_json_write_exclusive(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.report, args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "sha256": sha256_file(args.output),
                      "passed": result["all_objective_consistent_gates_passed"]}))


if __name__ == "__main__":
    main()
