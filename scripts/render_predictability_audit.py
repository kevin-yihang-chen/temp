from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from beyond_entropy.predictability_matrix_artifacts import load_hashed_json
from beyond_entropy.predictability_verdict import (
    classify_formal_report,
    render_predictability_audit,
    write_predictability_audit,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render the terminal predictability verdict from one formal report"
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    _, report = load_hashed_json(
        args.report,
        expected_sha256=args.expected_report_sha256,
        schema="predictability_matrix_report_v3",
    )
    markdown = render_predictability_audit(
        report, report_sha256=args.expected_report_sha256
    )
    write_predictability_audit(Path(args.output), markdown)
    print(
        json.dumps(
            {
                "schema": "predictability_audit_terminal_artifact_v1",
                "verdict": classify_formal_report(report).value,
                "output": str(Path(args.output).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
