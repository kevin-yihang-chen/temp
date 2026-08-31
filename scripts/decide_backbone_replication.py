#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.backbone_replication_decision import decide_backbone_replication


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen Qwen2.5-VL-7B backbone decision"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-report-sha256")
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument(
        "--expected-study-label",
        default="ScreenQA Qwen2.5-VL-7B opened development",
    )
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    result = decide_backbone_replication(
        report=args.report,
        protocol=args.protocol,
        output_dir=args.output_dir,
        expected_report_sha256=args.expected_report_sha256,
        expected_protocol_sha256=args.expected_protocol_sha256,
        expected_study_label=args.expected_study_label,
        code_revision=args.code_revision,
    )
    print(json.dumps({"decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
