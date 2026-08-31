#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from beyond_entropy.proxy_alignment_figure import (
    load_audit_figure_data,
    metric_rows,
    render_proxy_alignment_figure,
    write_metric_csv,
    write_provenance,
)


def _mapping(values: list[str], *, argument: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{argument} must use LABEL=VALUE")
        label, item = value.split("=", 1)
        if not label or not item or label in result:
            raise ValueError(f"invalid or duplicate {argument} label: {label!r}")
        result[label] = item
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render hash-bound proxy-alignment paper panels."
    )
    parser.add_argument("--report", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument(
        "--expected-sha256", action="append", required=True, metavar="LABEL=SHA256"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basename", default="proxy-alignment-v1")
    args = parser.parse_args()

    reports = _mapping(args.report, argument="--report")
    expected = _mapping(args.expected_sha256, argument="--expected-sha256")
    if reports.keys() != expected.keys():
        raise ValueError("report and expected-SHA labels must match exactly")
    if not args.basename or Path(args.basename).name != args.basename:
        raise ValueError("basename must be one non-empty path component")

    data = [
        load_audit_figure_data(
            label=label,
            report=Path(report),
            expected_sha256=expected[label],
        )
        for label, report in reports.items()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = args.output_dir / f"{args.basename}.pdf"
    output_png = args.output_dir / f"{args.basename}.png"
    metric_csv = args.output_dir / f"{args.basename}.csv"
    provenance = args.output_dir / f"{args.basename}.provenance.json"
    for path in (output_pdf, output_png, metric_csv, provenance):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    write_metric_csv(metric_csv, metric_rows(data))
    render_proxy_alignment_figure(data, output_pdf=output_pdf, output_png=output_png)
    write_provenance(
        provenance,
        data=data,
        output_pdf=output_pdf,
        output_png=output_png,
        metric_csv=metric_csv,
    )
    print(
        f"proxy_alignment_figure={output_pdf} reports={len(data)} "
        f"provenance={provenance}"
    )


if __name__ == "__main__":
    main()
