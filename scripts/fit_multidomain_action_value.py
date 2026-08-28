#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from beyond_entropy.action_value import fit_multidomain_action_value_model
from beyond_entropy.dataset import read_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _domain_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("domain input must be NAME=ROLLOUTS_JSONL")
    raw_name, raw_path = value.split("=", 1)
    name = raw_name.strip()
    path = Path(raw_path).expanduser().resolve()
    if not name or not path.is_file():
        raise argparse.ArgumentTypeError("domain name and rollout file must exist")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a source-held-out multidomain direct action-value model"
    )
    parser.add_argument(
        "--domain",
        type=_domain_path,
        action="append",
        required=True,
        help="repeat NAME=ROLLOUTS_JSONL for each development-only domain",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument(
        "--alpha",
        type=float,
        action="append",
        dest="alpha_values",
        default=[],
    )
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    domain_paths = dict(args.domain)
    if len(domain_paths) != len(args.domain):
        raise SystemExit("development domain names must be unique")
    records_by_domain = {
        domain: read_jsonl(path) for domain, path in domain_paths.items()
    }
    alpha_values = args.alpha_values or [0.1, 1.0, 10.0, 100.0, 1000.0]
    report, model = fit_multidomain_action_value_model(
        records_by_domain,
        validation_fraction=args.validation_fraction,
        lambda_cost=args.lambda_cost,
        alpha_values=alpha_values,
        seed=args.seed,
    )
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report["run"] = {
        "code_revision": code_revision,
        "development_inputs": {
            domain: {
                "path": str(path),
                "sha256": _sha256(path),
                "records": len(records_by_domain[domain]),
            }
            for domain, path in sorted(domain_paths.items())
        },
        "formal_outcomes_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
