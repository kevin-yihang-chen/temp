#!/usr/bin/env python3
"""Materialize one frozen Phase-C selector-training seed from the matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.phase_c_training import materialize_phase_c_seed_configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = materialize_phase_c_seed_configs(
        matrix_path=Path(args.matrix), repository_root=Path(args.repository_root),
        seed=args.seed, output_dir=Path(args.output_dir),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

