"""Freeze one immutable three-domain development-pilot arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_entropy.predictability_matrix_artifacts import atomic_json_write_exclusive, sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True); parser.add_argument("--bundle",required=True)
    parser.add_argument("--output-root",required=True); parser.add_argument("--plan",required=True)
    args=parser.parse_args(); root=Path(__file__).resolve().parents[1]
    config=json.loads(Path(args.config).read_text())
    if (config.get("scope")!="three_domain_development_pilot"
            or config.get("test_authorized") is not False):
        raise ValueError("only development pilot config accepted")
    bundle=json.loads(Path(args.bundle).read_text())
    if (bundle.get("test_data_present") is not False
            or bundle.get("formal_test_eligible") is not False
            or bundle.get("split_audit",{}).get("passed") is not True):
        raise ValueError("invalid development bundle")
    script_names=("train_utility_sft_development.py","execute_utility_sft_development.py",
                  "slurm_utility_sft_development.sh","freeze_utility_sft_development_pilot.py")
    paths=sorted((root/"src/beyond_entropy").glob("*.py"))+[root/"scripts"/n for n in script_names]
    payload={"schema":"utility_sft_development_pilot_plan_v1","test_authorized":False,
        "method":config["method"],
        "config":{"path":str(Path(args.config).resolve()),"sha256":sha256_file(args.config)},
        "bundle":{"path":str(Path(args.bundle).resolve()),"sha256":sha256_file(args.bundle)},
        "code_hashes":{str(p.relative_to(root)):sha256_file(p) for p in paths},
        "output_root":str(Path(args.output_root).resolve()),"gpu":"1 H800",
        "maximum_gpu_hours":1.0,"formal_claim_eligible":False,
        "resource_rationale":"Three independent matched arms run concurrently on three H800s in one allowed job. Measured single-arm sanity was 3m53s; pilot adds source loading and held-out validation, with 60m fail-closed cap. This shortens wall time without changing aggregate GPU-hours or merging model states."}
    atomic_json_write_exclusive(args.plan,payload); print(sha256_file(args.plan))


if __name__=="__main__": main()
