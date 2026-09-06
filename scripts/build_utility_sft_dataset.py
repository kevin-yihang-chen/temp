"""Build train/validation utility data from sealed EXISTING development banks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from beyond_entropy.benchmarks import load_manifest, scorer_by_name
from beyond_entropy.dataset import read_jsonl
from beyond_entropy.predictability_features import decoded_rgb_sha256
from beyond_entropy.predictability_matrix_artifacts import atomic_json_write_exclusive, sha256_file
from beyond_entropy.utility_dataset import audit_utility_splits, build_utility_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-sources", type=int, default=0, help="0 = all; hash selection, whole sources")
    parser.add_argument("--aggregation", choices=("single", "mean"), default="single")
    args = parser.parse_args()
    if args.max_sources < 0:
        raise ValueError("max-sources must be nonnegative")
    completion = json.loads(Path(args.completion).read_text())
    if (completion.get("schema") != "predictability_formal_development_role_v1"
            or completion.get("passed") is not True
            or completion.get("role") not in ("train", "validation")):
        raise ValueError("only verified opened development roles are accepted; no test")
    artifacts = completion["artifacts"]
    for name in ("manifest", "rollouts", "rollout_provenance"):
        if sha256_file(artifacts[f"{name}_path"]) != artifacts[name]:
            raise ValueError(f"sealed {name} hash mismatch")
    examples = load_manifest(artifacts["manifest_path"])
    source_ids = sorted({e.state.source_id for e in examples}, key=lambda s: hashlib.sha256(f"utility-subset-v1:{s}".encode()).hexdigest())
    selected = set(source_ids[:args.max_sources] if args.max_sources else source_ids)
    examples = [e for e in examples if e.state.source_id in selected]
    states = {e.state.state_id: e.state for e in examples}
    if len(states) != len(examples):
        raise ValueError("duplicate manifest state")
    records = read_jsonl(artifacts["rollouts_path"])
    if len(records) != completion["sibling_records"]:
        raise ValueError("sealed record coverage mismatch")
    records = [r for r in records if r.state_id in states]
    if {r.state_id for r in records} != set(states):
        raise ValueError("missing manifest siblings")
    truth = {e.state.state_id: e.ground_truth for e in examples}
    scorer = scorer_by_name(completion["benchmark"])
    for record in records:
        if abs(scorer(record.answer_after, truth[record.state_id]) - record.correct_after) > 1e-9:
            raise ValueError("stored outcome disagrees with official scorer")
    hashes = {}
    for e in examples:
        path = e.state.image_path
        if path not in hashes:
            hashes[path] = decoded_rgb_sha256(path)
    samples = build_utility_samples(
        records, states=states,
        rgb_hashes={e.state.state_id: hashes[e.state.image_path] for e in examples},
        benchmark=completion["benchmark"], role=completion["role"], aggregation=args.aggregation,
    )
    payload = {
        "schema": "utility_sft_dataset_v1", "role": completion["role"],
        "benchmark": completion["benchmark"], "formal_test_eligible": False,
        "aggregation": args.aggregation, "samples": [s.to_dict() for s in samples],
        "split_audit": audit_utility_splits(samples),
        "provenance": {
            "completion_path": str(Path(args.completion).resolve()),
            "completion_sha256": sha256_file(args.completion),
            "manifest_sha256": artifacts["manifest"], "rollouts_sha256": artifacts["rollouts"],
            "scorer": completion["benchmark"], "scorer_recomputed": True,
            "selection": "whole-source SHA256 order utility-subset-v1; outcome independent",
            "max_sources": args.max_sources,
        },
    }
    atomic_json_write_exclusive(args.output, payload)
    print(json.dumps({"states": len(samples), "sources": len(selected), "output": args.output,
                      "sha256": sha256_file(args.output), "formal_test_eligible": False}))


if __name__ == "__main__":
    main()
