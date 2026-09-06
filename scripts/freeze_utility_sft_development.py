"""Hash-bind all six Utility-SFT development datasets and audit their splits.

This command has no test input. It validates one benchmark/role file at a time
so that large diagnostic outcome payloads do not all reside in memory together.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Sequence

from beyond_entropy.predictability_audit import SplitIdentity, audit_split_disjointness
from beyond_entropy.predictability_matrix_artifacts import atomic_json_write_exclusive, sha256_file
from beyond_entropy.utility_dataset import load_utility_development


BENCHMARKS = ("chartqa", "docvqa", "hrbench")
ROLES = ("train", "validation")


def freeze_development_dataset(paths: Sequence[str], output: str) -> dict[str, object]:
    if len(paths) != len(BENCHMARKS) * len(ROLES):
        raise ValueError("exactly six benchmark/role development datasets required")
    inventory: dict[str, object] = {}
    identities: list[SplitIdentity] = []
    assignments: dict[str, str] = {}
    image_roles: dict[str, set[str]] = {}
    total_states = 0
    total_sources: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        header = json.loads(path.read_text(encoding="utf-8"))
        benchmark, role = str(header.get("benchmark")), str(header.get("role"))
        if benchmark not in BENCHMARKS or role not in ROLES:
            raise ValueError("dataset is not one of the six required development roles")
        if (benchmark, role) in seen_pairs:
            raise ValueError("duplicate benchmark/role dataset")
        seen_pairs.add((benchmark, role))
        samples = load_utility_development(path, role=role)
        if len(samples) != len(header["samples"]):
            raise ValueError("loaded sample coverage mismatch")
        sources = {sample.inputs.state.source_id for sample in samples}
        positive_states = sum(max(sample.gains) > 0 for sample in samples)
        harmful_states = sum(min(sample.gains) < 0 for sample in samples)
        for sample in samples:
            state = sample.inputs.state
            item_id = f"{benchmark}:{state.state_id}"
            if item_id in assignments:
                raise ValueError("state duplicated across development datasets")
            assignments[item_id] = role
            # Prefix sources by benchmark: equal opaque IDs from unrelated public
            # datasets are not the same source. RGB hashes remain global.
            identities.append(
                SplitIdentity(item_id, f"{benchmark}:{state.source_id}", sample.image_rgb_sha256)
            )
            image_roles.setdefault(f"{benchmark}:{state.image_id}", set()).add(role)
        inventory[f"{benchmark}.{role}"] = {
            "path": str(path), "sha256": sha256_file(path),
            "states": len(samples), "sources": len(sources),
            "positive_gain_states": positive_states,
            "harmful_gain_states": harmful_states,
            "actions_per_state": sorted({len(sample.gains) for sample in samples}),
        }
        total_states += len(samples)
        total_sources.update(f"{benchmark}:{source}" for source in sources)
        del samples, header
        gc.collect()
    expected = {(b, r) for b in BENCHMARKS for r in ROLES}
    if seen_pairs != expected:
        raise ValueError(f"missing development roles: {sorted(expected-seen_pairs)}")
    if any(len(roles) > 1 for roles in image_roles.values()):
        raise ValueError("image ID crosses train/validation roles")
    split_audit = audit_split_disjointness(identities, assignments)
    payload: dict[str, object] = {
        "schema": "utility_sft_development_bundle_v1",
        "formal_test_eligible": False,
        "test_data_present": False,
        "benchmarks": list(BENCHMARKS), "roles": list(ROLES),
        "inventory": inventory, "states": total_states,
        "sources": len(total_sources), "split_audit": split_audit,
        "label_definition": "correct_after(action)-correct_after(ANSWER); no visual cost",
        "inference_input_contract": "UtilityInputs strict allowlist; labels/outcomes excluded",
    }
    atomic_json_write_exclusive(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = freeze_development_dataset(args.dataset, args.output)
    print(json.dumps({
        "output": str(Path(args.output).resolve()), "sha256": sha256_file(args.output),
        "states": payload["states"], "sources": payload["sources"],
        "split_audit": payload["split_audit"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
