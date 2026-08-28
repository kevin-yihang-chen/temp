from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from beyond_entropy.dataset import read_jsonl
from beyond_entropy.schema import ActionRecord


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"manifest row {line_number} is not an object")
            state_id = str(row["state_id"])
            if state_id in rows:
                raise ValueError(f"duplicate manifest state: {state_id}")
            rows[state_id] = row
    return rows


def _record_key(record: ActionRecord) -> tuple[str, str, str]:
    return record.state_id, record.replicate_id, record.action_id


def _index_records(records: Sequence[ActionRecord]) -> dict[tuple[str, str, str], ActionRecord]:
    indexed: dict[tuple[str, str, str], ActionRecord] = {}
    for record in records:
        key = _record_key(record)
        if key in indexed:
            raise ValueError(f"duplicate rollout key: {key}")
        indexed[key] = record
    return indexed


def _backend_metadata(record: ActionRecord) -> Mapping[str, Any]:
    name = "baseline_backend" if record.action_type == "ANSWER" else "action_backend"
    value = record.metadata.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing {name} metadata for {_record_key(record)}")
    return value


def _max_abs_difference(reference: Sequence[float], candidate: Sequence[float]) -> float:
    if len(reference) != len(candidate):
        return 1.0
    return max((abs(left - right) for left, right in zip(reference, candidate)), default=0.0)


def compare_replays(
    reference_records: Sequence[ActionRecord],
    candidate_records: Sequence[ActionRecord],
    reference_manifest: Mapping[str, Mapping[str, Any]],
    candidate_manifest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(reference_manifest) != set(candidate_manifest):
        raise ValueError("reference and candidate manifest state IDs differ")
    for state_id in reference_manifest:
        reference_row = reference_manifest[state_id]
        candidate_row = candidate_manifest[state_id]
        if (
            reference_row["image_id"] != candidate_row["image_id"]
            or reference_row["source_id"] != candidate_row["source_id"]
            or reference_row["target"] != candidate_row["target"]
            or reference_row["stratum"] != candidate_row["stratum"]
        ):
            raise ValueError(f"manifest identity or target mismatch: {state_id}")
        if reference_row["question"] != candidate_row.get("model_prompt"):
            raise ValueError(f"candidate backend prompt differs from reference: {state_id}")
        if candidate_row["question"] == candidate_row.get("model_prompt"):
            raise ValueError(f"candidate gate context is not isolated: {state_id}")

    reference = _index_records(reference_records)
    candidate = _index_records(candidate_records)
    if set(reference) != set(candidate):
        raise ValueError("reference and candidate rollout keys differ")

    answer_before_mismatches = 0
    answer_after_mismatches = 0
    correctness_mismatches = 0
    entropy_mismatches = 0
    generated_token_count_mismatches = 0
    token_entropy_mismatches = 0
    max_entropy_abs_difference = 0.0
    max_token_entropy_abs_difference = 0.0
    for key in sorted(reference):
        left = reference[key]
        right = candidate[key]
        candidate_row = candidate_manifest[right.state_id]
        reference_row = reference_manifest[left.state_id]
        if left.question != str(reference_row["question"]):
            raise ValueError(f"reference record question mismatch: {key}")
        if right.question != str(candidate_row["question"]):
            raise ValueError(f"candidate record gate context mismatch: {key}")
        structural_left = (
            left.image_id,
            left.source_id,
            left.replicate_id,
            left.generation_seed,
            left.action_id,
            left.action_type,
            left.candidate_bbox,
            left.tool_cost,
            dict(left.pre_action_features),
        )
        structural_right = (
            right.image_id,
            right.source_id,
            right.replicate_id,
            right.generation_seed,
            right.action_id,
            right.action_type,
            right.candidate_bbox,
            right.tool_cost,
            dict(right.pre_action_features),
        )
        if structural_left != structural_right:
            raise ValueError(f"rollout structure mismatch: {key}")
        answer_before_mismatches += int(left.answer_before != right.answer_before)
        answer_after_mismatches += int(left.answer_after != right.answer_after)
        correctness_mismatches += int(
            left.correct_before != right.correct_before
            or left.correct_after != right.correct_after
        )
        entropy_difference = max(
            abs(left.entropy_before - right.entropy_before),
            abs(left.entropy_after - right.entropy_after),
        )
        max_entropy_abs_difference = max(max_entropy_abs_difference, entropy_difference)
        entropy_mismatches += int(entropy_difference != 0.0)

        left_metadata = _backend_metadata(left)
        right_metadata = _backend_metadata(right)
        expected_prompt_hash = hashlib.sha256(
            str(candidate_row["model_prompt"]).encode()
        ).hexdigest()
        if (
            right_metadata.get("input_text_sha256") != expected_prompt_hash
            or right_metadata.get("distinct_model_prompt") is not True
        ):
            raise ValueError(f"candidate backend prompt metadata mismatch: {key}")
        left_generated = int(left_metadata["generated_tokens"])
        right_generated = int(right_metadata["generated_tokens"])
        generated_token_count_mismatches += int(left_generated != right_generated)
        left_token_entropies = [
            float(value) for value in left_metadata["normalized_token_entropies"]
        ]
        right_token_entropies = [
            float(value) for value in right_metadata["normalized_token_entropies"]
        ]
        token_difference = _max_abs_difference(
            left_token_entropies,
            right_token_entropies,
        )
        max_token_entropy_abs_difference = max(
            max_token_entropy_abs_difference,
            token_difference,
        )
        token_entropy_mismatches += int(token_difference != 0.0)

    mismatch_counts = {
        "answer_before": answer_before_mismatches,
        "answer_after": answer_after_mismatches,
        "correctness": correctness_mismatches,
        "aggregate_entropy": entropy_mismatches,
        "generated_token_count": generated_token_count_mismatches,
        "token_entropy_sequence": token_entropy_mismatches,
    }
    return {
        "scientific_status": "prompt-isolation replay audit",
        "states": len(candidate_manifest),
        "records": len(candidate),
        "manifest_identity_and_target_match": True,
        "backend_prompt_byte_match": True,
        "candidate_gate_context_isolated": True,
        "candidate_backend_prompt_metadata_match": True,
        "mismatch_counts": mismatch_counts,
        "max_aggregate_entropy_abs_difference": max_entropy_abs_difference,
        "max_token_entropy_abs_difference": max_token_entropy_abs_difference,
        "passed": not any(mismatch_counts.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare prompt-identical ChartQAPro pilot replays",
    )
    parser.add_argument("--reference-rollouts", type=Path, required=True)
    parser.add_argument("--candidate-rollouts", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_replays(
        read_jsonl(args.reference_rollouts),
        read_jsonl(args.candidate_rollouts),
        _read_manifest(args.reference_manifest),
        _read_manifest(args.candidate_manifest),
    )
    report["inputs"] = {
        "reference_rollouts_sha256": _sha256(args.reference_rollouts),
        "candidate_rollouts_sha256": _sha256(args.candidate_rollouts),
        "reference_manifest_sha256": _sha256(args.reference_manifest),
        "candidate_manifest_sha256": _sha256(args.candidate_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": _sha256(args.output),
                "passed": report["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
