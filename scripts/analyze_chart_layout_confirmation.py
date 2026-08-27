from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from beyond_entropy.candidate_ablation import compare_candidate_sets
from beyond_entropy.dataset import group_by_decision, read_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_manifest_states(path: Path) -> set[str]:
    states = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            state_id = str(value["state_id"])
            if state_id in states:
                raise ValueError(f"duplicate confirmation state: {state_id}")
            states.add(state_id)
    return states


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def primary_row(report: dict[str, object]) -> dict[str, object]:
    comparisons = report["policy_differences"]
    if not isinstance(comparisons, list):
        raise ValueError("candidate report lacks policy differences")
    matches = [
        row
        for row in comparisons
        if isinstance(row, dict)
        and row.get("right_policy") == "uniform_random_zoom_expectation"
    ]
    if len(matches) != 1:
        raise ValueError("candidate report lacks a unique uniform-random comparison")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen chart-layout confirmation")
    parser.add_argument("--baseline-rollouts", type=Path, required=True)
    parser.add_argument("--treatment-rollouts", type=Path, required=True)
    parser.add_argument("--treatment-provenance", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()

    expected_hashes = {
        "baseline_rollouts": "881526ccd3ff03753127307128c84dcf9dfa217f06635934ed2c5bca6d93973c",
        "target_manifest": "d7c96df369259c8c3645bf64c27c220936636c92e359171a50e420344c5ff0bd",
    }
    actual_hashes = {
        "baseline_rollouts": sha256(args.baseline_rollouts),
        "target_manifest": sha256(args.target_manifest),
    }
    if actual_hashes != expected_hashes:
        raise ValueError(f"chart-layout confirmation hash mismatch: {actual_hashes}")
    provenance = read_json(args.treatment_provenance)
    treatment_sha256 = sha256(args.treatment_rollouts)
    expected_provenance: dict[str, object] = {
        "code_revision": "075abc419f1d263fe208eb6539ae348dbdc8f8e9",
        "manifest_sha256": expected_hashes["target_manifest"],
        "output_sha256": treatment_sha256,
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "scorer": "chartqa",
        "proposer": "chart-layout",
        "examples": 2137,
        "completed_examples": 2137,
        "candidate_count": 4,
        "generation_seeds": [0],
        "max_new_tokens": 16,
        "min_pixels": 200704,
        "max_pixels": 602112,
        "attention_implementation": "sdpa",
        "system_prompt": (
            "Answer with only the final answer: a single number, word, or short phrase. "
            "Do not explain."
        ),
        "local_files_only": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": provenance.get(key)}
        for key, expected in expected_provenance.items()
        if provenance.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"chart-layout rollout provenance mismatch: {mismatches}")

    target_states = read_manifest_states(args.target_manifest)
    if len(target_states) != 2137:
        raise ValueError(f"expected 2137 target states, found {len(target_states)}")
    baseline_all = read_jsonl(args.baseline_rollouts)
    baseline = [record for record in baseline_all if record.state_id in target_states]
    treatment = read_jsonl(args.treatment_rollouts)
    if len(baseline) != 10685 or len(treatment) != 10685:
        raise ValueError(
            f"incomplete matched rollouts: baseline={len(baseline)}, treatment={len(treatment)}"
        )
    if {key[0] for key in group_by_decision(baseline)} != target_states:
        raise ValueError("baseline subset does not cover the frozen target")
    if {key[0] for key in group_by_decision(treatment)} != target_states:
        raise ValueError("treatment rollout does not cover the frozen target")

    state_report = compare_candidate_sets(
        baseline,
        treatment,
        lambda_cost=0.05,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=0,
        cluster_by="state_id",
    )
    image_report = compare_candidate_sets(
        baseline,
        treatment,
        lambda_cost=0.05,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=0,
        cluster_by="image_id",
    )
    state_primary = primary_row(state_report)
    image_primary = primary_row(image_report)
    state_metrics = state_primary["metrics"]
    image_metrics = image_primary["metrics"]
    if not isinstance(state_metrics, dict) or not isinstance(image_metrics, dict):
        raise ValueError("primary comparison metrics are malformed")
    state_utility = state_metrics["mean_policy_utility"]
    image_utility = image_metrics["mean_policy_utility"]
    if not isinstance(state_utility, dict) or not isinstance(image_utility, dict):
        raise ValueError("primary utility intervals are malformed")
    criterion = {
        "positive_point_estimate": float(state_utility["estimate"]) > 0.0,
        "state_ci_lower_above_zero": float(state_utility["ci_low"]) > 0.0,
        "image_ci_lower_above_zero": float(image_utility["ci_low"]) > 0.0,
    }
    criterion["passed"] = all(criterion.values())
    report: dict[str, object] = {
        "scientific_status": "image-disjoint frozen chart-layout confirmation",
        "run": {
            "baseline_rollouts": str(args.baseline_rollouts.resolve()),
            "baseline_rollouts_sha256": actual_hashes["baseline_rollouts"],
            "treatment_rollouts": str(args.treatment_rollouts.resolve()),
            "treatment_rollouts_sha256": treatment_sha256,
            "treatment_provenance": str(args.treatment_provenance.resolve()),
            "treatment_provenance_sha256": sha256(args.treatment_provenance),
            "target_manifest": str(args.target_manifest.resolve()),
            "target_manifest_sha256": actual_hashes["target_manifest"],
            "code_revision": os.environ.get("BE_CODE_REVISION"),
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        "input_validation": {
            "states": len(target_states),
            "baseline_records": len(baseline),
            "treatment_records": len(treatment),
            "treatment_protocol_fields": expected_provenance,
        },
        "primary_confirmation_criterion": criterion,
        "state_cluster_comparison": state_report,
        "image_cluster_comparison": image_report,
    }
    write_json(report, args.output_dir / "report.json")
    markdown = "\n".join(
        [
            "# Image-disjoint chart-layout confirmation",
            "",
            f"> Primary criterion passed: **{criterion['passed']}**.",
            "",
            "- Uniform-random utility difference: {:.4f}".format(
                float(state_utility["estimate"])
            ),
            "- State-bootstrap 95% CI: [{:.4f}, {:.4f}]".format(
                float(state_utility["ci_low"]), float(state_utility["ci_high"])
            ),
            "- Image-bootstrap 95% CI: [{:.4f}, {:.4f}]".format(
                float(image_utility["ci_low"]), float(image_utility["ci_high"])
            ),
            "",
        ]
    )
    (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
