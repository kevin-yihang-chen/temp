#!/usr/bin/env python3
"""Deterministically evaluate sequential stopping policies and cost frontiers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from beyond_entropy.acquisition_critic import AcquisitionCritic, examples_from_feature_dataset
from beyond_entropy.sequential_metrics import (
    paired_source_bootstrap_delta,
    policy_metrics,
    risk_metrics,
    sequential_diagnostic,
)
from beyond_entropy.sequential_schema import SequentialRolloutRecord
from beyond_entropy.stopping_policy import matched_rate_random_mask


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_records(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(SequentialRolloutRecord.from_dict(json.loads(line)))
    if not rows:
        raise ValueError("sequential rollout file is empty")
    return rows


def top_count_mask(scores, count):
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    selected = set(order[:count])
    return [index in selected for index in range(len(scores))]


def load_selected(checkpoint, target, input_dim):
    spec = checkpoint["selected"][target]
    if int(spec["input_dim"]) != input_dim:
        raise ValueError("checkpoint and feature dimensions differ")
    model = AcquisitionCritic(
        input_dim,
        architecture=spec["architecture"],
        hidden_dim=int(spec["hidden_dim"]),
    )
    model.load_state_dict(spec["state_dict"])
    model.eval()
    return model, spec


def authorize_test(path: Path | None, *, features: Path, checkpoint: Path) -> None:
    if path is None:
        raise ValueError("test evaluation requires a frozen one-shot authorization")
    value = json.loads(path.read_text())
    if (
        value.get("schema") != "sequential_test_freeze_v1"
        or value.get("one_shot") is not True
        or value.get("test_authorized") is not True
        or value.get("features_sha256") != sha256_file(features)
        or value.get("checkpoint_sha256") != sha256_file(checkpoint)
        or int(value.get("bootstrap_samples", 0)) < 10_000
    ):
        raise ValueError("invalid or drifted sequential test freeze")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--critics", required=True)
    parser.add_argument("--dataset-role", choices=("validation", "test"), required=True)
    parser.add_argument("--test-freeze")
    parser.add_argument("--static-voi-scores")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config.get("schema") != "sequential_critic_config_v1":
        raise ValueError("unexpected sequential evaluation config")
    feature_path = Path(args.features).resolve()
    rollout_path = Path(args.rollouts).resolve()
    critic_path = Path(args.critics).resolve()
    if args.dataset_role == "test":
        authorize_test(
            None if args.test_freeze is None else Path(args.test_freeze).resolve(),
            features=feature_path,
            checkpoint=critic_path,
        )
    payload = torch.load(feature_path, map_location="cpu", weights_only=True)
    if payload["metadata"].get("dataset_role") != args.dataset_role:
        raise ValueError("feature dataset role differs from evaluation role")
    examples = examples_from_feature_dataset(payload)
    records_by_id = {item.decision_id: item for item in read_records(rollout_path)}
    ids = [(item.inputs.state_id, item.replicate_id) for item in examples]
    if set(ids) != set(records_by_id) or len(ids) != len(records_by_id):
        raise ValueError("feature and rollout decisions do not exactly match")
    records = [records_by_id[item] for item in ids]
    checkpoint = torch.load(critic_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "sequential_acquisition_critics_v1":
        raise ValueError("unexpected critic checkpoint")
    level = checkpoint["feature_level"]
    features = torch.tensor(
        [item.inputs.feature_vector(level) for item in examples], dtype=torch.float32
    )
    predictions = {}
    for target in ("risk", "gain"):
        model, spec = load_selected(checkpoint, target, features.shape[1])
        normalized = (features - spec["mean"]) / spec["scale"]
        with torch.no_grad():
            raw = model(normalized)
        predictions[target] = torch.sigmoid(raw).tolist() if target == "risk" else raw.tolist()

    static_scores = None
    if args.static_voi_scores:
        raw = json.loads(Path(args.static_voi_scores).read_text())
        static_scores = [float(raw[f"{a}::{b}"]) for a, b in ids]
    lambdas = [float(value) for value in config["lambda_sweep"]]
    report = {
        "schema": "sequential_policy_evaluation_v1",
        "scientific_status": (
            "one_shot_test" if args.dataset_role == "test" else "development_validation"
        ),
        "dataset_role": args.dataset_role,
        "benchmark": payload["metadata"]["benchmark"],
        "test_accessed": args.dataset_role == "test",
        "diagnostic": sequential_diagnostic(records),
        "risk_critic": risk_metrics(
            predictions["risk"], [item.stop_correct for item in records]
        ),
        "frontier": [],
        "static_voi_status": (
            "provided" if static_scores is not None else "unavailable_for_partial-prefix_state"
        ),
        "provenance": {
            "config_sha256": sha256_file(Path(args.config)),
            "features_sha256": sha256_file(feature_path),
            "rollouts_sha256": sha256_file(rollout_path),
            "critics_sha256": sha256_file(critic_path),
            "bootstrap_samples": int(config["bootstrap_samples"]),
            "bootstrap_seed": int(config["bootstrap_seed"]),
        },
    }
    for lambda_cost in lambdas:
        learned = [
            gain - lambda_cost * record.proposed_visual_cost > 0
            for gain, record in zip(predictions["gain"], records)
        ]
        count = sum(learned)
        entropy = top_count_mask([item.stop_entropy for item in records], count)
        confidence = top_count_mask([-item.stop_max_probability for item in records], count)
        margin = top_count_mask([-item.stop_top1_top2_margin for item in records], count)
        random_mask = matched_rate_random_mask(
            ids, rate=count / len(records), seed=int(config["bootstrap_seed"])
        )
        risk_gain = [
            use and risk > float(config["risk_threshold"])
            for use, risk in zip(learned, predictions["risk"])
        ]
        oracle = [item.incremental_utility(lambda_cost) > 0 for item in records]
        masks = {
            "always_stop": [False] * len(records),
            "always_continue": [True] * len(records),
            "random_matched": random_mask,
            "entropy_matched": entropy,
            "confidence_matched": confidence,
            "margin_matched": margin,
            "learned_gain": learned,
            "risk_plus_gain": risk_gain,
            "oracle": oracle,
        }
        if static_scores is not None:
            masks["static_voi_matched"] = top_count_mask(static_scores, count)
        metrics = {
            name: policy_metrics(
                records, mask, lambda_cost=lambda_cost, policy_name=name
            )
            for name, mask in masks.items()
        }
        comparisons = {}

        def utility_stat(subset, mask):
            return float(
                policy_metrics(
                    subset, mask, lambda_cost=lambda_cost, policy_name="bootstrap"
                )["incremental_net_utility"]
            )

        for baseline in (
            "random_matched",
            "entropy_matched",
            "confidence_matched",
            "margin_matched",
            "static_voi_matched",
        ):
            if baseline in masks:
                comparisons[f"learned_gain_minus_{baseline}"] = paired_source_bootstrap_delta(
                    records,
                    learned,
                    masks[baseline],
                    statistic=utility_stat,
                    samples=int(config["bootstrap_samples"]),
                    seed=int(config["bootstrap_seed"]),
                )
        report["frontier"].append(
            {
                "lambda_cost": lambda_cost,
                "policies": metrics,
                "paired_bootstrap": comparisons,
            }
        )

    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6.5, 4.5))
        styles = ("entropy_matched", "confidence_matched", "learned_gain", "risk_plus_gain", "oracle")
        for name in styles:
            xs = [row["policies"][name]["avg_incremental_visual_cost"] for row in report["frontier"]]
            ys = [row["policies"][name]["accuracy"] for row in report["frontier"]]
            plt.plot(xs, ys, marker="o", label=name)
        plt.xlabel("Average incremental visual cost")
        plt.ylabel("Accuracy")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(destination / "accuracy_cost_frontier.png", dpi=180)
        plt.close()

        plt.figure(figsize=(5, 5))
        actual = [item.delta_success for item in records]
        plt.scatter(predictions["gain"], actual, s=10, alpha=0.35)
        plt.axhline(0, color="black", linewidth=0.7)
        plt.axvline(0, color="black", linewidth=0.7)
        plt.xlabel("Predicted gain")
        plt.ylabel("Counterfactual gain")
        plt.tight_layout()
        plt.savefig(destination / "gain_prediction.png", dpi=180)
        plt.close()
    except ModuleNotFoundError:
        report["figure_status"] = "matplotlib_unavailable"
        report_path.write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(report_path), "states": len(records)}), flush=True)


if __name__ == "__main__":
    main()
