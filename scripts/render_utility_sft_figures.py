"""Render action-ranking and accuracy-cost figures from frozen evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from beyond_entropy.predictability_matrix_artifacts import (
    atomic_json_write_exclusive,
    sha256_file,
)
from beyond_entropy.utility_dataset import load_utility_development


BENCHMARKS = ("chartqa", "docvqa", "hrbench")
ARMS = ("format_sft", "best_action_sft", "utility_sft")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def ranking_summary(
    true_gains: Sequence[Sequence[float]],
    predicted_gains: Sequence[Sequence[float]],
) -> dict[str, Any]:
    if not true_gains or len(true_gains) != len(predicted_gains):
        raise ValueError("ranking summary requires aligned nonempty rows")
    width = len(true_gains[0])
    if width <= 1 or any(
        len(true) != width or len(predicted) != width
        for true, predicted in zip(true_gains, predicted_gains, strict=True)
    ):
        raise ValueError("ranking rows must have a common action width")
    matrix = [[0 for _ in range(width)] for _ in range(width)]
    top1, regrets, concordant, comparable = 0, [], 0.0, 0
    for true, predicted in zip(true_gains, predicted_gains, strict=True):
        true_order = sorted(range(width), key=lambda index: (-true[index], index))
        predicted_order = sorted(
            range(width), key=lambda index: (-predicted[index], index)
        )
        true_rank = {action: rank for rank, action in enumerate(true_order)}
        predicted_rank = {action: rank for rank, action in enumerate(predicted_order)}
        for action in range(width):
            matrix[true_rank[action]][predicted_rank[action]] += 1
        selected = predicted_order[0]
        top1 += int(selected == true_order[0])
        regrets.append(max(true) - true[selected])
        for left in range(width):
            for right in range(left + 1, width):
                true_difference = true[left] - true[right]
                if true_difference == 0:
                    continue
                predicted_difference = predicted[left] - predicted[right]
                comparable += 1
                if predicted_difference == 0:
                    concordant += 0.5
                elif (true_difference > 0) == (predicted_difference > 0):
                    concordant += 1
    return {
        "states": len(true_gains),
        "actions": width,
        "true_rank_by_predicted_rank_counts": matrix,
        "top1_action_accuracy": top1 / len(true_gains),
        "mean_top1_regret": mean(regrets),
        "pairwise_ranking_accuracy": None if not comparable else concordant / comparable,
        "comparable_action_pairs": comparable,
        "tie_break": "descending score then ascending stable action index",
    }


def _load_inputs(bundle_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if (
        bundle.get("schema") != "utility_sft_validation_evaluation_bundle_v1"
        or bundle.get("role") != "validation"
        or bundle.get("test_data_present") is not False
    ):
        raise ValueError("invalid development evaluation bundle")
    loaded: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        entry = _mapping(bundle["inventory"][benchmark], benchmark)
        report_path = Path(str(entry["evaluation"])).resolve()
        if sha256_file(report_path) != entry["evaluation_sha256"]:
            raise ValueError(f"{benchmark} evaluation report changed")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("schema") != "utility_sft_evaluation_v1"
            or report.get("role") != "validation"
            or report.get("benchmark") != benchmark
        ):
            raise ValueError(f"{benchmark} evaluation schema mismatch")
        dataset_spec = _mapping(report["dataset"], "dataset")
        prediction_spec = _mapping(report["predictions"], "predictions")
        dataset_path = Path(str(dataset_spec["path"])).resolve()
        prediction_path = Path(str(prediction_spec["path"])).resolve()
        if (
            sha256_file(dataset_path) != dataset_spec["sha256"]
            or sha256_file(prediction_path) != prediction_spec["sha256"]
        ):
            raise ValueError(f"{benchmark} figure input changed")
        samples = load_utility_development(dataset_path, role="validation")
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        state_ids = [sample.inputs.state.state_id for sample in samples]
        true = [list(sample.gains) for sample in samples]
        arm_summaries = {}
        for arm in ARMS:
            by_state = prediction["arms"][arm]["predicted_gain"]
            if set(by_state) != set(state_ids):
                raise ValueError(f"{benchmark} {arm} figure coverage mismatch")
            arm_summaries[arm] = ranking_summary(
                true, [by_state[state_id] for state_id in state_ids]
            )
        summaries[benchmark] = arm_summaries
        loaded[benchmark] = {"report": report, "ranking": arm_summaries}
    return loaded, summaries


def render(*, evaluation_bundle: str | Path, output_root: str | Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    bundle_path = Path(evaluation_bundle).resolve()
    loaded, summaries = _load_inputs(bundle_path)
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite figures: {destination}")
    destination.mkdir(parents=True)

    fig, axes = plt.subplots(len(ARMS), len(BENCHMARKS), figsize=(12, 10))
    for row, arm in enumerate(ARMS):
        for column, benchmark in enumerate(BENCHMARKS):
            ax = axes[row, column]
            summary = summaries[benchmark][arm]
            values = np.asarray(summary["true_rank_by_predicted_rank_counts"], dtype=float)
            values /= values.sum(axis=1, keepdims=True)
            image = ax.imshow(values, vmin=0, vmax=1, cmap="Blues")
            for y in range(values.shape[0]):
                for x in range(values.shape[1]):
                    ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if values[y, x] > .55 else "black")
            ax.set_title(
                f"{benchmark} / {arm.replace('_', ' ')}\n"
                f"top1={summary['top1_action_accuracy']:.3f}, "
                f"regret={summary['mean_top1_regret']:.3f}",
                fontsize=9,
            )
            ax.set_xlabel("Predicted rank (0 = best)")
            ax.set_ylabel("True utility rank (0 = best)")
            ax.set_xticks(range(values.shape[1]))
            ax.set_yticks(range(values.shape[0]))
    fig.suptitle("Figure A: Action-level utility ranking", fontsize=14)
    fig.subplots_adjust(left=.07, right=.86, bottom=.07, top=.92, hspace=.36, wspace=.3)
    color_axis = fig.add_axes([.89, .15, .018, .68])
    fig.colorbar(image, cax=color_axis, label="Row-normalized frequency")
    figure_a_png = destination / "figure-a-action-ranking.png"
    figure_a_pdf = destination / "figure-a-action-ranking.pdf"
    fig.savefig(figure_a_png, dpi=220)
    fig.savefig(figure_a_pdf)
    plt.close(fig)

    labels = {
        "ug": "UG (4 crops)", "best_action_sft": "Best-Action SFT",
        "utility_sft": "Utility-SFT", "oracle": "Oracle",
    }
    colors = {"ug": "#777777", "best_action_sft": "#d95f02",
              "utility_sft": "#1b9e77", "oracle": "#7570b3"}
    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(13, 4.1))
    for ax, benchmark in zip(axes, BENCHMARKS, strict=True):
        report = loaded[benchmark]["report"]
        lambdas = [str(value) for value in report["lambdas"]]
        for policy in labels:
            points = []
            for value in lambdas:
                metrics = report["frontier"][value]["policies"][policy]["source_balanced"]
                point = (float(metrics["avg_visual_cost"]), float(metrics["accuracy"]), float(value))
                if not points or point[:2] != points[-1][:2]:
                    points.append(point)
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                    label=labels[policy], color=colors[policy], linewidth=1.8)
            if policy == "utility_sft":
                for cost, accuracy, lambda_value in points:
                    ax.annotate(f"{lambda_value:g}", (cost, accuracy), xytext=(3, 3),
                                textcoords="offset points", fontsize=7)
        ax.set_title(benchmark)
        ax.set_xlabel("Average visual cost")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=.25)
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Figure B: Accuracy–visual-cost frontier (labels show Utility-SFT λ)")
    fig.tight_layout()
    figure_b_png = destination / "figure-b-accuracy-cost-frontier.png"
    figure_b_pdf = destination / "figure-b-accuracy-cost-frontier.pdf"
    fig.savefig(figure_b_png, dpi=220)
    fig.savefig(figure_b_pdf)
    plt.close(fig)

    result = {
        "schema": "utility_sft_validation_figures_v1",
        "role": "validation",
        "formal_claim_eligible": False,
        "evaluation_bundle": str(bundle_path),
        "evaluation_bundle_sha256": sha256_file(bundle_path),
        "ranking_summaries": summaries,
        "figures": {
            "figure_a_png": {"path": str(figure_a_png), "sha256": sha256_file(figure_a_png)},
            "figure_a_pdf": {"path": str(figure_a_pdf), "sha256": sha256_file(figure_a_pdf)},
            "figure_b_png": {"path": str(figure_b_png), "sha256": sha256_file(figure_b_png)},
            "figure_b_pdf": {"path": str(figure_b_pdf), "sha256": sha256_file(figure_b_pdf)},
        },
    }
    atomic_json_write_exclusive(destination / "FIGURES.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-bundle", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = render(
        evaluation_bundle=args.evaluation_bundle, output_root=args.output_root
    )
    print(json.dumps({
        "output": str(Path(args.output_root).resolve()),
        "sha256": sha256_file(Path(args.output_root) / "FIGURES.json"),
        "figures": result["figures"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
