from __future__ import annotations

import json
import math
import random
from dataclasses import replace as dataclass_replace
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .dataset import group_by_decision, read_jsonl, split_by_group
from .metrics import evaluate_policy
from .model import LinearGainModel
from .policies import (
    AnswerNowPolicy,
    EntropyReductionThresholdPolicy,
    EntropySearchPolicy,
    EntropyThresholdPolicy,
    FixedCenterZoomPolicy,
    LearnedVOIPolicy,
    OracleVOIPolicy,
    Policy,
    PolicyDecision,
    RandomZoomPolicy,
    tune_entropy_thresholds,
)
from .qwen_semantic import (
    load_semantic_feature_dataset,
    validate_semantic_feature_dataset,
)
from .schema import ActionRecord
from .semantic import SemanticGainHead, require_torch


DecisionKey = tuple[str, str]


def _decision_key(value: Mapping[str, Any]) -> DecisionKey:
    return str(value["state_id"]), str(value["replicate_id"])


def fit_affine_gain_calibration(
    predictions: Sequence[float],
    targets: Sequence[float],
) -> tuple[float, float]:
    """Fit a monotone affine calibration on validation-only action labels."""

    if len(predictions) != len(targets) or not predictions:
        raise ValueError("calibration requires paired non-empty predictions and targets")
    prediction_mean = mean(predictions)
    target_mean = mean(targets)
    variance = sum((value - prediction_mean) ** 2 for value in predictions)
    if variance <= 1e-12:
        return 0.0, target_mean
    covariance = sum(
        (prediction - prediction_mean) * (target - target_mean)
        for prediction, target in zip(predictions, targets)
    )
    slope = max(0.0, covariance / variance)
    intercept = target_mean - slope * prediction_mean
    return slope, intercept


def _cosine(left: Any, right: Any) -> float:
    import torch.nn.functional as functional  # type: ignore[import-not-found]

    return float(functional.cosine_similarity(left.float(), right.float(), dim=0))


def add_semantic_similarity_features(
    records: Sequence[ActionRecord],
    decision_by_key: Mapping[DecisionKey, Mapping[str, Any]],
) -> list[ActionRecord]:
    """Add low-capacity frozen-space similarities without outcome leakage."""

    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as functional  # type: ignore[import-not-found]

    result: list[ActionRecord] = []
    for record in records:
        if record.action_type == "ANSWER":
            result.append(record)
            continue
        key = (record.state_id, record.replicate_id)
        decision = decision_by_key[key]
        action_ids = [str(value) for value in decision["action_ids"]]
        try:
            index = action_ids.index(record.action_id)
        except ValueError as exc:
            raise ValueError(
                f"semantic features are missing action {record.action_id!r} in {key!r}"
            ) from exc
        question = decision["question_embedding"].float()
        global_visual = decision["global_visual_embedding"].float()
        region = decision["region_embeddings"][index].float()
        question_unit = functional.normalize(question, dim=0)
        global_unit = functional.normalize(global_visual, dim=0)
        region_unit = functional.normalize(region, dim=0)
        semantic_features = {
            "semantic_q_region_cosine": _cosine(question, region),
            "semantic_q_global_cosine": _cosine(question, global_visual),
            "semantic_global_region_cosine": _cosine(global_visual, region),
            "semantic_q_region_contrast": float(
                torch.dot(question_unit, region_unit - global_unit)
            ),
            "semantic_region_global_l2": float(torch.linalg.vector_norm(region_unit - global_unit)),
            "semantic_question_norm": float(torch.linalg.vector_norm(question)),
            "semantic_global_norm": float(torch.linalg.vector_norm(global_visual)),
            "semantic_region_norm": float(torch.linalg.vector_norm(region)),
        }
        result.append(
            dataclass_replace(
                record,
                pre_action_features={
                    **record.pre_action_features,
                    **semantic_features,
                },
            )
        )
    return result


class PrecomputedGainPolicy:
    """Apply pre-action gain predictions to sibling rollout evaluation."""

    def __init__(
        self,
        predictions: Mapping[tuple[str, str, str], float],
        *,
        lambda_cost: float,
        name: str,
        decision_threshold: float = 0.0,
    ) -> None:
        if lambda_cost < 0.0:
            raise ValueError("lambda_cost must be non-negative")
        self.predictions = predictions
        self.lambda_cost = lambda_cost
        self.name = name
        self.decision_threshold = decision_threshold

    def select(self, siblings: Sequence[ActionRecord]) -> PolicyDecision:
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        if len(answers) != 1 or not zooms:
            raise ValueError("semantic policy requires one ANSWER and at least one ZOOM")
        scored: list[tuple[float, ActionRecord]] = []
        for record in zooms:
            key = (record.state_id, record.replicate_id, record.action_id)
            if key not in self.predictions:
                raise ValueError(f"missing semantic gain prediction for {key!r}")
            utility = self.predictions[key] - self.lambda_cost * record.tool_cost
            scored.append((utility, record))
        best_utility, selected = max(
            scored,
            key=lambda item: (item[0], item[1].action_id),
        )
        if best_utility <= self.decision_threshold:
            return PolicyDecision(answers[0], tool_calls=0, visual_cost=0.0)
        return PolicyDecision(selected, tool_calls=1, visual_cost=selected.tool_cost)


def tune_precomputed_gain_threshold(
    predictions: Mapping[tuple[str, str, str], float],
    records: Sequence[ActionRecord],
    *,
    lambda_cost: float,
) -> float:
    grouped = group_by_decision(records)
    best_utilities: list[float] = []
    for siblings in grouped.values():
        zooms = [record for record in siblings if record.action_type == "ZOOM"]
        best_utilities.append(
            max(
                predictions[(record.state_id, record.replicate_id, record.action_id)]
                - lambda_cost * record.tool_cost
                for record in zooms
            )
        )
    unique = sorted(set(best_utilities))
    thresholds = [unique[0] - 1e-9]
    thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    thresholds.append(unique[-1] + 1e-9)
    best_threshold = thresholds[0]
    best_score = (float("-inf"), float("-inf"))
    for threshold in thresholds:
        result = evaluate_policy(
            records,
            PrecomputedGainPolicy(
                predictions,
                lambda_cost=lambda_cost,
                name="threshold_tuning",
                decision_threshold=threshold,
            ),
            lambda_cost=lambda_cost,
        )
        mean_utility = result["mean_policy_utility"]
        tool_use_rate = result["tool_use_rate"]
        if not isinstance(mean_utility, (int, float)) or not isinstance(
            tool_use_rate, (int, float)
        ):
            raise RuntimeError("policy evaluation returned non-numeric tuning metrics")
        score = (
            float(mean_utility),
            -float(tool_use_rate),
        )
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def _keys(records: Sequence[ActionRecord]) -> set[DecisionKey]:
    return set(group_by_decision(records))


def grouped_kfold_records(
    records: Sequence[ActionRecord],
    *,
    group: str,
    n_folds: int,
    seed: int,
) -> list[tuple[list[ActionRecord], list[ActionRecord]]]:
    if group not in ("source_id", "image_id", "state_id"):
        raise ValueError(f"unsupported split group: {group}")
    group_ids = sorted({str(getattr(record, group)) for record in records})
    if n_folds < 2 or n_folds > len(group_ids):
        raise ValueError("n_folds must be between 2 and the number of groups")
    random.Random(seed).shuffle(group_ids)
    fold_ids = [set(group_ids[index::n_folds]) for index in range(n_folds)]
    result: list[tuple[list[ActionRecord], list[ActionRecord]]] = []
    for validation_ids in fold_ids:
        training = [
            record for record in records if str(getattr(record, group)) not in validation_ids
        ]
        validation = [
            record for record in records if str(getattr(record, group)) in validation_ids
        ]
        result.append((training, validation))
    return result


def cross_validated_linear_predictions(
    records: Sequence[ActionRecord],
    *,
    group: str,
    n_folds: int,
    seed: int,
) -> dict[tuple[str, str, str], float]:
    predictions: dict[tuple[str, str, str], float] = {}
    for training, validation in grouped_kfold_records(
        records,
        group=group,
        n_folds=n_folds,
        seed=seed,
    ):
        model = LinearGainModel.fit(training)
        for record in validation:
            if record.action_type != "ZOOM":
                continue
            key = (record.state_id, record.replicate_id, record.action_id)
            if key in predictions:
                raise RuntimeError(f"duplicate out-of-fold prediction for {key!r}")
            predictions[key] = model.predict_gain(record)
    expected = {
        (record.state_id, record.replicate_id, record.action_id)
        for record in records
        if record.action_type == "ZOOM"
    }
    if set(predictions) != expected:
        raise RuntimeError("out-of-fold predictions do not cover every ZOOM action")
    return predictions


def _select_decisions(
    decision_by_key: Mapping[DecisionKey, Mapping[str, Any]],
    keys: set[DecisionKey],
) -> list[Mapping[str, Any]]:
    missing = keys - set(decision_by_key)
    if missing:
        raise ValueError(f"semantic dataset is missing decisions: {sorted(missing)[:5]}")
    return [decision_by_key[key] for key in sorted(keys)]


def _state_signal_normalization(
    decisions: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any]:
    import torch  # type: ignore[import-not-found]

    signals = torch.stack([decision["state_signals"] for decision in decisions]).float()
    means = signals.mean(dim=0)
    scales = signals.std(dim=0, unbiased=False).clamp_min(1e-6)
    return means, scales


def _batch(
    decisions: Sequence[Mapping[str, Any]],
    *,
    signal_means: Any,
    signal_scales: Any,
    device: Any,
) -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as functional  # type: ignore[import-not-found]

    if not decisions:
        raise ValueError("semantic batch cannot be empty")
    candidate_counts = {int(decision["region_embeddings"].shape[0]) for decision in decisions}
    if len(candidate_counts) != 1:
        raise ValueError("all decisions in one semantic experiment need equal candidate counts")
    questions = torch.stack([decision["question_embedding"] for decision in decisions]).float()
    globals_ = torch.stack(
        [decision["global_visual_embedding"] for decision in decisions]
    ).float()
    regions = torch.stack([decision["region_embeddings"] for decision in decisions]).float()
    signals = torch.stack([decision["state_signals"] for decision in decisions]).float()
    return {
        "question_embedding": functional.normalize(questions, dim=-1).to(device),
        "global_visual_embedding": functional.normalize(globals_, dim=-1).to(device),
        "region_embeddings": functional.normalize(regions, dim=-1).to(device),
        "bboxes": torch.stack([decision["bboxes"] for decision in decisions]).float().to(
            device
        ),
        "state_signals": ((signals - signal_means) / signal_scales).to(device),
        "targets": torch.stack(
            [
                decision["success_after"].float() - float(decision["success_before"])
                for decision in decisions
            ]
        ).to(device),
    }


def _semantic_loss(
    predictions: Any,
    targets: Any,
    rank_weight: float,
    nonzero_weight: float,
) -> Any:
    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as functional  # type: ignore[import-not-found]

    point_weights = 1.0 + nonzero_weight * targets.abs()
    point_loss = ((predictions - targets).square() * point_weights).sum() / point_weights.sum()
    if rank_weight == 0.0:
        return point_loss
    predicted_differences = predictions[:, :, None] - predictions[:, None, :]
    target_differences = targets[:, :, None] - targets[:, None, :]
    informative = target_differences.abs() > 1e-12
    if not bool(torch.any(informative)):
        return point_loss
    rank_loss = functional.mse_loss(
        predicted_differences[informative],
        target_differences[informative],
    )
    return point_loss + rank_weight * rank_loss


def _predict(
    model: Any,
    decisions: Sequence[Mapping[str, Any]],
    *,
    signal_means: Any,
    signal_scales: Any,
    device: Any,
) -> tuple[Any, Any]:
    import torch  # type: ignore[import-not-found]

    batch = _batch(
        decisions,
        signal_means=signal_means,
        signal_scales=signal_scales,
        device=device,
    )
    model.eval()
    with torch.inference_mode():
        predictions = model(
            question_embedding=batch["question_embedding"],
            global_visual_embedding=batch["global_visual_embedding"],
            region_embeddings=batch["region_embeddings"],
            bboxes=batch["bboxes"],
            state_signals=batch["state_signals"],
        )
    return predictions.detach().cpu(), batch["targets"].detach().cpu()


def _prediction_map(
    decisions: Sequence[Mapping[str, Any]],
    predictions: Any,
    *,
    slope: float = 1.0,
    intercept: float = 0.0,
) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    for decision, row in zip(decisions, predictions):
        state_id, replicate_id = _decision_key(decision)
        for action_id, value in zip(decision["action_ids"], row):
            result[(state_id, replicate_id, str(action_id))] = (
                slope * float(value) + intercept
            )
    return result


def _atomic_torch_save(value: object, destination: Path) -> None:
    import torch  # type: ignore[import-not-found]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def _write_json(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_semantic_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen semantic gain pilot",
        "",
        "> Diagnostic pilot only; test groups are held out by image/source.",
        "",
        "| Lambda | Policy | Accuracy | Gain | Tool use | Mean utility |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for sweep in report["lambda_sweep"]:
        for result in sweep["policy_results"]:
            lines.append(
                "| {lambda_cost:.3f} | {policy} | {accuracy:.4f} | "
                "{accuracy_gain:.4f} | {tool_use_rate:.4f} | {mean_policy_utility:.4f} |".format(
                    lambda_cost=sweep["lambda_cost"],
                    **result,
                )
            )
    lines.extend(
        (
            "",
            "## Guardrails",
            "",
            "- The semantic head never consumes crop outcomes or ground truth as inputs.",
            "- Early stopping and affine calibration use only the inner validation split.",
            "- The outer test split is never used for fitting, stopping, or calibration.",
            "- Oracle VOI consumes labels and is not deployable.",
            "",
        )
    )
    return "\n".join(lines)


def fit_semantic_gain_experiment(
    *,
    feature_path: str | Path,
    rollouts_path: str | Path,
    output_dir: str | Path,
    split_group: str = "image_id",
    train_fraction: float = 0.7,
    validation_fraction: float = 0.2,
    lambdas: Sequence[float] = (0.0, 0.01, 0.05, 0.1, 0.2),
    hidden_dim: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    rank_weight: float = 1.0,
    nonzero_weight: float = 8.0,
    similarity_cv_folds: int = 5,
    max_epochs: int = 500,
    patience: int = 50,
    seed: int = 17,
    device_name: str = "cuda",
) -> dict[str, Any]:
    require_torch()
    import torch  # type: ignore[import-not-found]

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if hidden_dim <= 0 or max_epochs <= 0 or patience <= 0:
        raise ValueError("hidden_dim, max_epochs, and patience must be positive")
    if (
        learning_rate <= 0.0
        or weight_decay < 0.0
        or rank_weight < 0.0
        or nonzero_weight < 0.0
    ):
        raise ValueError("optimizer and rank-loss settings must be non-negative")
    if not lambdas or any(value < 0.0 for value in lambdas):
        raise ValueError("lambdas must be non-empty and non-negative")
    records = read_jsonl(rollouts_path)
    feature_dataset = load_semantic_feature_dataset(feature_path)
    validate_semantic_feature_dataset(feature_dataset, records)
    decision_by_key = {
        _decision_key(decision): decision for decision in feature_dataset["decisions"]
    }
    outer_train_records, test_records = split_by_group(
        records,
        group=split_group,  # type: ignore[arg-type]
        train_fraction=train_fraction,
        seed=seed,
    )
    inner_fraction = 1.0 - validation_fraction
    model_train_records, validation_records = split_by_group(
        outer_train_records,
        group=split_group,  # type: ignore[arg-type]
        train_fraction=inner_fraction,
        seed=seed + 1,
    )
    train_decisions = _select_decisions(decision_by_key, _keys(model_train_records))
    validation_decisions = _select_decisions(decision_by_key, _keys(validation_records))
    test_decisions = _select_decisions(decision_by_key, _keys(test_records))
    signal_means, signal_scales = _state_signal_normalization(train_decisions)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for semantic training but is unavailable")
    device = torch.device(device_name)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    exemplar = train_decisions[0]
    model = SemanticGainHead(
        question_dim=int(exemplar["question_embedding"].shape[-1]),
        visual_dim=int(exemplar["global_visual_embedding"].shape[-1]),
        state_signal_dim=int(exemplar["state_signals"].shape[-1]),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_batch = _batch(
        train_decisions,
        signal_means=signal_means,
        signal_scales=signal_scales,
        device=device,
    )
    validation_batch = _batch(
        validation_decisions,
        signal_means=signal_means,
        signal_scales=signal_scales,
        device=device,
    )
    best_validation_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_predictions = model(
            question_embedding=train_batch["question_embedding"],
            global_visual_embedding=train_batch["global_visual_embedding"],
            region_embeddings=train_batch["region_embeddings"],
            bboxes=train_batch["bboxes"],
            state_signals=train_batch["state_signals"],
        )
        train_loss = _semantic_loss(
            train_predictions,
            train_batch["targets"],
            rank_weight,
            nonzero_weight,
        )
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_predictions = model(
                question_embedding=validation_batch["question_embedding"],
                global_visual_embedding=validation_batch["global_visual_embedding"],
                region_embeddings=validation_batch["region_embeddings"],
                bboxes=validation_batch["bboxes"],
                state_signals=validation_batch["state_signals"],
            )
            validation_loss = _semantic_loss(
                validation_predictions,
                validation_batch["targets"],
                rank_weight,
                nonzero_weight,
            )
        train_value = float(train_loss.detach().cpu())
        validation_value = float(validation_loss.detach().cpu())
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_value,
                "validation_loss": validation_value,
            }
        )
        if validation_value < best_validation_loss - 1e-7:
            best_validation_loss = validation_value
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("semantic training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_predictions, validation_targets = _predict(
        model,
        validation_decisions,
        signal_means=signal_means,
        signal_scales=signal_scales,
        device=device,
    )
    calibration_slope, calibration_intercept = fit_affine_gain_calibration(
        validation_predictions.reshape(-1).tolist(),
        validation_targets.reshape(-1).tolist(),
    )
    test_predictions, test_targets = _predict(
        model,
        test_decisions,
        signal_means=signal_means,
        signal_scales=signal_scales,
        device=device,
    )
    raw_prediction_map = _prediction_map(test_decisions, test_predictions)
    calibrated_prediction_map = _prediction_map(
        test_decisions,
        test_predictions,
        slope=calibration_slope,
        intercept=calibration_intercept,
    )
    validation_raw_prediction_map = _prediction_map(
        validation_decisions,
        validation_predictions,
    )
    semantic_train_records = add_semantic_similarity_features(
        outer_train_records,
        decision_by_key,
    )
    semantic_test_records = add_semantic_similarity_features(
        test_records,
        decision_by_key,
    )
    similarity_model = LinearGainModel.fit(semantic_train_records)
    validation_similarity_predictions = cross_validated_linear_predictions(
        semantic_train_records,
        group=split_group,
        n_folds=similarity_cv_folds,
        seed=seed + 2,
    )
    test_similarity_predictions = {
        (record.state_id, record.replicate_id, record.action_id): similarity_model.predict_gain(
            record
        )
        for record in semantic_test_records
        if record.action_type == "ZOOM"
    }
    scalar_model = LinearGainModel.fit(model_train_records)
    lambda_sweep: list[dict[str, Any]] = []
    for lambda_cost in lambdas:
        entropy_threshold, entropy_reduction_threshold = tune_entropy_thresholds(
            validation_records,
            lambda_cost=lambda_cost,
        )
        semantic_threshold = tune_precomputed_gain_threshold(
            validation_raw_prediction_map,
            validation_records,
            lambda_cost=lambda_cost,
        )
        similarity_threshold = tune_precomputed_gain_threshold(
            validation_similarity_predictions,
            semantic_train_records,
            lambda_cost=lambda_cost,
        )
        policies: list[Policy] = [
            AnswerNowPolicy(),
            RandomZoomPolicy(seed=seed),
            FixedCenterZoomPolicy(),
            EntropySearchPolicy(),
            EntropyThresholdPolicy(entropy_threshold),
            EntropyReductionThresholdPolicy(entropy_reduction_threshold),
            LearnedVOIPolicy(scalar_model, lambda_cost=lambda_cost),
            PrecomputedGainPolicy(
                raw_prediction_map,
                lambda_cost=lambda_cost,
                name="semantic_gain_raw",
            ),
            PrecomputedGainPolicy(
                calibrated_prediction_map,
                lambda_cost=lambda_cost,
                name="semantic_gain_calibrated",
            ),
            PrecomputedGainPolicy(
                raw_prediction_map,
                lambda_cost=lambda_cost,
                name="semantic_gain_val_threshold",
                decision_threshold=semantic_threshold,
            ),
            PrecomputedGainPolicy(
                test_similarity_predictions,
                lambda_cost=lambda_cost,
                name="semantic_similarity_ridge",
                decision_threshold=similarity_threshold,
            ),
            OracleVOIPolicy(lambda_cost),
        ]
        lambda_sweep.append(
            {
                "lambda_cost": lambda_cost,
                "semantic_gain_validation_threshold": semantic_threshold,
                "semantic_similarity_validation_threshold": similarity_threshold,
                "policy_results": [
                    evaluate_policy(test_records, policy, lambda_cost=lambda_cost)
                    for policy in policies
                ],
            }
        )
    output = Path(output_dir)
    model_path = output / "semantic_gain_model.pt"
    similarity_model_path = output / "semantic_similarity_gain_model.json"
    report_path = output / "report.json"
    markdown_path = output / "report.md"
    checkpoint = {
        "format_version": 1,
        "model_type": "qwen_roi_semantic_success_gain",
        "state_dict": best_state,
        "model_config": {
            "question_dim": int(exemplar["question_embedding"].shape[-1]),
            "visual_dim": int(exemplar["global_visual_embedding"].shape[-1]),
            "state_signal_dim": int(exemplar["state_signals"].shape[-1]),
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        },
        "signal_means": signal_means,
        "signal_scales": signal_scales,
        "calibration": {
            "slope": calibration_slope,
            "intercept": calibration_intercept,
        },
        "feature_metadata": feature_dataset["metadata"],
    }
    report: dict[str, Any] = {
        "scientific_status": "diagnostic; not a benchmark claim",
        "run": {
            "features": str(Path(feature_path).resolve()),
            "rollouts": str(Path(rollouts_path).resolve()),
            "split_group": split_group,
            "train_fraction": train_fraction,
            "validation_fraction_within_train": validation_fraction,
            "seed": seed,
            "outer_train_decisions": len(_keys(outer_train_records)),
            "model_train_decisions": len(train_decisions),
            "validation_decisions": len(validation_decisions),
            "test_decisions": len(test_decisions),
            "candidate_count": int(test_predictions.shape[1]),
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "rank_weight": rank_weight,
            "nonzero_weight": nonzero_weight,
            "similarity_cv_folds": similarity_cv_folds,
            "max_epochs": max_epochs,
            "patience": patience,
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "best_validation_loss": best_validation_loss,
            "test_action_mse_raw": float(
                torch.nn.functional.mse_loss(test_predictions, test_targets)
            ),
            "calibration": {
                "slope": calibration_slope,
                "intercept": calibration_intercept,
            },
        },
        "training_history": history,
        "lambda_sweep": lambda_sweep,
    }
    _atomic_torch_save(checkpoint, model_path)
    similarity_model.save(similarity_model_path)
    _write_json(report, report_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_semantic_markdown_report(report), encoding="utf-8")
    return report
