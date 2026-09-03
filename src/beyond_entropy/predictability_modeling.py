from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .predictability_audit import (
    PREDICTOR_LEVELS,
    TARGET_FAMILIES,
    BinaryToolOutcome,
    PreActionInputs,
)
from .predictability_evaluation import (
    Prediction,
    align_predictions,
    calls_at_threshold,
    policy_metrics,
    select_validation_threshold,
)


class LabeledOutcomeExample(Protocol):
    @property
    def outcome(self) -> BinaryToolOutcome: ...


@dataclass(frozen=True)
class AuditExample:
    inputs: PreActionInputs
    outcome: BinaryToolOutcome
    image_rgb_sha256: str

    def __post_init__(self) -> None:
        if (
            self.inputs.state_id,
            self.inputs.image_id,
            self.inputs.source_id,
        ) != (
            self.outcome.state_id,
            self.outcome.image_id,
            self.outcome.source_id,
        ):
            raise ValueError("pre-action inputs and outcome identities differ")
        if len(self.image_rgb_sha256) != 64:
            raise ValueError("audit example requires a decoded-RGB SHA-256")
        try:
            int(self.image_rgb_sha256, 16)
        except ValueError as exc:
            raise ValueError("image_rgb_sha256 is not hexadecimal") from exc


@dataclass(frozen=True)
class ConstantBinaryEstimator:
    probability: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("constant probability must be in [0, 1]")

    def predict_positive_probability(self, rows: Any) -> Any:
        import numpy as np  # type: ignore[import-not-found]

        return np.full(len(rows), self.probability, dtype=np.float64)


def registered_model_variants(level: str) -> tuple[str, ...]:
    if level == "l0_uncertainty":
        return ("entropy", "max_probability", "top1_top2_margin")
    if level == "l1_shallow":
        return ("linear", "small_mlp")
    if level == "l2_semantic":
        return ("small_mlp",)
    if level == "l3_frozen_qwen":
        return ("linear", "two_layer_mlp")
    raise ValueError(f"unsupported predictor level: {level}")


def _vector(inputs: PreActionInputs, *, level: str, variant: str) -> tuple[float, ...]:
    values = inputs.feature_vector(level)
    if level != "l0_uncertainty":
        return values
    index = registered_model_variants(level).index(variant)
    return (values[index],)


def source_balanced_weights(examples: Sequence[LabeledOutcomeExample]) -> Any:
    import numpy as np  # type: ignore[import-not-found]

    counts = Counter(item.outcome.source_id for item in examples)
    raw = np.asarray([1.0 / counts[item.outcome.source_id] for item in examples])
    return raw * len(raw) / raw.sum()


def _fit_regressor(
    rows: Any,
    targets: Any,
    weights: Any,
    *,
    family: str,
    level: str,
    seed: int,
) -> Any:
    from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
    from sklearn.neural_network import MLPRegressor  # type: ignore[import-untyped]
    from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    if family == "linear":
        estimator = Ridge(alpha=1.0)
    else:
        hidden = {
            "l1_shallow": (32,),
            "l2_semantic": (64,),
            "l3_frozen_qwen": (128, 32),
        }[level]
        estimator = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            alpha=0.0001,
            max_iter=500,
            random_state=seed,
        )
    pipeline = make_pipeline(StandardScaler(), estimator)
    pipeline.fit(
        rows,
        targets,
        **{f"{estimator.__class__.__name__.lower()}__sample_weight": weights},
    )
    return pipeline


def _fit_binary(
    rows: Any,
    labels: Any,
    weights: Any,
    *,
    family: str,
    level: str,
    seed: int,
) -> Any:
    import numpy as np  # type: ignore[import-not-found]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.neural_network import MLPClassifier  # type: ignore[import-untyped]
    from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0:
        return ConstantBinaryEstimator(0.0)
    if len(set(labels.tolist())) == 1:
        return ConstantBinaryEstimator(float(labels[0]))
    if family == "linear":
        estimator = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    else:
        hidden = {
            "l1_shallow": (32,),
            "l2_semantic": (64,),
            "l3_frozen_qwen": (128, 32),
        }[level]
        estimator = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            alpha=0.0001,
            max_iter=500,
            random_state=seed,
        )
    pipeline = make_pipeline(StandardScaler(), estimator)
    pipeline.fit(
        rows,
        labels,
        **{f"{estimator.__class__.__name__.lower()}__sample_weight": weights},
    )
    return pipeline


def _positive_probability(estimator: Any, rows: Any) -> Any:
    if isinstance(estimator, ConstantBinaryEstimator):
        return estimator.predict_positive_probability(rows)
    return estimator.predict_proba(rows)[:, 1]


@dataclass
class RawTargetModel:
    level: str
    target: str
    variant: str
    estimators: dict[str, Any]

    def scores(self, inputs: Sequence[PreActionInputs]) -> list[float]:
        import numpy as np  # type: ignore[import-not-found]

        rows = np.asarray(
            [_vector(item, level=self.level, variant=self.variant) for item in inputs],
            dtype=np.float64,
        )
        if self.target == "direct_gain":
            return [float(item) for item in self.estimators["gain"].predict(rows)]
        if self.target == "rescue_harm":
            rescue = _positive_probability(self.estimators["rescue"], rows)
            harm = _positive_probability(self.estimators["harm"], rows)
            return [float(left - right) for left, right in zip(rescue, harm)]
        error = _positive_probability(self.estimators["error"], rows)
        rescue = _positive_probability(self.estimators["rescue_given_error"], rows)
        harm = _positive_probability(self.estimators["harm_given_correct"], rows)
        return [
            float(p_error * p_rescue - (1.0 - p_error) * p_harm)
            for p_error, p_rescue, p_harm in zip(error, rescue, harm)
        ]


def fit_raw_target_model(
    examples: Sequence[AuditExample],
    *,
    level: str,
    target: str,
    variant: str,
    seed: int,
) -> RawTargetModel:
    if not examples:
        raise ValueError("model fitting requires training examples")
    if level not in PREDICTOR_LEVELS or target not in TARGET_FAMILIES:
        raise ValueError("unsupported predictor level or target")
    if variant not in registered_model_variants(level):
        raise ValueError("model variant is not registered for predictor level")
    import numpy as np  # type: ignore[import-not-found]

    rows = np.asarray(
        [_vector(item.inputs, level=level, variant=variant) for item in examples],
        dtype=np.float64,
    )
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError("training features must form a finite two-dimensional matrix")
    weights = source_balanced_weights(examples)
    family = "linear" if level == "l0_uncertainty" or variant == "linear" else "mlp"
    if target == "direct_gain":
        estimators = {
            "gain": _fit_regressor(
                rows,
                np.asarray([item.outcome.gain for item in examples]),
                weights,
                family=family,
                level=level,
                seed=seed,
            )
        }
    elif target == "rescue_harm":
        estimators = {
            "rescue": _fit_binary(
                rows,
                [item.outcome.rescue for item in examples],
                weights,
                family=family,
                level=level,
                seed=seed,
            ),
            "harm": _fit_binary(
                rows,
                [item.outcome.harm for item in examples],
                weights,
                family=family,
                level=level,
                seed=seed + 1,
            ),
        }
    else:
        errors = np.asarray([item.outcome.y0 < 1.0 for item in examples])
        correct = ~errors
        estimators = {
            "error": _fit_binary(
                rows,
                errors,
                weights,
                family=family,
                level=level,
                seed=seed,
            ),
            "rescue_given_error": _fit_binary(
                rows[errors],
                [
                    item.outcome.rescue
                    for item, selected in zip(examples, errors)
                    if selected
                ],
                weights[errors],
                family=family,
                level=level,
                seed=seed + 1,
            ),
            "harm_given_correct": _fit_binary(
                rows[correct],
                [
                    item.outcome.harm
                    for item, selected in zip(examples, correct)
                    if selected
                ],
                weights[correct],
                family=family,
                level=level,
                seed=seed + 2,
            ),
        }
    return RawTargetModel(
        level=level, target=target, variant=variant, estimators=estimators
    )


@dataclass
class ScoreCalibrators:
    positive_net: Any
    rescue: Any
    harm: Any

    def predictions(
        self,
        examples: Sequence[LabeledOutcomeExample],
        scores: Sequence[float],
    ) -> list[Prediction]:
        import numpy as np  # type: ignore[import-not-found]

        rows = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        positive = _positive_probability(self.positive_net, rows)
        rescue = _positive_probability(self.rescue, rows)
        harm = _positive_probability(self.harm, rows)
        return [
            Prediction(
                state_id=example.outcome.state_id,
                replicate_id=example.outcome.replicate_id,
                score=float(score),
                positive_net_probability=float(p_positive),
                rescue_probability=float(p_rescue),
                harm_probability=float(p_harm),
            )
            for example, score, p_positive, p_rescue, p_harm in zip(
                examples, scores, positive, rescue, harm
            )
        ]


def fit_score_calibrators(
    examples: Sequence[LabeledOutcomeExample],
    scores: Sequence[float],
    *,
    lambda_cost: float,
    seed: int,
) -> ScoreCalibrators:
    import numpy as np  # type: ignore[import-not-found]

    if len(examples) != len(scores) or not examples:
        raise ValueError("calibration requires aligned non-empty examples and scores")
    rows = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    weights = source_balanced_weights(examples)
    labels = {
        "positive_net": [
            item.outcome.incremental_utility(lambda_cost) > 0.0 for item in examples
        ],
        "rescue": [item.outcome.rescue for item in examples],
        "harm": [item.outcome.harm for item in examples],
    }
    fitted = {
        name: _fit_binary(
            rows,
            values,
            weights,
            family="linear",
            level="l1_shallow",
            seed=seed + index,
        )
        for index, (name, values) in enumerate(labels.items())
    }
    return ScoreCalibrators(**fitted)


@dataclass
class FrozenAuditCell:
    level: str
    target: str
    seed: int
    variant: str
    raw_model: RawTargetModel
    calibrators: ScoreCalibrators
    threshold: float
    validation_metrics: dict[str, float | int]

    def predict(self, examples: Sequence[AuditExample]) -> list[Prediction]:
        scores = self.raw_model.scores([item.inputs for item in examples])
        return self.calibrators.predictions(examples, scores)


def fit_frozen_audit_cell(
    train: Sequence[AuditExample],
    validation: Sequence[AuditExample],
    *,
    level: str,
    target: str,
    seed: int,
    lambda_cost: float,
) -> FrozenAuditCell:
    if not train or not validation:
        raise ValueError("cell fitting requires non-empty train and validation roles")
    candidate_cells: list[FrozenAuditCell] = []
    for variant_index, variant in enumerate(registered_model_variants(level)):
        raw_model = fit_raw_target_model(
            train,
            level=level,
            target=target,
            variant=variant,
            seed=seed + variant_index * 10,
        )
        validation_scores = raw_model.scores([item.inputs for item in validation])
        calibrators = fit_score_calibrators(
            validation,
            validation_scores,
            lambda_cost=lambda_cost,
            seed=seed + variant_index * 10 + 3,
        )
        predictions = calibrators.predictions(validation, validation_scores)
        threshold = select_validation_threshold(
            [item.outcome for item in validation], predictions, lambda_cost=lambda_cost
        )
        candidate_cells.append(
            FrozenAuditCell(
                level=level,
                target=target,
                seed=seed,
                variant=variant,
                raw_model=raw_model,
                calibrators=calibrators,
                threshold=float(threshold["threshold"]),
                validation_metrics=threshold,
            )
        )
    return max(
        candidate_cells,
        key=lambda item: (
            float(item.validation_metrics["validation_utility"]),
            -int(item.validation_metrics["validation_calls"]),
            -registered_model_variants(level).index(item.variant),
        ),
    )


def evaluate_frozen_audit_cell(
    cell: FrozenAuditCell,
    test: Sequence[AuditExample],
    *,
    lambda_cost: float,
) -> tuple[list[Prediction], dict[str, float | int | None]]:
    if not test:
        raise ValueError("frozen evaluation requires non-empty test examples")
    predictions = cell.predict(test)
    aligned = align_predictions([item.outcome for item in test], predictions)
    metrics = policy_metrics(
        [item.outcome for item in test],
        calls_at_threshold(aligned, cell.threshold),
        lambda_cost=lambda_cost,
    )
    return predictions, metrics
