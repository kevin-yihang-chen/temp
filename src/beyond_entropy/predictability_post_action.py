from __future__ import annotations

import math
import inspect
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .predictability_audit import BinaryToolOutcome
from .predictability_baselines import FROZEN_UG_ACTION_IDS
from .predictability_evaluation import (
    Prediction,
    align_predictions,
    calls_at_threshold,
    policy_metrics,
    select_validation_threshold,
)
from .predictability_modeling import (
    ScoreCalibrators,
    fit_score_calibrators,
    source_balanced_weights,
)
from .schema import BBox


POST_ACTION_PROBE_HIDDEN_LAYERS = (128, 32)
POST_ACTION_PROBE_ALPHA = 0.0001
POST_ACTION_PROBE_MAX_ITERATIONS = 500
_POST_ACTION_FIELDS = frozenset(
    {
        "selected_action_id",
        "candidate_action_ids",
        "baseline_confidence",
        "candidate_post_action_confidence",
        "selected_bbox",
        "selected_action_one_hot",
        "pooled_language_state",
        "pooled_visual_state",
        "fused_multimodal_state",
    }
)


def _finite_vector(value: Any, *, name: str) -> tuple[float, ...]:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a numeric sequence")
    result = tuple(float(item) for item in value)
    if not result or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be non-empty and finite")
    return result


@dataclass(frozen=True)
class PostActionProbeInputs:
    """Typed view reserved for the explicitly non-deployable oracle probe."""

    state_id: str
    image_id: str
    source_id: str
    selected_action_id: str
    baseline_confidence: tuple[float, ...]
    candidate_post_action_confidence: tuple[float, ...]
    selected_bbox: tuple[float, ...]
    selected_action_one_hot: tuple[float, ...]
    pooled_language_state: tuple[float, ...]
    pooled_visual_state: tuple[float, ...]
    fused_multimodal_state: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.state_id or not self.image_id or not self.source_id:
            raise ValueError("post-action probe identities must be non-empty")
        if self.selected_action_id not in FROZEN_UG_ACTION_IDS:
            raise ValueError("post-action probe selected action is not frozen")
        if len(self.baseline_confidence) != 3:
            raise ValueError(
                "baseline confidence must contain entropy, max, and margin"
            )
        if len(self.candidate_post_action_confidence) != 12:
            raise ValueError("candidate confidence trace must contain four triples")
        for offset in range(0, 12, 3):
            entropy, maximum, margin = self.candidate_post_action_confidence[
                offset : offset + 3
            ]
            if entropy < 0.0 or not 0.0 <= maximum <= 1.0 or not 0.0 <= margin <= 1.0:
                raise ValueError("candidate post-action confidence values are invalid")
        entropy, maximum, margin = self.baseline_confidence
        if entropy < 0.0 or not 0.0 <= maximum <= 1.0 or not 0.0 <= margin <= 1.0:
            raise ValueError("baseline confidence values are invalid")
        if len(self.selected_bbox) != 4:
            raise ValueError("selected crop bbox must have four coordinates")
        BBox(*self.selected_bbox)
        if len(self.selected_action_one_hot) != 4 or set(
            self.selected_action_one_hot
        ) - {0.0, 1.0}:
            raise ValueError("selected action one-hot must contain four binary values")
        selected_index = FROZEN_UG_ACTION_IDS.index(self.selected_action_id)
        expected = tuple(float(index == selected_index) for index in range(4))
        if self.selected_action_one_hot != expected:
            raise ValueError(
                "selected action one-hot disagrees with selected action ID"
            )
        state_dimensions = {
            len(self.pooled_language_state),
            len(self.pooled_visual_state),
            len(self.fused_multimodal_state),
        }
        if len(state_dimensions) != 1:
            raise ValueError("post-action Qwen state dimensions must match")
        values = self.feature_vector()
        if not all(math.isfinite(item) for item in values):
            raise ValueError("post-action probe features must be finite")

    @classmethod
    def from_untrusted_mapping(
        cls, value: Mapping[str, Any]
    ) -> "PostActionProbeInputs":
        raw = value.get("post_action_probe")
        if not isinstance(raw, Mapping):
            raise ValueError("row must contain a post_action_probe mapping")
        unknown = set(raw) - _POST_ACTION_FIELDS
        if unknown:
            raise ValueError(f"unknown post_action_probe fields: {sorted(unknown)}")
        missing = _POST_ACTION_FIELDS - set(raw)
        if missing:
            raise ValueError(f"missing post_action_probe fields: {sorted(missing)}")
        action_ids = tuple(str(item) for item in raw["candidate_action_ids"])
        if action_ids != FROZEN_UG_ACTION_IDS:
            raise ValueError("post-action probe candidate action bank is not frozen")
        return cls(
            state_id=str(value["state_id"]),
            image_id=str(value["image_id"]),
            source_id=str(value["source_id"]),
            selected_action_id=str(raw["selected_action_id"]),
            baseline_confidence=_finite_vector(
                raw["baseline_confidence"], name="baseline_confidence"
            ),
            candidate_post_action_confidence=_finite_vector(
                raw["candidate_post_action_confidence"],
                name="candidate_post_action_confidence",
            ),
            selected_bbox=_finite_vector(raw["selected_bbox"], name="selected_bbox"),
            selected_action_one_hot=_finite_vector(
                raw["selected_action_one_hot"], name="selected_action_one_hot"
            ),
            pooled_language_state=_finite_vector(
                raw["pooled_language_state"], name="pooled_language_state"
            ),
            pooled_visual_state=_finite_vector(
                raw["pooled_visual_state"], name="pooled_visual_state"
            ),
            fused_multimodal_state=_finite_vector(
                raw["fused_multimodal_state"], name="fused_multimodal_state"
            ),
        )

    def feature_vector(self) -> tuple[float, ...]:
        return (
            self.baseline_confidence
            + self.candidate_post_action_confidence
            + self.selected_bbox
            + self.selected_action_one_hot
            + self.pooled_language_state
            + self.pooled_visual_state
            + self.fused_multimodal_state
        )


@dataclass(frozen=True)
class PostActionProbeExample:
    inputs: PostActionProbeInputs
    outcome: BinaryToolOutcome
    image_rgb_sha256: str

    def __post_init__(self) -> None:
        if (
            self.inputs.state_id,
            self.inputs.image_id,
            self.inputs.source_id,
            self.inputs.selected_action_id,
        ) != (
            self.outcome.state_id,
            self.outcome.image_id,
            self.outcome.source_id,
            self.outcome.selected_action_id,
        ):
            raise ValueError("post-action inputs and outcome identities differ")
        if len(self.image_rgb_sha256) != 64:
            raise ValueError("post-action example requires a decoded-RGB SHA-256")
        try:
            int(self.image_rgb_sha256, 16)
        except ValueError as exc:
            raise ValueError("image_rgb_sha256 is not hexadecimal") from exc


@dataclass
class FrozenPostActionProbe:
    seed: int
    estimator: Any
    calibrators: ScoreCalibrators
    threshold: float
    validation_metrics: dict[str, float | int]
    input_dimension: int

    def predict(self, examples: Sequence[PostActionProbeExample]) -> list[Prediction]:
        import numpy as np  # type: ignore[import-not-found]

        rows = np.asarray(
            [item.inputs.feature_vector() for item in examples], dtype=np.float64
        )
        scores = [float(item) for item in self.estimator.predict(rows)]
        return self.calibrators.predictions(examples, scores)


def fit_frozen_post_action_probe(
    train: Sequence[PostActionProbeExample],
    validation: Sequence[PostActionProbeExample],
    *,
    seed: int,
    lambda_cost: float,
) -> FrozenPostActionProbe:
    """Fit the one registered non-deployable two-layer direct-gain probe."""

    if not train or not validation:
        raise ValueError("post-action probe requires non-empty train and validation")
    import numpy as np  # type: ignore[import-not-found]
    from sklearn.neural_network import MLPRegressor  # type: ignore[import-untyped]
    from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    rows = np.asarray(
        [item.inputs.feature_vector() for item in train], dtype=np.float64
    )
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError("post-action training features must form a finite matrix")
    estimator = MLPRegressor(
        hidden_layer_sizes=POST_ACTION_PROBE_HIDDEN_LAYERS,
        activation="relu",
        alpha=POST_ACTION_PROBE_ALPHA,
        max_iter=POST_ACTION_PROBE_MAX_ITERATIONS,
        random_state=seed,
    )
    pipeline = make_pipeline(StandardScaler(), estimator)
    targets = np.asarray([item.outcome.gain for item in train])
    weights = source_balanced_weights(train)
    if "sample_weight" in inspect.signature(estimator.fit).parameters:
        pipeline.fit(rows, targets, mlpregressor__sample_weight=weights)
    elif np.allclose(weights, weights[0], atol=0.0, rtol=0.0):
        # Equal source weights reduce exactly to the unweighted objective. This
        # keeps dependency-light synthetic tests runnable on sklearn < 1.7;
        # nonuniform formal data must use the pinned >=1.7 environment.
        pipeline.fit(rows, targets)
    else:
        raise RuntimeError(
            "nonuniform post-action source weights require scikit-learn>=1.7"
        )
    validation_rows = np.asarray(
        [item.inputs.feature_vector() for item in validation], dtype=np.float64
    )
    validation_scores = [float(item) for item in pipeline.predict(validation_rows)]
    calibrators = fit_score_calibrators(
        validation,
        validation_scores,
        lambda_cost=lambda_cost,
        seed=seed + 3,
    )
    predictions = calibrators.predictions(validation, validation_scores)
    threshold = select_validation_threshold(
        [item.outcome for item in validation],
        predictions,
        lambda_cost=lambda_cost,
    )
    return FrozenPostActionProbe(
        seed=seed,
        estimator=pipeline,
        calibrators=calibrators,
        threshold=float(threshold["threshold"]),
        validation_metrics=threshold,
        input_dimension=int(rows.shape[1]),
    )


def evaluate_frozen_post_action_probe(
    probe: FrozenPostActionProbe,
    test: Sequence[PostActionProbeExample],
    *,
    lambda_cost: float,
) -> tuple[list[Prediction], dict[str, float | int | None]]:
    if not test:
        raise ValueError("post-action probe evaluation requires non-empty test")
    predictions = probe.predict(test)
    outcomes = [item.outcome for item in test]
    aligned = align_predictions(outcomes, predictions)
    metrics = policy_metrics(
        outcomes,
        calls_at_threshold(aligned, probe.threshold),
        lambda_cost=lambda_cost,
    )
    return predictions, metrics
