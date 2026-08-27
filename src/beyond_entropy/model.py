from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .features import FeatureEncoder, select_zooms
from .schema import ActionRecord


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve Ax=b with partial-pivoting Gaussian elimination."""

    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("ridge system is singular; increase alpha")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for item in range(column, size + 1):
            augmented[column][item] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for item in range(column, size + 1):
                augmented[row][item] -= factor * augmented[column][item]
    return [augmented[row][-1] for row in range(size)]


@dataclass(frozen=True)
class LinearGainModel:
    """Dependency-free ridge model for pre-action success-gain prediction.

    Cost preference is deliberately excluded from both inputs and targets. A
    policy converts predicted gain to VOI using its runtime lambda.
    """

    encoder: FeatureEncoder
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    alpha: float

    @classmethod
    def fit(
        cls,
        records: Sequence[ActionRecord],
        *,
        alpha: float = 1.0,
    ) -> "LinearGainModel":
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        zooms = select_zooms(records)
        if not zooms:
            raise ValueError("training requires ZOOM records")
        encoder = FeatureEncoder.fit(zooms)
        raw = encoder.transform(zooms)
        dimension = len(raw[0])
        means = [sum(row[column] for row in raw) / len(raw) for column in range(dimension)]
        scales: list[float] = []
        for column in range(dimension):
            variance = sum((row[column] - means[column]) ** 2 for row in raw) / len(raw)
            scales.append(max(math.sqrt(variance), 1e-8))
        design = [
            [1.0]
            + [(value - means[column]) / scales[column] for column, value in enumerate(row)]
            for row in raw
        ]
        targets = [record.delta_success for record in zooms]
        width = dimension + 1
        xtx = [[0.0 for _ in range(width)] for _ in range(width)]
        xty = [0.0 for _ in range(width)]
        for row, target in zip(design, targets):
            for left in range(width):
                xty[left] += row[left] * target
                for right in range(width):
                    xtx[left][right] += row[left] * row[right]
        for diagonal in range(1, width):
            xtx[diagonal][diagonal] += alpha
        weights = _solve_linear_system(xtx, xty)
        return cls(
            encoder=encoder,
            means=tuple(means),
            scales=tuple(scales),
            weights=tuple(weights),
            alpha=alpha,
        )

    def predict_gain(self, record: ActionRecord) -> float:
        raw = self.encoder.transform_one(record)
        standardized = [
            (value - self.means[column]) / self.scales[column]
            for column, value in enumerate(raw)
        ]
        return self.weights[0] + sum(
            weight * value for weight, value in zip(self.weights[1:], standardized)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": 2,
            "model_type": "linear_ridge_success_gain",
            "encoder": self.encoder.to_dict(),
            "feature_names": list(self.encoder.names),
            "means": list(self.means),
            "scales": list(self.scales),
            "weights": list(self.weights),
            "alpha": self.alpha,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearGainModel":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format_version") != 2 or value.get("model_type") != "linear_ridge_success_gain":
            raise ValueError(
                "unsupported gain-model format; retrain v1 models because their targets embed lambda"
            )
        model = cls(
            encoder=FeatureEncoder.from_dict(value["encoder"]),
            means=tuple(float(item) for item in value["means"]),
            scales=tuple(float(item) for item in value["scales"]),
            weights=tuple(float(item) for item in value["weights"]),
            alpha=float(value["alpha"]),
        )
        expected = len(model.encoder.names)
        if len(model.means) != expected or len(model.scales) != expected:
            raise ValueError("serialized model feature dimensions do not match encoder")
        if len(model.weights) != expected + 1:
            raise ValueError("serialized model weight dimensions do not match encoder")
        return model


# Compatibility import for early scaffold users. New code should use
# ``LinearGainModel`` because the model predicts gain, not cost-adjusted VOI.
LinearValueModel = LinearGainModel
