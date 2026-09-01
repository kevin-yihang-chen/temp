from __future__ import annotations

import numpy as np
import pytest

from beyond_entropy.cost_sensitive_direct_action_value import (
    _fit_oof,
    _score_head,
    _source_utility_weights,
)
from beyond_entropy.scaled_action_value import _prepare_decisions
from beyond_entropy.simulate import simulate_counterfactual_dataset


def test_cost_sensitive_weights_equalize_source_mass() -> None:
    utilities = [-0.05, 0.95, -1.05, -0.05]
    sources = ["a", "a", "b", "b"]
    weights = _source_utility_weights(utilities, sources)
    assert weights.sum() == pytest.approx(4.0, abs=1e-12)
    assert weights[:2].sum() == pytest.approx(2.0, abs=1e-12)
    assert weights[2:].sum() == pytest.approx(2.0, abs=1e-12)
    assert weights[1] > weights[0]
    assert weights[2] > weights[3]


def test_cost_sensitive_weights_reject_zero_utility() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        _source_utility_weights([-0.05, 0.0], ["a", "a"])


def test_cost_sensitive_oof_is_source_disjoint_and_complete() -> None:
    records = simulate_counterfactual_dataset(
        n_states=80, num_candidates=4, questions_per_image=2, seed=59
    )
    prepared = _prepare_decisions(
        records, feature_mode="context-geometry", semantic_decisions=None
    )
    actions, scores, folds, audits, fold_by_key = _fit_oof(
        prepared, n_folds=4, seed=59
    )
    assert set(actions) == set(scores) == set(fold_by_key) == set(prepared.keys)
    assert len(folds) == len(audits) == 4
    assert all(audit["source_overlap"] == 0 for audit in audits)
    assert all(audit["source_mass_min"] == pytest.approx(audit["source_mass_max"], abs=1e-8) for audit in audits)
    assert all(np.isfinite(value) for value in scores.values())


class _IdentityScaler:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return values


class _TiedModel:
    def decision_function(self, values: np.ndarray) -> np.ndarray:
        return np.zeros(values.shape[0], dtype=np.float64)


def test_cost_sensitive_ties_choose_smaller_action_id() -> None:
    records = simulate_counterfactual_dataset(
        n_states=8, num_candidates=4, questions_per_image=2, seed=61
    )
    prepared = _prepare_decisions(
        records, feature_mode="context-geometry", semantic_decisions=None
    )
    key = prepared.keys[0]
    actions, scores = _score_head(
        prepared,
        [key],
        head={"scaler": _IdentityScaler(), "model": _TiedModel()},
    )
    assert actions[key] == min(action.action_id for action in prepared.zooms[key])
    assert scores[key] == 0.0
