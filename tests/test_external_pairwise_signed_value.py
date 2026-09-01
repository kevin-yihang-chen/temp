from __future__ import annotations

import numpy as np
import pytest

from beyond_entropy.external_pairwise_signed_value import (
    _fit_external_oof,
    _incumbent_index,
    _rank_actions_small_tie,
)
from beyond_entropy.scaled_action_value import _prepare_decisions
from beyond_entropy.simulate import simulate_counterfactual_dataset


class _IdentityScaler:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return values


class _TiedModel:
    def decision_function(self, values: np.ndarray) -> np.ndarray:
        return np.zeros(values.shape[0], dtype=np.float64)


def test_external_pairwise_ties_choose_smaller_action_id() -> None:
    records = simulate_counterfactual_dataset(
        n_states=8, num_candidates=4, questions_per_image=2, seed=43
    )
    prepared = _prepare_decisions(
        records, feature_mode="context-geometry", semantic_decisions=None
    )
    key = prepared.keys[0]
    ranking = _rank_actions_small_tie(
        prepared,
        [key],
        scaler=_IdentityScaler(),
        model=_TiedModel(),
    )[key]
    assert ranking.action_id == min(action.action_id for action in prepared.zooms[key])


def test_external_pairwise_nested_oof_covers_sources() -> None:
    records = simulate_counterfactual_dataset(
        n_states=80, num_candidates=4, questions_per_image=2, seed=47
    )
    prepared = _prepare_decisions(
        records, feature_mode="context-geometry", semantic_decisions=None
    )
    rankings, gains, folds, audits = _fit_external_oof(
        prepared, n_folds=4, seed=47
    )
    assert set(rankings) == set(gains) == set(prepared.keys)
    assert len(folds) == len(audits) == 4
    assert all(audit["source_overlap"] == 0 for audit in audits)
    assert all(audit["inner_source_exclusion_passed"] for audit in audits)
    assert all(np.isfinite(value) for value in gains.values())


def test_external_pairwise_incumbent_index_rejects_outcomes() -> None:
    records = simulate_counterfactual_dataset(
        n_states=8, num_candidates=4, questions_per_image=2, seed=53
    )
    prepared = _prepare_decisions(
        records, feature_mode="context-geometry", semantic_decisions=None
    )
    rows = []
    for key in prepared.keys:
        rows.append(
            {
                "state_id": key[0],
                "replicate_id": key[1],
                "source_id": prepared.baselines[key].source_id,
                "incumbent_action_id": prepared.zooms[key][0].action_id,
                "incumbent_score": 0.0,
                "incumbent_called": False,
            }
        )
    actions, scores, calls = _incumbent_index(
        rows, baselines=prepared.baselines
    )
    assert set(actions) == set(scores) == set(calls) == set(prepared.keys)
    rows[0]["correct_after"] = 1.0
    with pytest.raises(ValueError, match="leak"):
        _incumbent_index(rows, baselines=prepared.baselines)
