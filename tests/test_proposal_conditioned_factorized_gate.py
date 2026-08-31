from __future__ import annotations

import math

import numpy as np

from beyond_entropy.proposal_conditioned_factorized_gate import (
    FACTORIZED_CONDITIONED_ACTION_FEATURE_COUNT,
    FACTORIZED_CONDITIONED_STATE_FEATURE_COUNT,
    _fit_binary_head,
    _fit_factorized_conditioned_heads,
    _rename_candidate,
    _score_factorized_conditioned_heads,
    _weight_mass_matches_rows,
)


def test_binary_head_uses_source_balanced_row_normalized_weights_without_class_balance():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(12, 5))
    labels = [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    domains = ["d"] * 12
    sources = ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3
    _head, weights, audit = _fit_binary_head(
        features, labels, domains, sources, seed=9
    )
    assert math.isclose(float(weights.sum()), 12.0)
    assert audit["class_balancing"] is False
    for start in (0, 3, 6, 9):
        assert math.isclose(float(weights[start : start + 3].sum()), 3.0)


def test_weight_mass_audit_accepts_only_floating_sum_noise_at_full_scale():
    assert _weight_mass_matches_rows(13579.999999998905, 13580)
    assert not _weight_mass_matches_rows(13579.9999, 13580)


def test_factorized_heads_are_deterministic_and_follow_registered_score():
    rng = np.random.default_rng(12)
    n_rows = 80
    states = rng.normal(size=(n_rows, FACTORIZED_CONDITIONED_STATE_FEATURE_COUNT))
    actions = rng.normal(size=(n_rows, FACTORIZED_CONDITIONED_ACTION_FEATURE_COUNT))
    errors = np.asarray([int(index % 3 == 0) for index in range(n_rows)])
    deltas = np.zeros(n_rows)
    error_indices = np.flatnonzero(errors == 1)
    correct_indices = np.flatnonzero(errors == 0)
    deltas[error_indices[::4]] = 1.0
    deltas[correct_indices[::7]] = -1.0
    domains = ["docvqa"] * n_rows
    sources = [f"source-{index // 2}" for index in range(n_rows)]
    first, first_audit = _fit_factorized_conditioned_heads(
        states, actions, errors.tolist(), deltas.tolist(), domains, sources, seed=22
    )
    second, second_audit = _fit_factorized_conditioned_heads(
        states, actions, errors.tolist(), deltas.tolist(), domains, sources, seed=22
    )
    first_values = _score_factorized_conditioned_heads(
        first, states, actions, [1.0] * n_rows
    )
    second_values = _score_factorized_conditioned_heads(
        second, states, actions, [1.0] * n_rows
    )
    assert first_audit == second_audit
    for left, right in zip(first_values, second_values):
        assert np.array_equal(left, right)
    error_p, rescue_p, harm_p, scores = first_values
    expected = (
        error_p * rescue_p * first["rescue_magnitude"]
        - (1.0 - error_p) * harm_p * first["harm_magnitude"]
        - 0.05
    )
    assert np.allclose(scores, expected, rtol=0.0, atol=1e-15)
    assert np.isfinite(scores).all()


def test_factorized_candidate_renaming_is_recursive_and_exact():
    value = {
        "decoupled_score": 1.0,
        "nested": [{"decoupled_called": True}],
    }
    assert _rename_candidate(value) == {
        "proposal_conditioned_factorized_score": 1.0,
        "nested": [{"proposal_conditioned_factorized_called": True}],
    }
