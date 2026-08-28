import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from beyond_entropy.risk_control import (
    RiskConstraint,
    calibrate_source_risk_threshold,
)
from beyond_entropy.scaled_action_value import (
    _serialize_linear,
    _serialized_linear_predict,
    acquisition_calibration_rows,
    fit_scaled_pairwise_action_value_model,
    predict_scaled_action_value,
)
from beyond_entropy.simulate import simulate_counterfactual_dataset


@pytest.mark.parametrize("model", [Ridge(alpha=1.0), LogisticRegression(C=0.5)])
def test_serialized_linear_predictions_match_sklearn(model):
    features = np.asarray(
        [
            [-2.0, 0.0, 1.0],
            [-1.0, 1.0, 0.5],
            [0.5, -1.0, -0.5],
            [1.5, 0.5, -1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([0, 0, 1, 1, 1], dtype=np.float64)
    scaler = StandardScaler().fit(features)
    model.fit(scaler.transform(features), targets)
    payload = _serialize_linear(scaler, model)
    expected = (
        model.decision_function(scaler.transform(features))
        if isinstance(model, LogisticRegression)
        else model.predict(scaler.transform(features))
    )
    actual = np.asarray(
        [_serialized_linear_predict(payload, row) for row in features]
    )
    assert actual == pytest.approx(expected, abs=1e-12)


def test_scaled_pairwise_action_value_crossfits_and_round_trips():
    records = simulate_counterfactual_dataset(
        n_states=80,
        num_candidates=4,
        questions_per_image=2,
        seed=31,
    )
    report, model = fit_scaled_pairwise_action_value_model(
        records,
        n_folds=4,
        ranker_c_values=(0.1, 1.0),
        call_alpha_values=(1.0, 10.0),
        max_thresholds=12,
        bootstrap_resamples=50,
        seed=31,
    )
    predictions = predict_scaled_action_value(model, records)
    assert report["n_sources"] == 40
    assert report["n_decisions"] == 80
    assert report["calibration_outcomes_used"] is False
    assert model["calibrated_threshold"] is None
    assert len(model["threshold_grid"]) <= 12
    assert len(predictions) == 80
    assert len({(item.state_id, item.replicate_id) for item in predictions}) == 80
    assert all(item.action_id.startswith("zoom-") for item in predictions)


def test_scaled_predictions_join_to_independent_risk_calibration_rows():
    records = simulate_counterfactual_dataset(
        n_states=60,
        num_candidates=4,
        questions_per_image=2,
        seed=37,
    )
    _, model = fit_scaled_pairwise_action_value_model(
        records,
        n_folds=3,
        ranker_c_values=(0.1,),
        call_alpha_values=(10.0,),
        max_thresholds=8,
        bootstrap_resamples=30,
        seed=37,
    )
    predictions = predict_scaled_action_value(model, records)
    rows = acquisition_calibration_rows(predictions, records)
    result = calibrate_source_risk_threshold(
        rows,
        model["threshold_grid"],
        constraints=[RiskConstraint("induced_harm", 0.4)],
        min_source_call_rate=0.01,
        min_source_utility=-1.0,
    )
    assert result["n_sources"] == 30
    assert result["n_decisions"] == 60
