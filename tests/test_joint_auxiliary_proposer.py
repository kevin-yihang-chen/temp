from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import torch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - optional semantic extra
    raise unittest.SkipTest("joint auxiliary tests require torch") from exc

from beyond_entropy.joint_auxiliary_proposer import (
    _evaluate_proposals,
    _fit_variant,
    _nll_index,
    _predict_variant,
)
from beyond_entropy.schema import ActionRecord, BBox


def _decision(
    state: str,
    source: str,
    *,
    helpful_action: str = "ug-grid-00",
) -> list[ActionRecord]:
    common = {
        "state_id": state,
        "image_id": f"image-{source}",
        "source_id": source,
        "question": "where is the value?",
        "original_image": f"{source}.png",
        "replicate_id": "replicate-000",
        "generation_seed": 0,
        "entropy_before": 0.5,
        "answer_before": "wrong",
        "correct_before": 0.0,
    }
    records = [
        ActionRecord(
            **common,
            action_id="answer-now",
            action_type="ANSWER",
            candidate_bbox=None,
            entropy_after=0.5,
            answer_after="wrong",
            correct_after=0.0,
            tool_cost=0.0,
        )
    ]
    for index in range(4):
        action_id = f"ug-grid-0{index}"
        records.append(
            ActionRecord(
                **common,
                action_id=action_id,
                action_type="ZOOM",
                candidate_bbox=BBox(0.0, 0.0, 0.5, 0.5),
                entropy_after=0.4 + 0.01 * index,
                answer_after="right" if action_id == helpful_action else "wrong",
                correct_after=float(action_id == helpful_action),
                tool_cost=1.0,
            )
        )
    return records


def test_registered_variants_are_deterministic_and_keep_ablation_targets_separate():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(24, 6))
    rescue = np.asarray([index % 5 == 0 for index in range(24)], dtype=np.float64)
    harm = np.asarray([index % 7 == 0 for index in range(24)], dtype=np.float64)
    loss_gap = rng.normal(size=24)
    weights = np.linspace(0.5, 1.5, 24)
    kwargs = {
        "seed": 19,
        "epochs": 4,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "hidden_dims": (5, 3),
        "loss_weight": 0.5,
        "device": "cpu",
    }
    first, first_model = _fit_variant(
        features,
        rescue,
        harm,
        np.full_like(loss_gap, np.nan),
        weights,
        variant="task_only",
        **kwargs,
    )
    second, _ = _fit_variant(
        features,
        rescue,
        harm,
        np.full_like(loss_gap, np.nan),
        weights,
        variant="task_only",
        **kwargs,
    )
    assert first == second
    assert first["loss_gap_center"] == 0.0
    assert first["loss_gap_scale"] == 1.0
    assert set(first["final_training_loss"]) == {"rescue", "harm", "objective"}
    scores = _predict_variant(first_model, first, features, device="cpu")
    assert scores.shape == (24,)
    assert np.isfinite(scores).all()

    loss_only, _ = _fit_variant(
        features,
        np.full_like(rescue, np.nan),
        np.full_like(harm, np.nan),
        loss_gap,
        weights,
        variant="loss_only",
        **kwargs,
    )
    assert set(loss_only["final_training_loss"]) == {"loss_gap", "objective"}


def test_answer_nll_join_is_exact_and_fails_on_rollout_disagreement():
    records = _decision("state-1", "source-1")
    rows = [
        {
            "state_id": record.state_id,
            "replicate_id": record.replicate_id,
            "action_id": record.action_id,
            "source_id": record.source_id,
            "action_type": record.action_type,
            "correct_before": record.correct_before,
            "correct_after": record.correct_after,
            "answer_mean_nll": 0.5,
            "config_sha256": "a" * 64,
        }
        for record in records
    ]
    index, configs = _nll_index({"domain": records}, {"domain": rows})
    assert len(index) == 5
    assert configs == {"domain": ["a" * 64]}

    rows[1]["correct_after"] = 0.25
    try:
        _nll_index({"domain": records}, {"domain": rows})
    except ValueError as exc:
        assert "differs" in str(exc)
    else:
        raise AssertionError("answer-NLL/rollout outcome disagreement must fail")


def test_proposal_evaluation_uses_whole_sources_and_exact_random_expectation():
    records = [
        *_decision("state-1", "source-1", helpful_action="ug-grid-00"),
        *_decision("state-2", "source-2", helpful_action="ug-grid-00"),
        *_decision("state-3", "source-3", helpful_action="ug-grid-00"),
        *_decision("state-4", "source-4", helpful_action="ug-grid-00"),
    ]
    baselines = {}
    zooms = {}
    for record in records:
        key = (record.state_id, record.replicate_id)
        if record.action_type == "ANSWER":
            baselines[key] = record
        else:
            zooms.setdefault(key, []).append(record)
    helpful = {key: "ug-grid-00" for key in baselines}
    wrong = {key: "ug-grid-01" for key in baselines}
    report = _evaluate_proposals(
        actions_by_method={
            "joint": helpful,
            "task_only": wrong,
            "loss_only": wrong,
            "factorized": wrong,
            "random_exact": None,
        },
        baselines=baselines,
        zooms=zooms,
        bootstrap_resamples=100,
        bootstrap_seed=3,
    )
    assert report["source_balanced"]["joint"]["gain"] == 1.0
    assert report["source_balanced"]["joint"]["helpful_state_recovery"] == 1.0
    assert report["source_balanced"]["random_exact"]["gain"] == 0.25
    assert report["primary_comparisons"]["joint_minus_task_only_gain"][
        "point_estimate"
    ] == 1.0
    assert math.isfinite(
        report["primary_comparisons"]["joint_minus_factorized_gain"]["ci_low"]
    )
