import json

import pytest

from beyond_entropy.dataset import read_jsonl, split_by_state, validate_sibling_groups, write_jsonl
from beyond_entropy.features import FeatureEncoder
from beyond_entropy.schema import ActionRecord, BBox
from beyond_entropy.simulate import simulate_counterfactual_dataset


def test_bbox_and_record_round_trip(tmp_path):
    records = simulate_counterfactual_dataset(n_states=2, num_candidates=2, seed=1)
    path = tmp_path / "records.jsonl"
    write_jsonl(records, path)
    loaded = read_jsonl(path)
    assert loaded == records
    assert loaded[1].candidate_bbox.area > 0.0


def test_split_keeps_siblings_together():
    records = simulate_counterfactual_dataset(n_states=10, num_candidates=4, seed=2)
    train, test = split_by_state(records, train_fraction=0.6, seed=2)
    assert {record.state_id for record in train}.isdisjoint(
        {record.state_id for record in test}
    )


def test_feature_encoder_rejects_post_action_leakage():
    records = simulate_counterfactual_dataset(n_states=2, num_candidates=2, seed=3)
    zoom = records[1]
    leaked = ActionRecord.from_dict(
        {**zoom.to_dict(), "pre_action_features": {"entropy_after_hint": 0.2}}
    )
    with pytest.raises(ValueError, match="leakage"):
        FeatureEncoder.fit([leaked])


def test_group_requires_answer_sibling():
    records = simulate_counterfactual_dataset(n_states=2, num_candidates=2, seed=4)
    with pytest.raises(ValueError, match="exactly one ANSWER"):
        validate_sibling_groups([record for record in records if record.action_type == "ZOOM"])


def test_invalid_bbox_is_rejected():
    with pytest.raises(ValueError):
        BBox(0.5, 0.2, 0.4, 0.8)
