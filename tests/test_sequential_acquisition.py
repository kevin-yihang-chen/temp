import json

import pytest

from beyond_entropy.acquisition_critic import (
    AcquisitionInputs,
    audit_development_disjointness,
    build_sequential_feature_row,
)
from beyond_entropy.rollout import ActionSpec, AgentState, GroundTruth, ModelOutput, TaskExample
from beyond_entropy.schema import BBox
from beyond_entropy.sequential_metrics import (
    paired_source_bootstrap_utility_delta,
    policy_metrics,
)
from beyond_entropy.sequential_rollout import SequentialPrefix, collect_counterfactual_prefixes
from beyond_entropy.sequential_schema import SequentialRolloutRecord
from beyond_entropy.sequential_test_transaction import (
    start_test_transaction,
    validate_test_access,
)


class PrefixBackend:
    def __init__(self):
        self.calls = []

    def infer(self, *, state, observations, generation_seed):
        self.calls.append((state, observations, generation_seed))
        count = len(observations)
        assert count in (2, 3)
        return ModelOutput(
            "yes" if count == 3 else "no",
            0.2 if count == 3 else 0.7,
            {
                "num_observations": count,
                "mean_maximum_token_probability": 0.8 if count == 3 else 0.6,
                "mean_top1_top2_token_probability_margin": 0.7 if count == 3 else 0.4,
                "normalized_token_entropies": [0.7],
                "generated_token_log_probabilities": [-0.4],
                "system_prompt": "system",
            },
        )


def fixed_prefix(state):
    assert isinstance(state, AgentState)
    assert not hasattr(state, "target")
    return SequentialPrefix(
        (ActionSpec("crop-a", BBox(0, 0, 0.5, 0.5), 2.0),),
        ActionSpec("crop-b", BBox(0.5, 0.5, 1, 1), 3.0),
        "test-fixed-prefix",
    )


def one_record():
    state = AgentState("s1", "i1", "source-1", "/tmp/image.png", "Question?")
    backend = PrefixBackend()
    records = collect_counterfactual_prefixes(
        (TaskExample(state, GroundTruth("yes")),),
        prefixes=fixed_prefix,
        backend=backend,
        scorer=lambda answer, truth: float(answer == truth.target),
        generation_seeds=(11,),
    )
    return records[0], backend


def test_paired_prefix_isolates_ground_truth_and_freezes_seed_and_configuration():
    record, backend = one_record()
    assert record.delta_success == 1.0
    assert len(backend.calls) == 2
    stop, continued = backend.calls
    assert stop[0] == continued[0]
    assert stop[2] == continued[2] == 11
    assert stop[1] == continued[1][:-1]
    assert [item.action_id for item in stop[1]] == ["original", "crop-a"]
    assert continued[1][-1].action_id == "crop-b"
    assert record.stop_backend["num_observations"] == 2
    assert record.continue_backend["num_observations"] == 3


def test_stop_has_no_extra_cost_and_total_cost_is_additive():
    record, _ = one_record()
    assert record.stop_total_visual_cost == 2.0
    assert record.continue_total_visual_cost == 5.0
    stop = policy_metrics((record,), (False,), lambda_cost=0.1, policy_name="stop")
    continued = policy_metrics((record,), (True,), lambda_cost=0.1, policy_name="continue")
    assert stop["avg_incremental_visual_cost"] == 0.0
    assert stop["avg_total_visual_cost"] == 2.0
    assert continued["avg_incremental_visual_cost"] == 3.0
    assert continued["avg_total_visual_cost"] == 5.0


def test_sequential_schema_round_trip_preserves_counterfactual_pair():
    record, _ = one_record()
    rebuilt = SequentialRolloutRecord.from_dict(json.loads(json.dumps(record.to_dict())))
    assert rebuilt == record
    assert rebuilt.generation_seed == 11


def test_feature_contract_excludes_post_action_values():
    record, _ = one_record()
    dimension = 4
    semantic = {
        "question_embedding": [0.1] * dimension,
        "global_visual_embedding": [0.2] * dimension,
        "region_embeddings": [[0.3] * dimension, [0.4] * dimension],
        "bboxes": [
            record.acquired_observations[0].bbox.to_list(),
            record.proposed_bbox.to_list(),
        ],
    }
    current = {
        "pooled_language_state": [0.5] * dimension,
        "pooled_visual_state": [0.6] * dimension,
        "fused_multimodal_state": [0.7] * dimension,
    }
    row = build_sequential_feature_row(
        record,
        semantic=semantic,
        current_multimodal=current,
        image_rgb_sha256="a" * 64,
    )
    inputs = AcquisitionInputs.from_untrusted_mapping(row)
    assert len(inputs.feature_vector("state_semantic")) > dimension
    assert "labels" not in inputs.__dict__
    tampered = dict(row)
    tampered["pre_action"] = dict(row["pre_action"], continue_entropy=0.0)
    with pytest.raises(ValueError, match="strict allowlist"):
        AcquisitionInputs.from_untrusted_mapping(tampered)


def test_source_and_rgb_split_audit_rejects_leakage():
    train = [{"source_id": "a", "image_rgb_sha256": "1" * 64}]
    validation = [{"source_id": "b", "image_rgb_sha256": "2" * 64}]
    assert audit_development_disjointness(train, validation)["passed"] is True
    with pytest.raises(ValueError, match="leakage"):
        audit_development_disjointness(
            train, [{"source_id": "a", "image_rgb_sha256": "3" * 64}]
        )


def test_vectorized_paired_source_bootstrap_preserves_observed_utility_delta():
    record, _ = one_record()
    result = paired_source_bootstrap_utility_delta(
        (record,),
        (True,),
        (False,),
        lambda_cost=0.1,
        samples=10_000,
        seed=3,
    )
    assert result["observed_delta"] == pytest.approx(0.7)
    assert result["ci_low"] == pytest.approx(0.7)
    assert result["ci_high"] == pytest.approx(0.7)


def test_test_transaction_writes_irreversible_ledger_before_manifest_access(tmp_path):
    missing_manifest = tmp_path / "not-opened-test.jsonl"
    rollouts = tmp_path / "formal" / "rollouts.jsonl"
    features = tmp_path / "formal" / "features.pt"
    freeze_path = tmp_path / "freeze.json"
    ledger_path = tmp_path / "formal" / "access.json"
    freeze = {
        "schema": "sequential_test_freeze_v1",
        "one_shot": True,
        "test_authorized": True,
        "benchmark": "chartqa",
        "manifest_path": str(missing_manifest.resolve()),
        "expected_manifest_sha256": "a" * 64,
        "rollouts_output": str(rollouts.resolve()),
        "features_output": str(features.resolve()),
        "config_path": str((tmp_path / "config.json").resolve()),
        "config_sha256": "b" * 64,
        "critics_path": str((tmp_path / "critics.pt").resolve()),
        "critics_sha256": "c" * 64,
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_revision": "revision",
        "generation_seeds": [0],
        "proposer": "sequential-opposite-ug-v1",
        "candidate_count": 4,
        "visual_cost_per_crop": 1.0,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "max_new_tokens": 16,
        "min_pixels": 200704,
        "max_pixels": 602112,
        "manifest_limit": None,
        "shard_count": 1,
        "shard_index": 0,
        "bootstrap_samples": 10000,
        "bootstrap_seed": 3,
        "code_revision": "d" * 40,
    }
    freeze_path.write_text(json.dumps(freeze))
    start_test_transaction(
        freeze_path,
        ledger_path,
        benchmark="chartqa",
        manifest_path=missing_manifest,
        rollouts_output=rollouts,
        features_output=features,
        model=freeze["model"],
        model_revision=freeze["model_revision"],
        generation_seeds=[0],
        code_revision=freeze["code_revision"],
        dtype="bfloat16",
        attention_implementation="sdpa",
        max_new_tokens=16,
        min_pixels=200704,
        max_pixels=602112,
        manifest_limit=None,
        shard_count=1,
        shard_index=0,
    )
    assert ledger_path.exists()
    validate_test_access(freeze_path, ledger_path)
    with pytest.raises(FileExistsError):
        start_test_transaction(
            freeze_path,
            ledger_path,
            benchmark="chartqa",
            manifest_path=missing_manifest,
            rollouts_output=rollouts,
            features_output=features,
            model=freeze["model"],
            model_revision=freeze["model_revision"],
            generation_seeds=[0],
            code_revision=freeze["code_revision"],
            dtype="bfloat16",
            attention_implementation="sdpa",
            max_new_tokens=16,
            min_pixels=200704,
            max_pixels=602112,
            manifest_limit=None,
            shard_count=1,
            shard_index=0,
        )
