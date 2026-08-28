from beyond_entropy.rollout import (
    ActionSpec,
    AgentState,
    CachedVisualBackend,
    GroundTruth,
    InferenceRequest,
    ModelOutput,
    TaskExample,
    VisualObservation,
    collect_sibling_rollouts,
    exact_match,
    infer_many,
)
from beyond_entropy.schema import BBox


class FakeBackend:
    def __init__(self):
        self.calls = []

    def infer(self, *, state, observations, generation_seed):
        self.calls.append((state, observations, generation_seed))
        if len(observations) == 1:
            return ModelOutput("no", 1.0, {"mode": "full"})
        return ModelOutput("yes", 0.4, {"mode": "additive-zoom"})


def test_collection_isolates_target_and_adds_zoom_observation():
    state = AgentState(
        state_id="s1",
        image_id="i1",
        source_id="source-1",
        image_path="/tmp/image.png",
        question="Is it present?",
    )
    examples = [TaskExample(state, GroundTruth("yes"))]
    proposer_inputs = []

    def proposals(agent_state):
        proposer_inputs.append(agent_state)
        assert not hasattr(agent_state, "target")
        return [
            ActionSpec(
                "zoom-0",
                BBox(0.0, 0.0, 0.5, 0.5),
                pre_action_features={"proposal_score": 0.8},
            )
        ]

    backend = FakeBackend()
    records = collect_sibling_rollouts(
        examples,
        proposals=proposals,
        backend=backend,
        scorer=exact_match,
        generation_seeds=(11, 12),
    )
    assert len(records) == 4
    assert proposer_inputs == [state]
    assert {record.replicate_id for record in records} == {
        "replicate-000",
        "replicate-001",
    }
    assert records[0].correct_before == 0.0
    assert records[1].delta_success == 1.0
    assert records[1].delta_entropy == 0.6
    assert records[1].pre_action_features == {"proposal_score": 0.8}
    zoom_calls = [call for call in backend.calls if len(call[1]) == 2]
    assert len(zoom_calls) == 2
    assert all(call[1][0].kind == "ORIGINAL" for call in zoom_calls)
    assert all(call[1][1].kind == "ZOOM" for call in zoom_calls)


def test_cached_backend_avoids_duplicate_requests():
    state = AgentState("s1", "i1", "source-1", "/tmp/image.png", "Question?")
    request = InferenceRequest(
        state,
        (VisualObservation("ORIGINAL", state.image_path, "original", None),),
        5,
    )
    raw_backend = FakeBackend()
    cached = CachedVisualBackend(raw_backend)
    first = infer_many(cached, [request])
    second = infer_many(cached, [request])
    assert first == second
    assert len(raw_backend.calls) == 1


def test_inference_cache_key_binds_backend_only_model_prompt():
    observation = VisualObservation("ORIGINAL", "/tmp/image.png", "original", None)
    plain = AgentState("s1", "i1", "source-1", "/tmp/image.png", "Core question")
    formatted = AgentState(
        "s1",
        "i1",
        "source-1",
        "/tmp/image.png",
        "Core question",
        model_prompt="Formatted backend prompt",
    )
    plain_request = InferenceRequest(plain, (observation,), 0)
    formatted_request = InferenceRequest(formatted, (observation,), 0)
    assert plain_request.cache_key() != formatted_request.cache_key()
