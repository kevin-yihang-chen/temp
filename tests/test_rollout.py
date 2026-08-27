from beyond_entropy.rollout import (
    CandidateProposal,
    ModelOutput,
    TaskSample,
    collect_sibling_rollouts,
    exact_match,
)
from beyond_entropy.schema import BBox


class FakeBackend:
    def infer(self, *, image_path, question, bbox):
        if bbox is None:
            return ModelOutput("no", 1.0, {"mode": "full"})
        return ModelOutput("yes", 0.4, {"mode": "crop"})


def test_real_backend_collection_separates_pre_and_post_fields():
    samples = [TaskSample("s1", "/tmp/image.png", "Is it present?", "yes")]

    def proposals(_sample):
        return [
            CandidateProposal(
                "zoom-0",
                BBox(0.0, 0.0, 0.5, 0.5),
                pre_action_features={"proposal_score": 0.8},
            )
        ]

    records = collect_sibling_rollouts(
        samples,
        proposals=proposals,
        backend=FakeBackend(),
        scorer=exact_match,
    )
    assert len(records) == 2
    assert records[0].correct_before == 0.0
    assert records[1].delta_success == 1.0
    assert records[1].delta_entropy == 0.6
    assert records[1].pre_action_features == {"proposal_score": 0.8}
