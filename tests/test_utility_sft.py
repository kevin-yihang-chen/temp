from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from beyond_entropy.rollout import AgentState
from beyond_entropy.schema import ActionRecord, BBox
from beyond_entropy.spatial_action_space import SpatialActionSpace
from beyond_entropy.utility_dataset import UtilityInputs, audit_utility_splits, build_utility_samples, load_utility_development


def siblings(*, replicate="r0", seed=17, state_id="s1", image_id="im1",
             source_id="source1", image_path="/tmp/utility-test-image.png"):
    common = dict(state_id=state_id, image_id=image_id, source_id=source_id, question="q?",
                  original_image=image_path, replicate_id=replicate,
                  generation_seed=seed, entropy_before=.4, answer_before="base", correct_before=.5)
    return [
        ActionRecord(**common, action_id="answer-now", action_type="ANSWER", candidate_bbox=None,
                     entropy_after=.4, answer_after="base", correct_after=.5, tool_cost=0),
        ActionRecord(**common, action_id="crop-b", action_type="ZOOM", candidate_bbox=BBox(.5, 0, 1, 1),
                     entropy_after=.1, answer_after="bad", correct_after=0, tool_cost=1),
        ActionRecord(**common, action_id="crop-a", action_type="ZOOM", candidate_bbox=BBox(0, 0, .5, 1),
                     entropy_after=.3, answer_after="good", correct_after=1, tool_cost=1),
    ]


def make_samples(rows=None, *, state_id="s1", image_id="im1", source_id="source1",
                 image_path="/tmp/utility-test-image.png", benchmark="chartqa",
                 role="train", rgb_hash="a"*64, **kwargs):
    records = siblings(state_id=state_id, image_id=image_id, source_id=source_id,
                       image_path=image_path) if rows is None else rows
    return build_utility_samples(
        records,
        states={state_id: AgentState(state_id, image_id, source_id, image_path, "q?", model_prompt="q? A: yes B: no")},
        rgb_hashes={state_id: rgb_hash}, benchmark=benchmark, role=role, **kwargs,
    )


def test_action_mapping_cost_and_execution():
    space = SpatialActionSpace.from_siblings(siblings())
    assert [a.name for a in space.actions] == ["ANSWER", "ZOOM_1", "ZOOM_2"]
    assert [a.action_id for a in space.actions] == ["answer-now", "crop-a", "crop-b"]
    assert space == SpatialActionSpace.from_siblings(siblings()[::-1])
    assert space.select([0, .5, -.5], lambda_cost=0) == 1
    assert space.select([0, .5, -.5], lambda_cost=.5) == 0
    sample = make_samples()[0]
    request = space.request(1, sample.inputs.state, generation_seed=17)
    assert len(request.observations) == 2
    assert request.observations[1].bbox == BBox(0, 0, .5, 1)
    assert request.state.backend_prompt.endswith("A: yes B: no")
    assert len(space.request(0, sample.inputs.state, generation_seed=17).observations) == 1
    for bad in (-1, 3, True):
        with pytest.raises(ValueError):
            space.request(bad, sample.inputs.state, generation_seed=17)
    with pytest.raises(ValueError):
        space.select([0, float("nan"), 1], lambda_cost=0)
    with pytest.raises(ValueError):
        space.select([0, 1, 1], lambda_cost=float("nan"))


def test_dataset_keeps_paired_labels_and_isolates_inputs():
    sample = make_samples()[0]
    assert sample.rewards == (.5, 1, 0)
    assert sample.gains == (0, .5, -.5)
    assert sample.best_action == 1
    assert sample.inputs == UtilityInputs.from_dict(sample.inputs.to_dict())
    assert make_samples(siblings()[::-1])[0] == sample
    changed = [replace(r, correct_after=.5, entropy_after=.9, answer_after="other")
               if r.action_type == "ZOOM" else r for r in siblings()]
    changed_sample = make_samples(changed)[0]
    assert changed_sample.inputs == sample.inputs
    assert changed_sample.support_action == sample.support_action
    assert changed_sample.gains != sample.gains
    assert changed_sample.best_action == 0
    for field in ("answer_after", "correct_after", "reward", "target", "entropy_after", "sibling_outcome"):
        row = sample.inputs.to_dict()
        row[field] = "leak"
        with pytest.raises(ValueError, match="allowlist"):
            UtilityInputs.from_dict(row)
        row = sample.inputs.to_dict()
        row["actions"][1][field] = "leak"
        with pytest.raises(ValueError, match="unexpected"):
            UtilityInputs.from_dict(row)


def test_replicate_aggregation_and_incomplete_mapping_rejected():
    r1 = siblings(replicate="r1", seed=29)
    r1 = [replace(r, correct_after=.5) if r.action_type == "ZOOM" else r for r in r1]
    combined = siblings() + r1
    with pytest.raises(ValueError, match="exactly one"):
        make_samples(combined)
    sample = make_samples(combined, aggregation="mean")[0]
    assert sample.gains == (0, .25, -.25)
    assert len(sample.outcomes) == 6
    with pytest.raises(ValueError, match="mapping changed"):
        make_samples(combined[:-1], aggregation="mean")
    duplicate_seed = siblings() + siblings(replicate="r1", seed=17)
    with pytest.raises(ValueError, match="duplicate seeds"):
        make_samples(duplicate_seed, aggregation="mean")
    with pytest.raises(ValueError, match="generation_seed"):
        make_samples([replace(siblings()[0], generation_seed=1), *siblings()[1:]])
    with pytest.raises(ValueError, match="baseline answer"):
        make_samples([siblings()[0], replace(siblings()[1], answer_before="wrong"), siblings()[2]])


def test_source_rgb_and_image_disjointness():
    train = make_samples()[0]
    state = replace(train.inputs.state, state_id="s2", source_id="source2", image_id="im2")
    val = replace(train, inputs=replace(train.inputs, state=state), role="validation", image_rgb_sha256="b"*64)
    assert audit_utility_splits([train, val])["passed"]
    with pytest.raises(ValueError, match="RGB split leakage"):
        audit_utility_splits([train, replace(val, image_rgb_sha256=train.image_rgb_sha256)])
    with pytest.raises(ValueError, match="RGB split leakage"):
        audit_utility_splits([train, replace(val, inputs=replace(val.inputs, state=replace(state, source_id="source1")))])
    with pytest.raises(ValueError, match="image-ID"):
        audit_utility_splits([train, replace(val, inputs=replace(val.inputs, state=replace(state, image_id="im1")))])


def test_development_roundtrip_rechecks_labels_and_refuses_test(tmp_path):
    sample = make_samples()[0]
    payload = {"schema": "utility_sft_dataset_v1", "role": "train", "benchmark": "chartqa",
               "aggregation": "single", "samples": [sample.to_dict()]}
    path = tmp_path / "data.json"
    path.write_text(json.dumps(payload))
    assert load_utility_development(path, role="train") == [sample]
    with pytest.raises(ValueError, match="cannot open test"):
        load_utility_development(tmp_path / "nonexistent-test.json", role="test")
    changed = copy.deepcopy(payload)
    changed["samples"][0]["labels"]["gain"][1] = .9
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differ"):
        load_utility_development(path, role="train")


def test_loss_formula_scale_ties_and_format_label_isolation():
    torch = pytest.importorskip("torch")
    from beyond_entropy.utility_head import utility_sft_loss
    gains = torch.tensor([[0., .5, -.5], [0., 0., -1.]])
    logits = (gains / .25).requires_grad_()
    assert abs(utility_sft_loss(logits, method="utility", gains=gains).item()) < 1e-6
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 0]))
    assert torch.allclose(utility_sft_loss(logits, method="best_action", gains=gains), ce)
    with pytest.raises(ValueError, match="must not receive"):
        utility_sft_loss(logits, method="format", gains=gains, support_labels=torch.tensor([0, 1]))
    assert torch.isfinite(utility_sft_loss(logits, method="format", support_labels=torch.tensor([0, 1])))
    wrong = (-logits.detach()).requires_grad_()
    loss = utility_sft_loss(wrong, method="utility", gains=gains)
    loss.backward()
    assert wrong.grad[0, 1] < 0 and wrong.grad[0, 2] > 0
    assert utility_sft_loss(logits, method="pairwise", gains=gains) < utility_sft_loss(-logits, method="pairwise", gains=gains)


def test_head_roi_gradient_and_answer_anchor():
    torch = pytest.importorskip("torch")
    from beyond_entropy.utility_head import SpatialUtilityHead, utility_sft_loss
    torch.manual_seed(17)
    head = SpatialUtilityHead(8, head_dim=12)
    question = torch.randn(1, 8, requires_grad=True)
    visual = torch.randn(1, 4, 4, 8, requires_grad=True)
    boxes = torch.tensor([[[0., 0., .5, 1.], [.5, 0., 1., 1.]]])
    out = head(question, visual, boxes)
    assert out["predicted_gain"].shape == (1, 3)
    assert out["predicted_gain"][0, 0] == 0
    assert torch.allclose(out["action_logits"]*.25, out["predicted_gain"])
    utility_sft_loss(out["action_logits"], method="utility", gains=torch.tensor([[0., 1., -1.]])).backward()
    for grad in (question.grad, visual.grad, head.answer_embedding.grad):
        assert grad is not None and grad.abs().sum() > 0


def test_training_configs_are_matched_except_objective():
    root = Path(__file__).resolve().parents[1] / "configs"
    configs = [json.loads((root / f"utility_sft_{name}_v1.json").read_text())
               for name in ("format", "best_action", "utility")]
    assert {c.pop("method") for c in configs} == {"format", "best_action", "utility"}
    assert configs[0] == configs[1] == configs[2]
    assert configs[0]["test_authorized"] is False


def test_sanity_gates_only_require_identifiable_objective_outputs():
    from beyond_entropy.utility_training import required_sanity_checks, sanity_passed
    common = {name: True for name in required_sanity_checks("utility")}
    common["support_memorized"] = False
    assert sanity_passed("utility", common, engineering=False)
    assert not sanity_passed("utility", common, engineering=True)
    best = dict(common, positive_negative_separated=False)
    assert sanity_passed("best_action", best, engineering=False)
    formatted = dict(common, support_memorized=True, overfit_regret_zero=False,
                     positive_negative_separated=False)
    assert sanity_passed("format", formatted, engineering=False)
    with pytest.raises(ValueError, match="unknown"):
        required_sanity_checks("rl")


def test_source_hash_subset_is_deterministic_whole_source_and_outcome_free():
    from beyond_entropy.utility_training import source_hash_subset
    samples = []
    for index, source in enumerate(("a", "a", "b", "c", "d")):
        sample = make_samples(
            state_id=f"s{index}", image_id=f"i{index}", source_id=source,
            rgb_hash=f"{index+1:064x}",
        )[0]
        samples.append(sample)
    selected = source_hash_subset(samples, maximum_sources=2, seed=17, namespace="pilot")
    assert selected == source_hash_subset(samples[::-1], maximum_sources=2, seed=17, namespace="pilot")
    selected_sources = {s.inputs.state.source_id for s in selected}
    assert len(selected_sources) == 2
    assert all((s in selected) == (s.inputs.state.source_id in selected_sources) for s in samples)
    changed = [SimpleNamespace(inputs=s.inputs, gains=tuple(-g for g in s.gains)) for s in samples]
    # Selection depends only on source identity; labels do not enter ordering.
    assert [s.inputs.state.state_id for s in source_hash_subset(changed, maximum_sources=2, seed=17, namespace="pilot")] == [s.inputs.state.state_id for s in selected]
    with pytest.raises(ValueError, match="positive"):
        source_hash_subset(samples, maximum_sources=0, seed=17, namespace="pilot")


def test_development_configs_are_matched_except_objective():
    root = Path(__file__).resolve().parents[1] / "configs"
    configs = [json.loads((root / f"utility_sft_development_{name}_v1.json").read_text())
               for name in ("format", "best_action", "utility")]
    assert {c.pop("method") for c in configs} == {"format", "best_action", "utility"}
    assert configs[0] == configs[1] == configs[2]
    assert configs[0]["scope"] == "three_domain_development_pilot"
    assert configs[0]["domain_sampling"] == "uniform_domain_then_source"
    assert configs[0]["test_authorized"] is False


def test_correction_configs_are_matched_and_pre_registered():
    root = Path(__file__).resolve().parents[1] / "configs"
    configs = [
        json.loads((root / f"utility_sft_correction_{name}_v1.json").read_text())
        for name in ("format", "best_action", "utility")
    ]
    assert {config.pop("method") for config in configs} == {
        "format", "best_action", "utility"
    }
    assert configs[0] == configs[1] == configs[2]
    assert configs[0]["scope"] == "three_domain_development_correction"
    assert configs[0]["domain_sampling"] == "uniform_domain_then_source_cycle"
    assert configs[0]["steps"] == 1024
    assert configs[0]["test_authorized"] is False


def test_source_cycle_is_deterministic_and_covers_every_source_per_cycle():
    from beyond_entropy.utility_training import source_cycle_samples

    grouped = {}
    for index, source in enumerate(("source-a", "source-b", "source-c")):
        grouped[source] = [make_samples(
            state_id=f"cycle-{index}", image_id=f"image-{index}",
            source_id=source, rgb_hash=f"{index+1:064x}",
        )[0]]
    first = source_cycle_samples(grouped, draws=7, seed=17, namespace="test")
    second = source_cycle_samples(grouped, draws=7, seed=17, namespace="test")
    assert first == second
    expected = set(grouped)
    assert {sample.inputs.state.source_id for sample in first[:3]} == expected
    assert {sample.inputs.state.source_id for sample in first[3:6]} == expected
    with pytest.raises(ValueError, match="positive draws"):
        source_cycle_samples(grouped, draws=0, seed=17, namespace="test")
