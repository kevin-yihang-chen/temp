import json

import pytest

from beyond_entropy.benchmarks import load_manifest, scorer_by_name
from beyond_entropy.chartqapro import (
    build_chartqapro_direct_prompt,
    build_chartqapro_gate_context,
    canonicalize_chartqapro_constrained_answer,
    chartqapro_match,
    chartqapro_spec_match,
    chartqapro_target,
    select_chartqapro_pilot_images,
)
from beyond_entropy.rollout import GroundTruth


def test_direct_prompt_keeps_paragraph_and_excludes_final_answer():
    prompt = build_chartqapro_direct_prompt(
        ["Which category is largest?"],
        ["FINAL-ANSWER-SENTINEL"],
        "Factoid",
        "Relevant paragraph context.",
    )
    assert prompt.startswith("Relevant paragraph context.")
    assert "Which category is largest?" in prompt
    assert "FINAL-ANSWER-SENTINEL" not in prompt
    assert "final answer only" in prompt


def test_conversational_prompt_keeps_history_but_excludes_final_answer():
    prompt = build_chartqapro_direct_prompt(
        ["First question?", "Follow-up question?", "Final question?"],
        ["first answer", "second answer", "FINAL-ANSWER-SENTINEL"],
        "Conversational",
        None,
    )
    assert "First question?" in prompt
    assert "first answer" in prompt
    assert "Follow-up question?" in prompt
    assert "second answer" in prompt
    assert "Final question?" in prompt
    assert "FINAL-ANSWER-SENTINEL" not in prompt

    gate_context = build_chartqapro_gate_context(
        ["First question?", "Follow-up question?", "Final question?"],
        ["first answer", "second answer", "FINAL-ANSWER-SENTINEL"],
        "Conversational",
        "Relevant paragraph.",
    )
    assert "Relevant paragraph." in gate_context
    assert "First question?" in gate_context
    assert "first answer" in gate_context
    assert "Final question?" in gate_context
    assert "FINAL-ANSWER-SENTINEL" not in gate_context
    assert "generate the final answer" not in gate_context


def test_chartqapro_prompt_validates_turn_structure():
    with pytest.raises(ValueError, match="same number"):
        build_chartqapro_direct_prompt(
            ["First?", "Final?"],
            ["only one answer"],
            "Conversational",
            None,
        )
    with pytest.raises(ValueError, match="non-empty"):
        chartqapro_target([""], ["NO"], "Factoid")
    with pytest.raises(ValueError, match="exactly one"):
        build_chartqapro_direct_prompt(
            ["one", "two"],
            ["a", "b"],
            "Factoid",
            None,
        )


@pytest.mark.parametrize(
    ("answer", "question_type", "expected"),
    [
        ("b", "Multi Choice", "b"),
        ("b) 31%", "Multi Choice", "b"),
        ("(C)", "Multi Choice", "c"),
        ("[true]", "Fact Checking", "true"),
        ("[’FALSE’]", "Fact Checking", "false"),
        ("the answer is b", "Multi Choice", "the answer is b"),
        ("because it is true", "Fact Checking", "because it is true"),
        ("North", "Factoid", "North"),
    ],
)
def test_constrained_answer_canonicalization_is_conservative(
    answer: str,
    question_type: str,
    expected: str,
):
    assert canonicalize_chartqapro_constrained_answer(answer, question_type) == expected


def test_chartqapro_match_numeric_year_text_and_lists():
    numeric = GroundTruth(chartqapro_target(["100"], ["NO"], "Factoid"))
    assert chartqapro_match("104.9", numeric) == 1.0
    assert chartqapro_match("106", numeric) == 0.0

    year = GroundTruth(chartqapro_target(["2020"], ["YES"], "Factoid"))
    assert chartqapro_match("2020", year) == 1.0
    assert chartqapro_match("2021", year) == 0.0

    text = GroundTruth(chartqapro_target(["females"], ["NO"], "Factoid"))
    assert chartqapro_match("female", text) == pytest.approx(6 / 7)

    answers = GroundTruth(
        chartqapro_target(["['north', 'south']"], ["NO"], "Factoid")
    )
    assert chartqapro_match("['north', 'south']", answers) == 1.0
    assert chartqapro_match("['north']", answers) == 0.5


def test_chartqapro_match_scores_only_final_conversation_turn():
    target = GroundTruth(
        chartqapro_target(
            ["old answer", "final answer"],
            ["YES", "NO"],
            "Conversational",
        )
    )
    assert chartqapro_match("final answer", target) == 1.0


def test_conversation_uses_final_year_flag_despite_upstream_length_anomaly():
    target = GroundTruth(
        chartqapro_target(
            ["old answer", "2020"],
            ["NO", "NO", "NO", "YES"],
            "Conversational",
        )
    )
    assert chartqapro_match("2020", target) == 1.0
    assert chartqapro_match("2021", target) == 0.0


def test_paper_spec_exact_match_is_frozen_as_sensitivity_metric():
    target = GroundTruth(chartqapro_target(["true"], ["NO"], "Fact Checking"))
    assert chartqapro_match("tru", target) == 0.75
    assert chartqapro_spec_match("tru", target) == 0.0
    assert chartqapro_spec_match("true", target) == 1.0


def test_nonfinite_numeric_like_gold_is_not_scorer_self_consistent():
    target = GroundTruth(chartqapro_target(["nan"], ["NO"], "Factoid"))
    assert chartqapro_match("nan", target) == 0.0
    assert chartqapro_spec_match("nan", target) == 0.0


def test_pilot_image_selection_is_deterministic_and_group_disjoint():
    image_ids = [f"{index:064x}" for index in range(1, 8)] + [f"{1:064x}"]
    first = select_chartqapro_pilot_images(image_ids, count=3)
    second = select_chartqapro_pilot_images(list(reversed(image_ids)), count=3)
    assert first == second
    assert len(first) == len(set(first)) == 3
    assert set(first) < set(image_ids)


def test_chartqapro_manifest_target_round_trip(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"placeholder")
    target = chartqapro_target(["42"], ["NO"], "Factoid")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "state_id": "chartqapro:0",
                "image_id": "a" * 64,
                "source_id": "a" * 64,
                "image_path": "image.png",
                "question": "Question prompt",
                "target": target,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    example = load_manifest(manifest)[0]
    assert scorer_by_name("chartqapro")("42", example.ground_truth) == 1.0
