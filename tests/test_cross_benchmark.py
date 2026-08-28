from __future__ import annotations

import pytest

from beyond_entropy.benchmarks import hrbench_match, scorer_by_name
from beyond_entropy.cross_benchmark import (
    build_docvqa_prompt,
    build_hrbench_prompt,
    build_textvqa_prompt,
    docvqa_anls_match,
    docvqa_target,
    evalai_normalize_answer,
    hrbench_target,
    textvqa_soft_match,
    textvqa_target,
)
from beyond_entropy.rollout import GroundTruth


def test_released_prompt_adapters_do_not_expose_targets():
    doc_prompt = build_docvqa_prompt("What is the invoice total?")
    assert doc_prompt == (
        "What is the invoice total?\n"
        "Answer the question using a single word or phrase."
    )

    text_prompt = build_textvqa_prompt(
        "WHAT Brand is shown?",
        ["ACME", "42"],
    )
    assert text_prompt == (
        "What brand is shown?\n"
        "Reference OCR token: ACME, 42\n"
        "Answer the question using a single word or phrase."
    )

    hr_prompt = build_hrbench_prompt(
        "Which label is smallest?",
        {"B": "large", "A": "small"},
    )
    assert hr_prompt == (
        "Which label is smallest?\n"
        "A. small\n"
        "B. large\n"
        "Answer the option letter directly."
    )
    assert "correct" not in hr_prompt.casefold()


def test_docvqa_anls_matches_pinned_threshold_and_multi_reference_rules():
    target = GroundTruth(docvqa_target(["abcd", "different answer"]))
    assert docvqa_anls_match("abcd", target) == 1.0
    assert docvqa_anls_match("abce", target) == pytest.approx(0.75)
    # Released lmms-eval retains similarity exactly at the 0.5 threshold.
    assert docvqa_anls_match("abxx", target) == pytest.approx(0.5)
    assert docvqa_anls_match("xxxx", target) == 0.0
    assert docvqa_anls_match("DIFFERENT   ANSWER", target) == 1.0


def test_docvqa_anls_retains_released_raw_length_denominator():
    target = GroundTruth(docvqa_target(["a   b"]))
    # Normalized edit distance is one, but the released denominator is five.
    assert docvqa_anls_match("acb", target) == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The TWO, cats!", "2 cats"),
        ("cant", "can't"),
        ("1,000", "1000"),
        ("A red-blue sign", "red blue sign"),
        ("value 1.5.", "value 1.5"),
    ],
)
def test_evalai_answer_normalization(raw: str, expected: str):
    assert evalai_normalize_answer(raw) == expected


@pytest.mark.parametrize(
    ("matches", "expected"),
    [(0, 0.0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0), (10, 1.0)],
)
def test_textvqa_leave_one_out_soft_accuracy(matches: int, expected: float):
    references = ["two"] * matches + ["other"] * (10 - matches)
    target = GroundTruth(textvqa_target(references))
    assert textvqa_soft_match("2", target) == pytest.approx(expected)
    assert scorer_by_name("textvqa")("2", target) == pytest.approx(expected)


def test_textvqa_target_requires_ten_answers_and_is_not_mutated():
    answers = ["yes"] * 10
    payload = textvqa_target(answers)
    target = GroundTruth(payload)
    assert textvqa_soft_match("yes", target) == 1.0
    assert payload == {"answers": ["yes"] * 10}
    with pytest.raises(ValueError, match="exactly ten"):
        textvqa_target(["yes"] * 9)


def test_hrbench_uses_released_letter_extraction_and_structured_target():
    target = GroundTruth(
        hrbench_target("b", category="semantic", cycle_category="text")
    )
    assert hrbench_match("The answer is B.", target) == 1.0
    assert scorer_by_name("hrbench")("Option A", target) == 0.0
    with pytest.raises(ValueError, match="one of"):
        hrbench_target("E", category="semantic", cycle_category="text")


def test_prompt_and_ground_truth_validation_fail_closed():
    with pytest.raises(ValueError, match="non-empty"):
        build_docvqa_prompt(" ")
    with pytest.raises(ValueError, match="at least two"):
        build_hrbench_prompt("question", {"A": "only"})
    with pytest.raises(ValueError, match="answers"):
        docvqa_anls_match("answer", GroundTruth({"wrong": []}))
