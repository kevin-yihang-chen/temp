from __future__ import annotations

import ast
import hashlib
import re
import textwrap
from typing import Any, Mapping, Sequence

from .rollout import GroundTruth


CHARTQAPRO_QUESTION_TYPES = frozenset(
    {
        "Factoid",
        "Multi Choice",
        "Conversational",
        "Fact Checking",
        "Hypothetical",
    }
)
CHARTQAPRO_PROMPT_ADAPTER = "vlmevalkit-direct-compatible-v1"
CHARTQAPRO_PILOT_NAMESPACE = "chartqapro-gate3-pilot-v1"


def _string_sequence(value: Sequence[str], *, name: str) -> list[str]:
    result = [str(item) for item in value]
    if not result or any(not item.strip() for item in result):
        raise ValueError(f"{name} must be a non-empty sequence of non-empty strings")
    return result


def chartqapro_final_question(
    questions: Sequence[str],
    question_type: str,
) -> str:
    normalized = _string_sequence(questions, name="questions")
    if question_type not in CHARTQAPRO_QUESTION_TYPES:
        raise ValueError(f"unsupported ChartQAPro question type: {question_type!r}")
    if question_type == "Conversational":
        if len(normalized) < 2:
            raise ValueError("conversational rows must contain at least two turns")
        return normalized[-1]
    if len(normalized) != 1:
        raise ValueError("non-conversational rows must contain exactly one question")
    return normalized[0]


def _direct_question_context(
    questions: Sequence[str],
    answers: Sequence[str],
    question_type: str,
) -> str:
    """Reproduce the pinned VLMEvalKit ChartQAPro Direct prompt semantics."""

    question_values = _string_sequence(questions, name="questions")
    answer_values = _string_sequence(answers, name="answers")
    final_question = chartqapro_final_question(question_values, question_type)
    if len(question_values) != len(answer_values):
        raise ValueError("questions and answers must contain the same number of turns")

    common_tail = (
        "If there are multiple answers, put them in brackets using this format "
        "[’Answer1’, ’Answer2’]. "
        "Remember to generate the final answer only without any additional text!"
    )
    if question_type == "Factoid":
        body = f"""
        You are given a factoid question that you need to answer based on the provided image.
        Your answer should be a single word, number, or phrase. If the question is unanswerable based on
        the information in the provided image, your answer should be unanswerable. Do not generate units.
        But if numerical units such as million, m, billion, B, or K are required, use the exact notation
        shown in the chart.
        {common_tail}
        Question: {final_question}
        """
    elif question_type == "Multi Choice":
        body = f"""
        You are given a question along with different possible answers. You need to select the correct answer
        from them based on the provided image.
        Your answer should be one of the options letters only: a, b, c or d (just the letter itself without any
        additional text). If the question is unanswerable based on the information in the provided image, your
        answer should be unanswerable.
        {common_tail}
        Question: {final_question}
        """
    elif question_type == "Conversational":
        history = [
            item
            for question, answer in zip(question_values[:-1], answer_values[:-1])
            for item in (question, answer)
        ]
        body = f"""
        You are given a multi-turn conversation, and your job is to answer the final question based on the
        conversation history and the information in the provided image.
        Your answer should be a single word, number, or phrase. If the question is unanswerable based on
        the information in the provided image, your answer should be unanswerable. Do not generate units.
        But if numerical units such as million, m, billion, B, or K are required, use the exact notation
        shown in the chart.
        {common_tail}
        Conversation: {history} Question: {final_question}
        """
    elif question_type == "Fact Checking":
        body = f"""
        You are given a fact statement that you need to assess based on the provided image.
        Your answer should be either true or false (without any additional text). If the question is
        unanswerable based on the information in the provided image, your answer should be unanswerable.
        {common_tail}
        Question: {final_question}
        """
    elif question_type == "Hypothetical":
        body = f"""
        You are given a hypothetical question that you need to answer based on the provided image.
        Your answer should be a single word, number, or phrase. If the question is unanswerable based on
        the information in the provided image, your answer should be unanswerable. Do not generate units.
        But if numerical units such as million, m, billion, B, or K are required, use the exact notation
        shown in the chart.
        {common_tail}
        Question: {final_question}
        """
    else:  # pragma: no cover - validated by chartqapro_final_question
        raise AssertionError(question_type)
    return textwrap.dedent(body).strip()


def build_chartqapro_direct_prompt(
    questions: Sequence[str],
    answers: Sequence[str],
    question_type: str,
    paragraph: str | None,
) -> str:
    """Build one backend text turn without exposing the final answer.

    VLMEvalKit supplies the paragraph and the category prompt as two adjacent
    text messages. The project backend accepts one text turn, so the two are
    concatenated with a blank line while retaining their order and content.
    """

    context = _direct_question_context(questions, answers, question_type)
    paragraph_text = "" if paragraph is None else str(paragraph).strip()
    return f"{paragraph_text}\n\n{context}" if paragraph_text else context


def build_chartqapro_gate_context(
    questions: Sequence[str],
    answers: Sequence[str],
    question_type: str,
    paragraph: str | None,
) -> str:
    """Expose task context to the gate without prompt-template boilerplate."""

    question_values = _string_sequence(questions, name="questions")
    answer_values = _string_sequence(answers, name="answers")
    final_question = chartqapro_final_question(question_values, question_type)
    if len(question_values) != len(answer_values):
        raise ValueError("questions and answers must contain the same number of turns")
    parts: list[str] = []
    paragraph_text = "" if paragraph is None else str(paragraph).strip()
    if paragraph_text:
        parts.append(f"Context:\n{paragraph_text}")
    if question_type == "Conversational":
        history: list[str] = []
        for question, answer in zip(question_values[:-1], answer_values[:-1]):
            history.extend((f"Question: {question}", f"Answer: {answer}"))
        parts.append("Conversation history:\n" + "\n".join(history))
    parts.append(f"Question: {final_question}")
    return "\n\n".join(parts)


def chartqapro_target(
    answers: Sequence[str],
    year_flags: Sequence[str],
    question_type: str,
) -> dict[str, Any]:
    answer_values = _string_sequence(answers, name="answers")
    flag_values = _string_sequence(year_flags, name="year_flags")
    if question_type not in CHARTQAPRO_QUESTION_TYPES:
        raise ValueError(f"unsupported ChartQAPro question type: {question_type!r}")
    if any(flag.upper() not in {"YES", "NO"} for flag in flag_values):
        raise ValueError("year flags must contain only YES or NO")
    return {
        "answers": answer_values,
        "question_type": question_type,
        "year_flags": flag_values,
    }


def select_chartqapro_pilot_images(
    image_ids: Sequence[str],
    *,
    count: int = 200,
) -> list[str]:
    unique_ids = set(image_ids)
    if count <= 0:
        raise ValueError("pilot image count must be positive")
    if count >= len(unique_ids):
        raise ValueError("pilot image count must leave at least one formal image")
    for image_id in unique_ids:
        if len(image_id) != 64 or any(
            character not in "0123456789abcdef" for character in image_id
        ):
            raise ValueError("image IDs must be lowercase SHA-256 digests")

    def rank(image_id: str) -> str:
        payload = f"{CHARTQAPRO_PILOT_NAMESPACE}\0{image_id}".encode()
        return hashlib.sha256(payload).hexdigest()

    return sorted(unique_ids, key=lambda image_id: (rank(image_id), image_id))[:count]


def _levenshtein_distance(first: str, second: str) -> int:
    if len(first) > len(second):
        first, second = second, first
    distances = list(range(len(first) + 1))
    for second_index, second_character in enumerate(second):
        updated = [second_index + 1]
        for first_index, first_character in enumerate(first):
            if first_character == second_character:
                updated.append(distances[first_index])
            else:
                updated.append(
                    1
                    + min(
                        distances[first_index],
                        distances[first_index + 1],
                        updated[-1],
                    )
                )
        distances = updated
    return distances[-1]


def _anls_score(prediction: str, gold_label: str, threshold: float = 0.5) -> float:
    normalized_prediction = " ".join(prediction.strip().lower().split())
    normalized_gold = " ".join(gold_label.strip().lower().split())
    length = max(len(normalized_prediction.upper()), len(normalized_gold.upper()))
    normalized_distance = (
        0.0
        if length == 0
        else _levenshtein_distance(normalized_prediction, normalized_gold) / length
    )
    return 1.0 - normalized_distance if normalized_distance < threshold else 0.0


def _fix_list_format(item: str) -> Any:
    match = re.match(r"^\[(.*)\]$", item.strip())
    if not match:
        return item
    content = match.group(1)
    corrected = re.sub(r"(?<!['\w])(\w[^,]*?)(?!['\w])", r"'\1'", content)
    try:
        return ast.literal_eval(f"[{corrected}]")
    except (SyntaxError, ValueError):
        return item


def _parse_to_list(text: str) -> list[str] | None:
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, list):
        return [str(item).strip(" '") for item in parsed]
    return None


def _to_float(text: str) -> float | None:
    try:
        return float(text.strip().strip("%"))
    except ValueError:
        return None


def _evaluate_single_answer(
    target: str,
    prediction: str,
    *,
    max_relative_change: float = 0.05,
) -> float:
    normalized_target = target.strip().strip("%").strip()
    normalized_prediction = prediction.strip().strip("%").strip()
    target_float = _to_float(normalized_target)
    prediction_float = _to_float(normalized_prediction)
    if target_float is not None and prediction_float is not None:
        if target_float == 0.0:
            return float(prediction_float == 0.0)
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return float(relative_change <= max_relative_change)
    return _anls_score(normalized_prediction.lower(), normalized_target.lower())


def _relaxed_correctness(
    target: str,
    prediction: str,
    *,
    year_flags: Sequence[str],
    max_relative_change: float = 0.05,
    always_use_exact_match: bool = False,
) -> float:
    fixed_target = _fix_list_format(target)
    target_list = _parse_to_list(str(fixed_target)) or [str(target)]
    prediction_list = _parse_to_list(str(prediction)) or [str(prediction)]
    flags = [str(flag) for flag in year_flags]
    if len(flags) < len(target_list):
        flags = flags * len(target_list)
    if not flags:
        raise ValueError("year_flags must not be empty")
    scores: list[float] = []
    for index in range(max(len(target_list), len(prediction_list))):
        if index >= len(target_list) or index >= len(prediction_list):
            scores.append(0.0)
            continue
        target_item = target_list[index]
        prediction_item = prediction_list[index]
        if index >= len(flags):
            raise ValueError("year_flags do not cover every target item")
        if flags[index].upper() == "YES" or always_use_exact_match:
            scores.append(
                float(target_item.strip().lower() == prediction_item.strip().lower())
            )
        else:
            scores.append(
                _evaluate_single_answer(
                    target_item,
                    prediction_item,
                    max_relative_change=max_relative_change,
                )
            )
    return sum(scores) / len(scores) if scores else 0.0


def chartqapro_match(answer: str, ground_truth: GroundTruth) -> float:
    """Match the released ChartQAPro/VLMEvalKit scorer for one row."""

    target = ground_truth.target
    if not isinstance(target, Mapping):
        raise ValueError("ChartQAPro ground truth must be a mapping")
    answers = _string_sequence(target.get("answers", ()), name="target answers")
    flags = _string_sequence(target.get("year_flags", ()), name="target year_flags")
    question_type = str(target.get("question_type", ""))
    if question_type not in CHARTQAPRO_QUESTION_TYPES:
        raise ValueError(f"unsupported ChartQAPro question type: {question_type!r}")
    if question_type == "Conversational":
        flags = flags[-1:]
    target_answer = answers[-1].strip(".").strip("\n")
    prediction = str(answer).strip(".").strip("\n")

    # The pinned released scorer computes this category flag but does not pass
    # it into relaxed_correctness. We preserve that behavior for exact parity;
    # the manifest's year flags still enforce exact matching where released.
    _released_exact_match_flag = question_type in {"Fact Checking", "Multi Choice"}
    del _released_exact_match_flag
    return _relaxed_correctness(
        target_answer,
        prediction,
        year_flags=flags,
    )


def chartqapro_spec_match(answer: str, ground_truth: GroundTruth) -> float:
    """Apply the paper-specified exact-match rule as a frozen sensitivity metric."""

    target = ground_truth.target
    if not isinstance(target, Mapping):
        raise ValueError("ChartQAPro ground truth must be a mapping")
    answers = _string_sequence(target.get("answers", ()), name="target answers")
    flags = _string_sequence(target.get("year_flags", ()), name="target year_flags")
    question_type = str(target.get("question_type", ""))
    if question_type not in CHARTQAPRO_QUESTION_TYPES:
        raise ValueError(f"unsupported ChartQAPro question type: {question_type!r}")
    if question_type == "Conversational":
        flags = flags[-1:]
    return _relaxed_correctness(
        answers[-1].strip(".").strip("\n"),
        str(answer).strip(".").strip("\n"),
        year_flags=flags,
        always_use_exact_match=question_type in {"Fact Checking", "Multi Choice"},
    )
