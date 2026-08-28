from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .rollout import GroundTruth


UG_REFERENCE_COMMIT = "13050ee49865e4330519108f42d1ccfccff1aee1"
_SHORT_ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."


def _nonempty_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _answer_list(target: object, *, benchmark: str) -> tuple[str, ...]:
    raw_answers: object
    if isinstance(target, Mapping):
        if "answers" not in target:
            raise ValueError(f"{benchmark} ground truth must contain answers")
        raw_answers = target["answers"]
    else:
        raw_answers = target
    if isinstance(raw_answers, (str, bytes)) or not isinstance(raw_answers, Sequence):
        raise ValueError(f"{benchmark} answers must be a sequence of strings")
    answers = tuple(str(answer) for answer in raw_answers)
    if not answers:
        raise ValueError(f"{benchmark} answers must be non-empty")
    return answers


def docvqa_target(answers: Sequence[object]) -> dict[str, list[str]]:
    """Build the evaluation-only payload used by the pinned DocVQA scorer."""

    normalized = _answer_list(list(answers), benchmark="DocVQA")
    return {"answers": list(normalized)}


def textvqa_target(answers: Sequence[object]) -> dict[str, list[str]]:
    """Build a ten-annotator TextVQA evaluation payload."""

    normalized = _answer_list(list(answers), benchmark="TextVQA")
    if len(normalized) != 10:
        raise ValueError("TextVQA requires exactly ten reference answers")
    return {"answers": list(normalized)}


def hrbench_target(
    answer: object,
    *,
    category: object,
    cycle_category: object,
) -> dict[str, str]:
    normalized_answer = _nonempty_text(answer, name="HRBench answer").upper()
    if normalized_answer not in {"A", "B", "C", "D"}:
        raise ValueError("HRBench answer must be one of A, B, C, or D")
    return {
        "answer": normalized_answer,
        "category": _nonempty_text(category, name="HRBench category"),
        "cycle_category": _nonempty_text(
            cycle_category,
            name="HRBench cycle_category",
        ),
    }


def build_docvqa_prompt(question: object) -> str:
    return _nonempty_text(question, name="DocVQA question") + _SHORT_ANSWER_SUFFIX


def build_textvqa_prompt(
    question: object,
    ocr_tokens: Sequence[object],
    *,
    include_ocr: bool = True,
) -> str:
    """Reproduce the pinned lmms-eval default TextVQA prompt."""

    prompt = _nonempty_text(question, name="TextVQA question").capitalize()
    if include_ocr:
        prompt += "\nReference OCR token: " + ", ".join(
            str(token) for token in ocr_tokens
        )
    return prompt + _SHORT_ANSWER_SUFFIX


def _normalized_hrbench_options(
    options: Mapping[str, object],
) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for raw_key, raw_value in options.items():
        key = str(raw_key).strip().upper()
        if key not in {"A", "B", "C", "D"}:
            raise ValueError("HRBench option keys must be A, B, C, or D")
        normalized.append((key, _nonempty_text(raw_value, name=f"option {key}")))
    normalized.sort()
    if len(normalized) < 2 or len({key for key, _ in normalized}) != len(normalized):
        raise ValueError("HRBench requires at least two distinct options")
    return normalized


def build_hrbench_context(
    question: object,
    options: Mapping[str, object],
) -> str:
    """Build gate-visible HRBench content without answer-format boilerplate."""

    normalized = _normalized_hrbench_options(options)
    option_text = "".join(f"{key}. {value}\n" for key, value in normalized)
    return f"{_nonempty_text(question, name='HRBench question')}\n{option_text}".rstrip()


def build_hrbench_prompt(
    question: object,
    options: Mapping[str, object],
) -> str:
    """Reproduce the released HRBench direct-option prompt."""

    return build_hrbench_context(question, options) + "\nAnswer the option letter directly."


def _levenshtein_distance(first: str, second: str) -> int:
    if len(first) > len(second):
        first, second = second, first
    distances: Sequence[int] = range(len(first) + 1)
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


def docvqa_anls_match(answer: str, ground_truth: GroundTruth) -> float:
    """Match lmms-eval's pinned DocVQA ANLS behavior for one prediction."""

    references = _answer_list(ground_truth.target, benchmark="DocVQA")
    prediction = str(answer)
    normalized_prediction = " ".join(prediction.strip().lower().split())
    distances: list[float] = []
    for reference in references:
        normalized_reference = " ".join(reference.strip().lower().split())
        distance = _levenshtein_distance(
            normalized_reference,
            normalized_prediction,
        )
        # Deliberately retain the released implementation's raw-string
        # denominator, even though edit distance uses normalized strings.
        length = max(len(reference.upper()), len(prediction.upper()))
        distances.append(0.0 if length == 0 else distance / length)
    similarity = 1.0 - min(distances)
    return 0.0 if similarity < 0.5 else similarity


_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = frozenset({"a", "an", "the"})
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
_PUNCTUATION = (
    ";",
    "/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
)


def evalai_normalize_answer(answer: object) -> str:
    """Dependency-free parity implementation of EvalAIAnswerProcessor."""

    text = str(answer).lower()
    text = text.replace(",", "").replace("?", "").replace("'s", " 's")
    text = text.strip().replace("\n", " ").replace("\t", " ").strip()
    punctuated = text
    for punctuation in _PUNCTUATION:
        if (
            punctuation + " " in text
            or " " + punctuation in text
            or _COMMA_STRIP.search(text) is not None
        ):
            punctuated = punctuated.replace(punctuation, "")
        else:
            punctuated = punctuated.replace(punctuation, " ")
    punctuated = _PERIOD_STRIP.sub("", punctuated)
    words = []
    for word in punctuated.lower().split():
        normalized = _NUMBER_MAP.get(word, word)
        if normalized not in _ARTICLES:
            words.append(_CONTRACTIONS.get(normalized, normalized))
    return " ".join(words)


def textvqa_soft_match(answer: str, ground_truth: GroundTruth) -> float:
    """Match the pinned leave-one-annotator-out TextVQA soft accuracy."""

    references = _answer_list(ground_truth.target, benchmark="TextVQA")
    if len(references) != 10:
        raise ValueError("TextVQA requires exactly ten reference answers")
    prediction = evalai_normalize_answer(answer)
    normalized_references = tuple(evalai_normalize_answer(item) for item in references)
    per_annotator: list[float] = []
    for held_out in range(len(normalized_references)):
        matches = sum(
            reference == prediction
            for index, reference in enumerate(normalized_references)
            if index != held_out
        )
        per_annotator.append(min(1.0, matches / 3.0))
    return sum(per_annotator) / len(per_annotator)


def hrbench_answer(ground_truth: GroundTruth) -> str:
    target: Any = ground_truth.target
    raw_answer = target.get("answer") if isinstance(target, Mapping) else target
    answer = str(raw_answer).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError("HRBench ground truth answer must be A, B, C, or D")
    return answer
