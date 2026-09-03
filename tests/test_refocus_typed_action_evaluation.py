from __future__ import annotations

import pytest

from beyond_entropy.refocus_typed_action import RefocusTypedAction
from beyond_entropy.refocus_typed_action_evaluation import (
    analyze_typed_action_response,
)


AVAILABLE = {"x": ("A",), "y": ("B",)}
INTENT = ("focus_on_", "display(")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            "FINAL ANSWER: 7 TERMINATE",
            (False, False, False, False, False),
        ),
        (
            'display(focus_on_x_values_with_draw(image_1, ["A"], columns_bbox))',
            (True, False, False, False, False),
        ),
        (
            "```python\ndisplay(focus_on_x_values_with_draw(\n```",
            (True, True, False, False, False),
        ),
        (
            "```python\n"
            'display(focus_on_x_values_with_draw(image_1, ["UNKNOWN"], '
            "columns_bbox))\n```",
            (True, True, True, True, False),
        ),
        (
            "```python\n"
            'display(focus_on_x_values_with_draw(image_1, ["A"], '
            "columns_bbox))\n```",
            (True, True, True, True, True),
        ),
    ],
)
def test_typed_action_response_layers(
    response: str, expected: tuple[bool, bool, bool, bool, bool]
) -> None:
    result = analyze_typed_action_response(
        response,
        available_labels=AVAILABLE,
        intent_substrings=INTENT,
    )
    assert (
        result.tool_intent,
        result.complete_python_fence,
        result.python_syntax_valid,
        result.argument_contract_valid,
        result.parser_valid,
    ) == expected
    if result.parser_valid:
        assert result.action == RefocusTypedAction(axis="x", mode="draw", labels=("A",))
        assert result.contract_error is None
        assert result.parser_error is None
    else:
        assert result.action is None
        assert result.parser_error is not None


def test_argument_contract_rejects_wrong_bbox_even_with_literal_labels() -> None:
    result = analyze_typed_action_response(
        "```python\n"
        'display(focus_on_x_values_with_draw(image_1, ["A"], rows_bbox))\n'
        "```",
        available_labels=AVAILABLE,
        intent_substrings=INTENT,
    )
    assert result.python_syntax_valid is True
    assert result.argument_contract_valid is False
    assert result.parser_valid is False
    assert result.contract_error == "x-axis focus call must use columns_bbox"


def test_diagnostic_rejects_invalid_intent_configuration() -> None:
    with pytest.raises(ValueError, match="intent substrings"):
        analyze_typed_action_response(
            "FINAL ANSWER: 1 TERMINATE",
            available_labels=AVAILABLE,
            intent_substrings=(),
        )
