from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping, Sequence

from beyond_entropy.refocus_typed_action import (
    FENCED_PYTHON,
    RefocusTypedAction,
    parse_refocus_typed_action,
)


@dataclass(frozen=True)
class TypedActionResponseDiagnostics:
    tool_intent: bool
    complete_python_fence: bool
    python_syntax_valid: bool
    argument_contract_valid: bool
    parser_valid: bool
    contract_error: str | None
    parser_error: str | None
    action: RefocusTypedAction | None


def _unique_text(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def analyze_typed_action_response(
    response: str,
    *,
    available_labels: Mapping[str, Sequence[str]],
    intent_substrings: Sequence[str],
) -> TypedActionResponseDiagnostics:
    """Measure nested formatting layers without executing model text.

    ``argument_contract_valid`` asks whether the exact fenced API is valid while
    deliberately ignoring row-specific label membership. ``parser_valid`` then
    applies the real label lists. This isolates malformed arguments from a
    syntactically valid call that selected a nonexistent label.
    """

    if not isinstance(response, str):
        raise TypeError("typed-action response must be text")
    if (
        isinstance(intent_substrings, (str, bytes))
        or not intent_substrings
        or any(not isinstance(value, str) or not value for value in intent_substrings)
    ):
        raise ValueError("intent substrings must be a non-empty text sequence")
    tool_intent = any(value in response for value in intent_substrings)
    fenced = FENCED_PYTHON.fullmatch(response.strip())
    if fenced is None:
        return TypedActionResponseDiagnostics(
            tool_intent=tool_intent,
            complete_python_fence=False,
            python_syntax_valid=False,
            argument_contract_valid=False,
            parser_valid=False,
            contract_error="typed refocus action must be one complete python fence",
            parser_error="typed refocus action must be one complete python fence",
            action=None,
        )
    try:
        tree = ast.parse(fenced.group(1).strip(), mode="exec")
    except SyntaxError:
        return TypedActionResponseDiagnostics(
            tool_intent=tool_intent,
            complete_python_fence=True,
            python_syntax_valid=False,
            argument_contract_valid=False,
            parser_valid=False,
            contract_error="typed refocus action is not valid Python",
            parser_error="typed refocus action is not valid Python",
            action=None,
        )

    literal_strings = tuple(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    permissive_labels = {
        axis: _unique_text([*available_labels.get(axis, ()), *literal_strings])
        for axis in ("x", "y")
    }
    try:
        parse_refocus_typed_action(
            response,
            available_labels=permissive_labels,
        )
    except ValueError as exc:
        error = str(exc)
        return TypedActionResponseDiagnostics(
            tool_intent=tool_intent,
            complete_python_fence=True,
            python_syntax_valid=True,
            argument_contract_valid=False,
            parser_valid=False,
            contract_error=error,
            parser_error=error,
            action=None,
        )

    try:
        action = parse_refocus_typed_action(
            response,
            available_labels=available_labels,
        )
    except ValueError as exc:
        return TypedActionResponseDiagnostics(
            tool_intent=tool_intent,
            complete_python_fence=True,
            python_syntax_valid=True,
            argument_contract_valid=True,
            parser_valid=False,
            contract_error=None,
            parser_error=str(exc),
            action=None,
        )
    return TypedActionResponseDiagnostics(
        tool_intent=tool_intent,
        complete_python_fence=True,
        python_syntax_valid=True,
        argument_contract_valid=True,
        parser_valid=True,
        contract_error=None,
        parser_error=None,
        action=action,
    )
