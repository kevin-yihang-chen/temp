from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import re
from typing import Literal, Mapping, Sequence, cast


Axis = Literal["x", "y"]
Mode = Literal["draw", "highlight", "mask"]
ALLOWED_AXES = frozenset({"x", "y"})
ALLOWED_MODES = frozenset({"draw", "highlight", "mask"})
ALLOWED_FOCUS_FUNCTIONS = frozenset(
    f"focus_on_{axis}_values_with_{mode}"
    for axis in ALLOWED_AXES
    for mode in ALLOWED_MODES
)
FENCED_PYTHON = re.compile(r"\A\s*```python\s*(.*?)```\s*\Z", re.DOTALL)


@dataclass(frozen=True)
class RefocusTypedAction:
    axis: Axis
    mode: Mode
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.axis not in ALLOWED_AXES:
            raise ValueError(f"unsupported axis: {self.axis!r}")
        if self.mode not in ALLOWED_MODES:
            raise ValueError(f"unsupported mode: {self.mode!r}")
        if not self.labels:
            raise ValueError("typed refocus action requires at least one label")
        if any(not isinstance(label, str) or not label for label in self.labels):
            raise ValueError("typed refocus labels must be non-empty strings")
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("typed refocus labels must be unique")

    @property
    def function_name(self) -> str:
        return f"focus_on_{self.axis}_values_with_{self.mode}"

    @property
    def bbox_name(self) -> str:
        # The pinned VTool runtime exposes legacy aliases, not the metadata keys
        # x_values_bbox/y_values_bbox, in the executable context.
        return "columns_bbox" if self.axis == "x" else "rows_bbox"


def render_refocus_typed_action(action: RefocusTypedAction) -> str:
    labels = json.dumps(list(action.labels), ensure_ascii=False)
    code = f"display({action.function_name}(image_1, {labels}, " f"{action.bbox_name}))"
    return f"```python\n{code}\n```"


def _name(node: ast.AST, *, context: str) -> str:
    if not isinstance(node, ast.Name):
        raise ValueError(f"{context} must be a simple name")
    return node.id


def _literal_labels(node: ast.AST) -> tuple[str, ...]:
    if not isinstance(node, ast.List) or not node.elts:
        raise ValueError("labels must be a non-empty list literal")
    labels: list[str] = []
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            raise ValueError("labels must contain only string literals")
        labels.append(element.value)
    return tuple(labels)


def _parse_function_name(function_name: str) -> tuple[Axis, Mode]:
    match = re.fullmatch(
        r"focus_on_([xy])_values_with_(draw|highlight|mask)", function_name
    )
    if match is None or function_name not in ALLOWED_FOCUS_FUNCTIONS:
        raise ValueError("call must use exactly one allowed axis-focus function")
    return cast(Axis, match.group(1)), cast(Mode, match.group(2))


def parse_refocus_typed_action(
    response: str, *, available_labels: Mapping[str, Sequence[str]]
) -> RefocusTypedAction:
    if not isinstance(response, str):
        raise ValueError("typed refocus response must be text")
    fenced = FENCED_PYTHON.fullmatch(response.strip())
    if fenced is None:
        raise ValueError("typed refocus action must be one complete python fence")
    try:
        tree = ast.parse(fenced.group(1).strip(), mode="exec")
    except SyntaxError as exc:
        raise ValueError("typed refocus action is not valid Python") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise ValueError("typed refocus action must contain one expression")
    display_call = tree.body[0].value
    if (
        not isinstance(display_call, ast.Call)
        or _name(display_call.func, context="outer function") != "display"
        or len(display_call.args) != 1
        or display_call.keywords
    ):
        raise ValueError("typed refocus action must wrap one call in display")
    focus_call = display_call.args[0]
    if not isinstance(focus_call, ast.Call):
        raise ValueError("display argument must be an allowed focus call")
    function_name = _name(focus_call.func, context="focus function")
    axis, mode = _parse_function_name(function_name)
    if len(focus_call.args) != 3 or focus_call.keywords:
        raise ValueError("focus call must use exactly three positional arguments")
    if _name(focus_call.args[0], context="image argument") != "image_1":
        raise ValueError("focus call image argument must be image_1")
    labels = _literal_labels(focus_call.args[1])
    expected_bbox = "columns_bbox" if axis == "x" else "rows_bbox"
    if _name(focus_call.args[2], context="bbox argument") != expected_bbox:
        raise ValueError(f"{axis}-axis focus call must use {expected_bbox}")

    if axis not in available_labels:
        raise ValueError(f"available labels are missing the {axis!r} axis")
    axis_labels = available_labels[axis]
    if isinstance(axis_labels, (str, bytes)) or not isinstance(axis_labels, Sequence):
        raise ValueError(f"available {axis}-axis labels must be a sequence")
    normalized_available = tuple(axis_labels)
    if not all(isinstance(label, str) for label in normalized_available):
        raise ValueError(f"available {axis}-axis labels must be strings")
    if len(normalized_available) != len(set(normalized_available)):
        raise ValueError(f"available {axis}-axis labels must be unique")
    unavailable = [label for label in labels if label not in normalized_available]
    if unavailable:
        raise ValueError(
            f"focus labels are unavailable on the {axis}-axis: {unavailable}"
        )
    return RefocusTypedAction(axis=axis, mode=mode, labels=labels)
