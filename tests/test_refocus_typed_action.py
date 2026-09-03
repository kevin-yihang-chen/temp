from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

from beyond_entropy.refocus_g1_dataset import (
    ACTION_SYSTEM_PROMPT_V1,
    ACTION_SYSTEM_PROMPT_V2,
    build_typed_action_prompt,
)
from beyond_entropy.refocus_typed_action import (
    Axis,
    Mode,
    RefocusTypedAction,
    parse_refocus_typed_action,
    render_refocus_typed_action,
)


def _load_runtime_refocus_tools(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("g1_runtime_refocus_tools", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_typed_actions_round_trip_all_axes_and_modes() -> None:
    available = {"x": ("North America", "欧洲"), "y": ("Revenue", "利润")}
    cases: tuple[tuple[Axis, tuple[str, ...]], ...] = (
        ("x", ("North America", "欧洲")),
        ("y", ("利润",)),
    )
    modes: tuple[Mode, ...] = ("draw", "highlight", "mask")
    for axis, labels in cases:
        for mode in modes:
            action = RefocusTypedAction(axis=axis, mode=mode, labels=labels)
            rendered = render_refocus_typed_action(action)
            assert rendered.startswith("```python\ndisplay(focus_on_")
            assert rendered.endswith(")\n```")
            assert (
                parse_refocus_typed_action(rendered, available_labels=available)
                == action
            )


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (
            'display(focus_on_x_values_with_draw(image_1, ["A"], columns_bbox))',
            "complete python fence",
        ),
        (
            '```python\nfocus_on_x_values_with_draw(image_1, ["A"], columns_bbox)\n```',
            "wrap one call in display",
        ),
        (
            '```python\ndisplay(focus_on_x_values_with_draw(image_1, ["A"], '
            "y_values_bbox))\n```",
            "must use columns_bbox",
        ),
        (
            "```python\ndisplay(focus_on_x_values_with_draw(image=image_1, "
            'x_values_to_focus_on=["A"], all_x_values_bounding_boxes=columns_bbox))\n```',
            "three positional arguments",
        ),
        (
            '```python\ndisplay(focus_on_x_values_with_draw(image_1, ["missing"], '
            "columns_bbox))\n```",
            "unavailable",
        ),
        (
            '```python\ndisplay(focus_on_x_values_with_draw(image_1, ["A"], '
            'columns_bbox))\nprint("extra")\n```',
            "one expression",
        ),
        (
            '```python\ndisplay(focus_on_x_values_with_draw(image_1, ["A"], '
            "columns_bbox))\n```\nFINAL ANSWER: 1 TERMINATE",
            "complete python fence",
        ),
    ],
)
def test_typed_action_parser_rejects_noncanonical_responses(
    response: str, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        parse_refocus_typed_action(
            response,
            available_labels={"x": ("A",), "y": ("B",)},
        )


def test_typed_action_value_object_rejects_empty_or_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RefocusTypedAction(axis="x", mode="draw", labels=())
    with pytest.raises(ValueError, match="unique"):
        RefocusTypedAction(axis="y", mode="mask", labels=("A", "A"))


def test_v2_prompt_documents_exact_api_without_changing_frozen_v1() -> None:
    assert (
        hashlib.sha256(ACTION_SYSTEM_PROMPT_V1.encode("utf-8")).hexdigest()
        == "d8e1b93a3635901c6a5afcbf618e255e4923b01b11001ea56ca31de9fefca24f"
    )
    assert "focus_on_x_values_with_MODE(image_1" in ACTION_SYSTEM_PROMPT_V2
    assert "focus_on_y_values_with_MODE(image_1" in ACTION_SYSTEM_PROMPT_V2
    assert "columns_bbox" in ACTION_SYSTEM_PROMPT_V2
    assert "rows_bbox" in ACTION_SYSTEM_PROMPT_V2
    assert "Do not use keyword arguments" in ACTION_SYSTEM_PROMPT_V2
    prompt = build_typed_action_prompt(
        question="Which region is largest?",
        x_values=("North America",),
        y_values=(),
    )
    assert prompt[0] == {"role": "system", "content": ACTION_SYSTEM_PROMPT_V2}
    assert 'Available x-axis labels: ["North America"]' in prompt[1]["content"]
    assert "Available y-axis labels: []" in prompt[1]["content"]


def test_rendered_typed_actions_execute_in_pinned_runtime_context() -> None:
    runtime_path = Path(
        "/userhome/cs3/yihangc/Documents/runtime/"
        "vtool-action-credit-g1/recipe/vtool/refocus_tools.py"
    )
    if not runtime_path.is_file():
        pytest.skip("pinned VTool G1 runtime is unavailable")
    image_module = pytest.importorskip("PIL.Image")
    runtime = _load_runtime_refocus_tools(runtime_path)

    cases = (
        (
            RefocusTypedAction(axis="x", mode="draw", labels=("A",)),
            "chartqa_v_bar",
            {
                "x_values_bbox": {"A": {"x1": 1, "y1": 1, "x2": 8, "y2": 8}},
                "y_values_bbox": {},
            },
        ),
        (
            RefocusTypedAction(axis="y", mode="highlight", labels=("B",)),
            "chartqa_h_bar",
            {
                "x_values_bbox": {},
                "y_values_bbox": {"B": {"x1": 2, "y1": 2, "x2": 9, "y2": 9}},
            },
        ),
    )
    for action, source, bbox_metadata in cases:
        image = image_module.new("RGB", (16, 16), color="white")
        displayed: list[object] = []
        context = runtime.build_refocus_context(
            display_callback=displayed.append,
            image=image,
            metadata={"source": source, **bbox_metadata},
        )
        response = render_refocus_typed_action(action)
        assert (
            parse_refocus_typed_action(
                response,
                available_labels={"x": ("A",), "y": ("B",)},
            )
            == action
        )
        code = response.removeprefix("```python\n").removesuffix("\n```")
        exec(compile(code, "typed_refocus_smoke.py", "exec"), context)
        assert len(displayed) == 1
        assert isinstance(displayed[0], image_module.Image)
