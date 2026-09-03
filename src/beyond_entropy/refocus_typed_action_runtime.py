from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping

from PIL import Image

from beyond_entropy.refocus_typed_action import (
    RefocusTypedAction,
    parse_refocus_typed_action,
    render_refocus_typed_action,
)


def tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def image_sha256(image: Image.Image) -> str:
    normalized = image.convert("RGBA")
    digest = hashlib.sha256()
    digest.update(str(normalized.size).encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def load_refocus_runtime(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "typed_action_generation_runtime_refocus_tools", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned runtime module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build_refocus_context", None)):
        raise ValueError("pinned runtime does not expose build_refocus_context")
    return module


def execute_renderer_owned_action(
    *,
    runtime: ModuleType,
    image: Image.Image,
    metadata: Mapping[str, Any],
    action: RefocusTypedAction,
) -> tuple[str, Image.Image, bool]:
    """Execute only a canonical re-rendering of an already structured action."""

    available_labels = {
        "x": tuple(metadata.get("x_values", ())),
        "y": tuple(metadata.get("y_values", ())),
    }
    response = render_refocus_typed_action(action)
    parsed = parse_refocus_typed_action(
        response,
        available_labels=available_labels,
    )
    if parsed != action:
        raise AssertionError("typed action renderer/parser round-trip mismatch")
    trusted_response = render_refocus_typed_action(parsed)
    if trusted_response != response:
        raise AssertionError("typed action canonicalization is not stable")
    code = trusted_response.removeprefix("```python\n").removesuffix("\n```")
    displayed: list[Image.Image] = []
    execution_image = image.convert("RGB").copy()
    original_sha256 = image_sha256(execution_image)
    context = runtime.build_refocus_context(
        display_callback=displayed.append,
        image=execution_image,
        metadata=dict(metadata),
    )
    compile_context = {"__builtins__": {}, **context}
    exec(
        compile(code, "typed_action_generation_renderer_owned.py", "exec"),
        compile_context,
    )
    if len(displayed) != 1 or not isinstance(displayed[0], Image.Image):
        raise RuntimeError("typed action did not display exactly one PIL image")
    output = displayed[0]
    return response, output, image_sha256(output) != original_sha256
