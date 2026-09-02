from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from beyond_entropy.infographicvqa_literature_attention_extraction import (  # noqa: E402
    literature_messages,
    literature_prefill_texts,
    visual_attention_grids,
)
from beyond_entropy.infographicvqa_literature_attention_where import (  # noqa: E402
    VICROP_ANSWER_SUFFIX,
    VICROP_GENERIC_QUESTION,
)


def test_literature_prefills_are_exact_and_no_query_has_no_text() -> None:
    query, generic, no_query = literature_prefill_texts("Where is 42?")
    assert query == "Where is 42?" + VICROP_ANSWER_SUFFIX
    assert generic == VICROP_GENERIC_QUESTION + VICROP_ANSWER_SUFFIX
    assert no_query is None
    messages = literature_messages(
        image_path=Path("image.png"),
        text=no_query,
        min_pixels=1,
        max_pixels=2,
    )
    assert messages[0] == {
        "role": "system",
        "content": "You are a helpful assistant.",
    }
    assert [item["type"] for item in messages[1]["content"]] == ["image"]


def test_visual_attention_grids_preserve_layer_head_values() -> None:
    input_ids = torch.tensor([7, 99, 99, 99, 99, 8])
    layers = []
    for layer_index in range(2):
        value = torch.zeros(1, 3, 6, 6)
        for head_index in range(3):
            value[0, head_index, -1, 1:5] = (
                torch.tensor([1.0, 2.0, 3.0, 4.0]) + 10 * layer_index + head_index
            )
        layers.append(value)
    grids = visual_attention_grids(
        layers,
        input_ids,
        torch.tensor([1, 4, 4]),
        image_token_id=99,
        spatial_merge_size=2,
    )
    assert grids.shape == (2, 3, 2, 2)
    assert grids[0, 0].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert grids[1, 2].tolist() == [[13.0, 14.0], [15.0, 16.0]]


def test_visual_attention_grids_reject_missing_image_tokens() -> None:
    with pytest.raises(ValueError, match="no image tokens"):
        visual_attention_grids(
            [torch.zeros(1, 1, 2, 2)],
            torch.tensor([1, 2]),
            torch.tensor([1, 2, 2]),
            image_token_id=99,
            spatial_merge_size=1,
        )
