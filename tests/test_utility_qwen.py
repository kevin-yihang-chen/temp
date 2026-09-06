"""Actual tiny Transformers architecture test, not a frozen-feature mock."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_real_qwen_architecture_single_image_gradients_and_update(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from PIL import Image
    from beyond_entropy.rollout import AgentState
    from beyond_entropy.schema import BBox
    from beyond_entropy.spatial_action_space import SpatialAction, SpatialActionSpace
    from beyond_entropy.utility_dataset import UtilityInputs
    from beyond_entropy.utility_head import utility_sft_loss
    from beyond_entropy.utility_qwen import QwenSpatialUtility

    torch.set_num_threads(2)
    torch.manual_seed(17)
    config = transformers.Qwen2_5_VLConfig(
        text_config={"vocab_size": 100, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
                     "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1]}},
        vision_config={"hidden_size": 32, "out_hidden_size": 32, "intermediate_size": 64,
                       "depth": 2, "num_heads": 4, "fullatt_block_indexes": [1]},
        image_token_id=98, video_token_id=99, vision_start_token_id=97, vision_end_token_id=96,
    )
    config._attn_implementation = "sdpa"
    backbone = transformers.Qwen2_5_VLForConditionalGeneration(config)
    pixels = torch.randn(16, 3*2*14*14)

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert len([v for v in messages[1]["content"] if v["type"] == "image"]) == 1
            assert messages[1]["content"][-1]["text"] == "question?"
            return transformers.BatchFeature({
                "input_ids": torch.tensor([[1, 97, 98, 98, 98, 98, 96, 5, 6]]),
                "attention_mask": torch.ones(1, 9, dtype=torch.long),
                "pixel_values": pixels, "image_grid_thw": torch.tensor([[1, 4, 4]]),
            })

    path = tmp_path / "original.png"
    Image.new("RGB", (56, 56)).save(path)
    space = SpatialActionSpace((SpatialAction(0, "answer-now", None, 0),
                               SpatialAction(1, "a", BBox(0, 0, .5, 1), 1),
                               SpatialAction(2, "b", BBox(.5, 0, 1, 1), 1)))
    inputs = UtilityInputs(AgentState("s", "i", "source", str(path), "question?"), space)
    model = QwenSpatialUtility(backbone, Processor(), head_dim=12).train()
    trainable = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    out = model(inputs)
    loss = utility_sft_loss(out["action_logits"], method="utility", gains=torch.tensor([[0., 1., -1.]]))
    loss.backward()
    assert all(v > 0 for v in model.gradient_report().values())
    optimizer.step()
    assert model.vision_calls == 1
    assert model.last_measurement["candidate_crop_executions"] == 0
    changed = [n for n, p in model.named_parameters() if n in trainable and not torch.equal(trainable[n], p)]
    assert any("visual.merger" in n for n in changed)
    assert any("language_model.layers.1" in n for n in changed)
    assert all(torch.equal(frozen[n], p) for n, p in model.named_parameters() if n in frozen)
    assert set(model.trainable_state_dict()) == set(trainable)
    with pytest.raises(TypeError, match="outcome-free"):
        model({"inputs": inputs, "gain": [0, 1, -1]})
