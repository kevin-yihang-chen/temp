from dataclasses import replace
import json

import pytest

from beyond_entropy.decoding_sensitivity import (
    capped_state_ids,
    export_capped_manifest,
    generated_token_count,
)
from beyond_entropy.simulate import simulate_counterfactual_dataset


def test_capped_states_use_the_executed_action_backend_metadata():
    records = simulate_counterfactual_dataset(
        n_states=2,
        num_candidates=2,
        questions_per_image=1,
        seed=2,
    )
    decorated = []
    for index, record in enumerate(records):
        backend_name = "baseline_backend" if record.action_type == "ANSWER" else "action_backend"
        decorated.append(
            replace(
                record,
                metadata={backend_name: {"generated_tokens": 16 if index == 2 else 3}},
            )
        )
    assert generated_token_count(decorated[0]) == 3
    assert capped_state_ids(decorated, token_cap=16) == {decorated[2].state_id}


def test_generated_token_count_rejects_missing_metadata():
    record = simulate_counterfactual_dataset(n_states=2, seed=3)[0]
    with pytest.raises(ValueError, match="missing"):
        generated_token_count(record)


def test_exported_subset_rebases_relative_image_paths(tmp_path):
    records = simulate_counterfactual_dataset(
        n_states=2,
        num_candidates=2,
        questions_per_image=1,
        seed=5,
    )
    selected_state = records[0].state_id
    decorated = []
    for record in records:
        backend_name = "baseline_backend" if record.action_type == "ANSWER" else "action_backend"
        decorated.append(
            replace(
                record,
                metadata={
                    backend_name: {
                        "generated_tokens": 16 if record.state_id == selected_state else 3
                    }
                },
            )
        )
    source_dir = tmp_path / "source"
    image_dir = source_dir / "images"
    image_dir.mkdir(parents=True)
    manifest_rows = []
    for state_id in sorted({record.state_id for record in records}):
        image_path = image_dir / f"{state_id}.png"
        image_path.write_bytes(b"not decoded by the exporter")
        manifest_rows.append(
            {
                "state_id": state_id,
                "image_path": f"images/{state_id}.png",
                "question": "question",
                "target": "answer",
            }
        )
    source_manifest = source_dir / "manifest.jsonl"
    source_manifest.write_text(
        "\n".join(json.dumps(row) for row in manifest_rows) + "\n",
        encoding="utf-8",
    )
    source_rollouts = tmp_path / "rollouts.jsonl"
    source_rollouts.write_text("fixture\n", encoding="utf-8")
    output_manifest = tmp_path / "subset" / "manifest.jsonl"
    result = export_capped_manifest(
        records=decorated,
        source_manifest=source_manifest,
        source_rollouts=source_rollouts,
        output_manifest=output_manifest,
        token_cap=16,
    )
    exported = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert exported["state_id"] == selected_state
    assert (output_manifest.parent / exported["image_path"]).is_file()
    assert result["states"] == 1
