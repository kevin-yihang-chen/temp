from __future__ import annotations

import copy
import hashlib
import json

import pytest
from PIL import Image

from beyond_entropy.dataset import write_jsonl
from beyond_entropy.predictability_feature_shards import (
    merge_predictability_feature_shards,
)
from beyond_entropy.predictability_features import (
    PREDICTABILITY_FEATURE_FORMAT_VERSION,
    load_predictability_feature_dataset,
)
from beyond_entropy.rollout_shards import shard_directory_name
from beyond_entropy.sharding import SHARD_ALGORITHM, stable_shard_index
from test_predictability_baselines import _siblings
from test_predictability_features import _row


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature_row(state_id: str, index: int) -> dict:
    row = copy.deepcopy(_row())
    row.update(
        {
            "state_id": state_id,
            "image_id": f"image-{state_id}",
            "source_id": f"source-{index // 2}",
            "replicate_id": "replicate-000",
            "image_rgb_sha256": f"{index + 1:064x}",
        }
    )
    row["outcome"].update(
        {
            "state_id": state_id,
            "image_id": row["image_id"],
            "source_id": row["source_id"],
            "replicate_id": row["replicate_id"],
        }
    )
    return row


def test_predictability_feature_shards_merge_with_exact_canonical_coverage(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "white").save(image)
    states = [f"state-{index:03d}" for index in range(16)]
    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for index, state_id in enumerate(states):
            handle.write(
                json.dumps(
                    {
                        "state_id": state_id,
                        "image_id": f"image-{state_id}",
                        "source_id": f"source-{index // 2}",
                        "image_path": str(image),
                        "question": "Question?",
                        "target": {"answers": ["yes"]},
                    }
                )
                + "\n"
            )
    manifest_sha = _sha(manifest)
    all_records = [
        record
        for index, state_id in enumerate(states)
        for record in _siblings(
            state_id=state_id,
            source_id=f"source-{index // 2}",
            entropy_before=0.5,
            y0=0.0,
            crop_outcomes=(1.0, 0.0, 0.0, 0.0),
        )
    ]
    merged_rollouts = tmp_path / "merged-rollouts.jsonl"
    write_jsonl(all_records, merged_rollouts)
    run_root = tmp_path / "shards"
    shard_count = 2
    namespace = "predictability-test-v1"
    for shard_index in range(shard_count):
        shard_states = {
            state_id
            for state_id in states
            if stable_shard_index(state_id, shard_count, namespace=namespace)
            == shard_index
        }
        assert shard_states
        shard_dir = run_root / shard_directory_name(shard_index, shard_count)
        shard_dir.mkdir(parents=True)
        shard_rollouts = shard_dir / "rollouts.jsonl"
        write_jsonl(
            [item for item in all_records if item.state_id in shard_states],
            shard_rollouts,
        )
        rows = [
            _feature_row(state_id, states.index(state_id))
            for state_id in states
            if state_id in shard_states
        ]
        metadata = {
            "schema": "predictability_feature_metadata_v2",
            "dataset_role": "train",
            "manifest": str(manifest.resolve()),
            "manifest_sha256": manifest_sha,
            "rollouts": str(shard_rollouts.resolve()),
            "rollouts_sha256": _sha(shard_rollouts),
            "code_revision": "revision",
            "shard_algorithm": SHARD_ALGORITHM,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "shard_key": "state_id",
            "shard_namespace": namespace,
            "manifest_examples_before_sharding": len(states),
            "shard_states": len(shard_states),
            "invariant": "same",
        }
        torch.save(
            {
                "format_version": PREDICTABILITY_FEATURE_FORMAT_VERSION,
                "metadata": metadata,
                "rows": rows,
            },
            shard_dir / "features.pt",
        )

    output = tmp_path / "merged" / "features.pt"
    report_path = tmp_path / "merged" / "merge.json"
    report = merge_predictability_feature_shards(
        manifest_path=manifest,
        expected_manifest_sha256=manifest_sha,
        merged_rollouts_path=merged_rollouts,
        expected_merged_rollouts_sha256=_sha(merged_rollouts),
        run_root=run_root,
        shard_count=shard_count,
        shard_key="state_id",
        shard_namespace=namespace,
        expected_code_revision="revision",
        dataset_role="train",
        output_path=output,
        report_path=report_path,
    )
    payload, examples = load_predictability_feature_dataset(output)
    assert report["passed"] is True
    assert report["states"] == len(states)
    assert report["decisions"] == len(states)
    assert report["output_sha256"] == _sha(output)
    assert [row["state_id"] for row in payload["rows"]] == states
    assert len(examples) == len(states)
    assert payload["metadata"]["shard_merge"]["complete"] is True
    with pytest.raises(FileExistsError, match="overwrite"):
        merge_predictability_feature_shards(
            manifest_path=manifest,
            expected_manifest_sha256=manifest_sha,
            merged_rollouts_path=merged_rollouts,
            expected_merged_rollouts_sha256=_sha(merged_rollouts),
            run_root=run_root,
            shard_count=shard_count,
            shard_key="state_id",
            shard_namespace=namespace,
            expected_code_revision="revision",
            dataset_role="train",
            output_path=output,
            report_path=report_path,
        )
