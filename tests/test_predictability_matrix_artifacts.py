from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import pytest

from beyond_entropy.dataset import write_jsonl
from beyond_entropy.predictability_matrix_artifacts import load_role_artifacts
from beyond_entropy.predictability_matrix_smoke import build_synthetic_datasets


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_role_artifacts_require_hashes_metadata_and_fixed_tool_labels(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    role = build_synthetic_datasets()["chartqa"]
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "state_id": item.outcome.state_id,
                    "image_id": item.outcome.image_id,
                    "source_id": item.outcome.source_id,
                    "image_path": str(image),
                    "question": "synthetic question",
                    "target": "synthetic answer",
                },
                sort_keys=True,
            )
            + "\n"
            for item in role.validation
        ),
        encoding="utf-8",
    )
    rollouts = tmp_path / "rollouts.jsonl"
    write_jsonl(role.validation_siblings, rollouts)
    revision = "a" * 40

    rows = []
    for example, post in zip(role.validation, role.post_action_validation):
        pre_action = asdict(example.inputs)
        for name in ("state_id", "image_id", "source_id"):
            pre_action.pop(name)
        post_action = asdict(post.inputs)
        for name in ("state_id", "image_id", "source_id"):
            post_action.pop(name)
        post_action["candidate_action_ids"] = [
            "ug-grid-00",
            "ug-grid-01",
            "ug-grid-02",
            "ug-grid-03",
        ]
        rows.append(
            {
                "state_id": example.outcome.state_id,
                "image_id": example.outcome.image_id,
                "source_id": example.outcome.source_id,
                "replicate_id": example.outcome.replicate_id,
                "image_rgb_sha256": example.image_rgb_sha256,
                "pre_action": pre_action,
                "post_action_probe": post_action,
                "outcome": asdict(example.outcome),
            }
        )
    features = tmp_path / "features.pt"
    torch.save(
        {
            "format_version": 2,
            "metadata": {
                "dataset_role": "validation",
                "manifest_sha256": _sha256(manifest),
                "rollouts_sha256": _sha256(rollouts),
                "code_revision": revision,
            },
            "rows": rows,
        },
        features,
    )
    rollout_provenance = tmp_path / "rollouts.provenance.json"
    rollout_provenance.write_text(
        json.dumps(
            {
                "examples": len(role.validation),
                "completed_examples": len(role.validation),
                "scorer": "chartqa",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    spec = {
        "manifest": {"path": str(manifest), "sha256": _sha256(manifest)},
        "rollouts": {"path": str(rollouts), "sha256": _sha256(rollouts)},
        "rollout_provenance": {
            "path": str(rollout_provenance),
            "sha256": _sha256(rollout_provenance),
        },
        "features": {"path": str(features), "sha256": _sha256(features)},
    }
    loaded = load_role_artifacts(
        spec,
        benchmark="chartqa",
        role="validation",
        code_revision=revision,
    )
    assert len(loaded.examples) == len(role.validation)
    assert len(loaded.post_action_examples) == len(role.post_action_validation)
    assert len(loaded.siblings) == len(role.validation_siblings)

    spec["features"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="features SHA-256 mismatch"):
        load_role_artifacts(
            spec,
            benchmark="chartqa",
            role="validation",
            code_revision=revision,
        )
