import hashlib
import json

import pytest

from beyond_entropy.dataset import write_jsonl
from beyond_entropy.rollout_audit import audit_sibling_rollout_bank
from beyond_entropy.simulate import simulate_counterfactual_dataset


def _write_bank(tmp_path):
    records = simulate_counterfactual_dataset(
        n_states=6,
        num_candidates=4,
        questions_per_image=2,
        seed=43,
    )
    manifest = tmp_path / "manifest.jsonl"
    baselines = [record for record in records if record.action_type == "ANSWER"]
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "state_id": record.state_id,
                    "source_id": record.source_id,
                    "image_id": record.image_id,
                    "question": record.question,
                },
                sort_keys=True,
            )
            + "\n"
            for record in baselines
        ),
        encoding="utf-8",
    )
    rollouts = tmp_path / "rollouts.jsonl"
    write_jsonl(records, rollouts)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rollout_sha = hashlib.sha256(rollouts.read_bytes()).hexdigest()
    provenance = {
        "manifest_sha256": manifest_sha,
        "output_sha256": rollout_sha,
        "examples": 6,
        "completed_examples": 6,
        "candidate_count": 4,
        "model_revision": "revision",
        "scientific_status": "development bank",
        "code_revision": "commit",
        "checkpoint_interval": 2,
    }
    rollouts.with_suffix(".provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    rollouts.with_suffix(".diagnostic.json").write_text(
        json.dumps({"point_estimate": {}}), encoding="utf-8"
    )
    return manifest, manifest_sha, rollouts


def test_rollout_audit_accepts_exact_complete_bank(tmp_path):
    manifest, manifest_sha, rollouts = _write_bank(tmp_path)
    report = audit_sibling_rollout_bank(
        manifest,
        rollouts,
        expected_manifest_sha256=manifest_sha,
        expected_states=6,
        expected_model_revision="revision",
        expected_scientific_status="development bank",
    )
    assert report["passed"] is True
    assert report["records"] == 30
    assert report["unique_sources"] == 3
    assert report["checkpoint_interval"] == 2


def test_rollout_audit_rejects_incomplete_checkpoint(tmp_path):
    manifest, manifest_sha, rollouts = _write_bank(tmp_path)
    lines = rollouts.read_text(encoding="utf-8").splitlines()
    rollouts.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        audit_sibling_rollout_bank(
            manifest,
            rollouts,
            expected_manifest_sha256=manifest_sha,
            expected_states=6,
        )
