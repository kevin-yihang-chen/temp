from __future__ import annotations

import json
from pathlib import Path

import pytest

from beyond_entropy.backbone_diagnostic import sha256_file
from beyond_entropy.backbone_smoke import verify_backbone_engineering_smoke


MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "fixture-revision"
CODE = "fixture-code"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "state_id": f"state-{index}",
                "source_id": f"source-{index}",
                "image_path": f"images/{index}.png",
                "question": "fixture",
                "target": {"answers": ["answer"]},
            }
            for index in range(2)
        ],
    )
    manifest_sha256 = sha256_file(manifest)
    backend = {"model": MODEL, "model_revision": REVISION}
    rollout_rows: list[dict[str, object]] = []
    nll_rows: list[dict[str, object]] = []
    for index in range(2):
        for action_index, action_type in enumerate(("ANSWER", "ZOOM", "ZOOM", "ZOOM", "ZOOM")):
            action_id = "answer-now" if action_type == "ANSWER" else f"crop-{action_index}"
            metadata: dict[str, object] = {"baseline_backend": backend}
            if action_type == "ZOOM":
                metadata["action_backend"] = backend
            rollout_rows.append(
                {
                    "state_id": f"state-{index}",
                    "source_id": f"source-{index}",
                    "action_id": action_id,
                    "action_type": action_type,
                    "metadata": metadata,
                }
            )
            nll_rows.append(
                {
                    "state_id": f"state-{index}",
                    "source_id": f"source-{index}",
                    "action_id": action_id,
                    "action_type": action_type,
                    "answer_mean_nll": 0.5,
                    "answer_token_count": 2,
                    "config_sha256": "config",
                    "target_answer_sha256": "target-hash",
                }
            )
    rollouts = tmp_path / "rollouts.jsonl"
    answer_nll = tmp_path / "answer-nll.jsonl"
    _write_jsonl(rollouts, rollout_rows)
    _write_jsonl(answer_nll, nll_rows)
    rollout_provenance = tmp_path / "rollouts.provenance.json"
    _write_json(
        rollout_provenance,
        {
            "manifest_sha256": manifest_sha256,
            "manifest_limit": 2,
            "manifest_examples_before_sharding": 2,
            "shard_count": 1,
            "shard_index": 0,
            "examples": 2,
            "completed_examples": 2,
            "resumed_from_records": 10,
            "candidate_count": 4,
            "model": MODEL,
            "model_revision": REVISION,
            "code_revision": CODE,
            "output_sha256": sha256_file(rollouts),
            "scorer": "screenqa",
        },
    )
    rollout_resume_audit = tmp_path / "resume.audit.json"
    _write_json(
        rollout_resume_audit,
        {
            "passed": True,
            "records": 10,
            "resumed_from_records": 10,
            "rollouts_sha256_before_resume": sha256_file(rollouts),
            "rollouts_sha256_after_resume": sha256_file(rollouts),
        },
    )
    nll_provenance = tmp_path / "answer-nll.provenance.json"
    _write_json(
        nll_provenance,
        {
            "manifest_sha256": manifest_sha256,
            "manifest_limit": 2,
            "manifest_examples_before_sharding": 2,
            "rollouts_sha256": sha256_file(rollouts),
            "output_sha256": sha256_file(answer_nll),
            "decisions": 2,
            "records": 10,
            "sources": 2,
            "shard_count": 1,
            "shard_index": 0,
            "resumed_from_decisions": 2,
            "raw_targets_written": False,
            "model": MODEL,
            "model_revision": REVISION,
            "code_revision": CODE,
            "measurement_config": {"accelerator_name": "NVIDIA H100"},
        },
    )
    return {
        "manifest": manifest,
        "rollouts": rollouts,
        "rollout_provenance": rollout_provenance,
        "rollout_resume_audit": rollout_resume_audit,
        "answer_nll": answer_nll,
        "answer_nll_provenance": nll_provenance,
        "manifest_sha256": manifest_sha256,
    }


def _verify(paths: dict[str, Path | str], *, output: Path) -> dict[str, object]:
    return verify_backbone_engineering_smoke(
        manifest=paths["manifest"],
        rollouts=paths["rollouts"],
        rollout_provenance=paths["rollout_provenance"],
        rollout_resume_audit=paths["rollout_resume_audit"],
        answer_nll=paths["answer_nll"],
        answer_nll_provenance=paths["answer_nll_provenance"],
        output=output,
        expected_manifest_sha256=str(paths["manifest_sha256"]),
        expected_decisions=2,
        expected_model=MODEL,
        expected_model_revision=REVISION,
        expected_gpu_name="H100",
        expected_code_revision=CODE,
        rollout_seconds=10.0,
        rollout_resume_seconds=2.0,
        answer_nll_seconds=5.0,
        answer_nll_resume_seconds=3.0,
    )


def test_backbone_smoke_verifies_endpoint_blind_artifacts(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = _verify(paths, output=tmp_path / "smoke.complete.json")
    assert result["passed"] is True
    assert result["population"]["decisions"] == 2
    assert result["timing_seconds"]["first_pass_total"] == pytest.approx(15.0)
    assert result["timing_seconds"]["engineering_total"] == pytest.approx(20.0)
    assert result["outcome_use"]["task_endpoints_computed"] is False


def test_backbone_smoke_rejects_raw_target_and_wrong_gpu(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    answer_nll = Path(paths["answer_nll"])
    rows = [json.loads(line) for line in answer_nll.read_text(encoding="utf-8").splitlines()]
    rows[0]["target_answer"] = "forbidden"
    _write_jsonl(answer_nll, rows)
    with pytest.raises(ValueError, match="raw target"):
        _verify(paths, output=tmp_path / "raw-target.json")

    rows[0].pop("target_answer")
    _write_jsonl(answer_nll, rows)
    nll_provenance = Path(paths["answer_nll_provenance"])
    payload = json.loads(nll_provenance.read_text(encoding="utf-8"))
    payload["output_sha256"] = sha256_file(answer_nll)
    payload["measurement_config"]["accelerator_name"] = "NVIDIA RTX 4090"
    _write_json(nll_provenance, payload)
    with pytest.raises(ValueError, match="accelerator"):
        _verify(paths, output=tmp_path / "wrong-gpu.json")


def test_backbone_smoke_workers_are_contract_locked() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (root / "scripts/slurm_screenqa_backbone_7b_smoke.sh").read_text()
    submit = (root / "scripts/submit_screenqa_backbone_7b_smoke.sh").read_text()
    assert "cc594898137f460bfe9f0759e9844b3ce807cfb5" in worker
    assert "4af43ac80a1666c174774d1c33383ad" in worker
    assert "1cd70d11168e12a2855ec01e8a869d" in worker
    assert "a26b8bc6e8a7c81df3cad59f05ac3c" in worker
    assert "--limit 32" in worker
    assert "--manifest-limit 32" in worker
    assert "--expected-gpu-name" in worker
    assert "task endpoint may select hardware" in worker
    assert "HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache" in worker
    assert "q-hgpu-small" in submit
    assert "gpu:h100:1" in submit
    assert "gpu:h800:1" in submit
    assert "gpu:rtx_4090:1" in submit
    assert "show-cpu-gpu-quota" in submit
    assert "HF_HUB_CACHE=/userhome/cs3/yihangc/Data/hf_cache" in submit
    assert "--mail-type=ALL" in submit
