from __future__ import annotations

import json
from pathlib import Path

import pytest

from beyond_entropy.answer_likelihood import (
    AnswerLikelihoodScore,
    accepted_answers,
    canonical_target_answer,
    score_rollout_answer_likelihood,
    sha256_file,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_canonical_target_answer_uses_mode_then_shortest_then_order() -> None:
    assert accepted_answers({"answers": [" Cat ", "cat", "a feline"]}) == (
        "Cat",
        "cat",
        "a feline",
    )
    assert canonical_target_answer({"answers": [" Cat ", "cat", "a feline"]}) == (
        "Cat",
        0,
        2,
    )
    assert canonical_target_answer({"answers": ["two words", "short"]}) == (
        "short",
        1,
        1,
    )
    with pytest.raises(ValueError, match="answers or answer"):
        accepted_answers({"label": "missing"})


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "image.png"
    image.write_bytes(b"test-image-placeholder")
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "state_id": "state-1",
                "image_id": "image-1",
                "source_id": "source-1",
                "image_path": str(image),
                "question": "What?",
                "model_prompt": "What?\nAnswer briefly.",
                "target": {"answers": ["long answer", "yes", "yes"]},
            }
        ],
    )
    common = {
        "state_id": "state-1",
        "image_id": "image-1",
        "source_id": "source-1",
        "question": "What?",
        "original_image": str(image),
        "replicate_id": "replicate-000",
        "generation_seed": 0,
        "entropy_before": 0.4,
        "answer_before": "no",
        "correct_before": 0.0,
        "metadata": {},
        "pre_action_features": {},
    }
    rollouts = tmp_path / "rollouts.jsonl"
    write_jsonl(
        rollouts,
        [
            {
                **common,
                "action_id": "answer-now",
                "action_type": "ANSWER",
                "candidate_bbox": None,
                "entropy_after": 0.4,
                "answer_after": "no",
                "correct_after": 0.0,
                "tool_cost": 0.0,
            },
            {
                **common,
                "action_id": "crop-0",
                "action_type": "ZOOM",
                "candidate_bbox": [0.0, 0.0, 0.5, 0.5],
                "entropy_after": 0.2,
                "answer_after": "yes",
                "correct_after": 1.0,
                "tool_cost": 1.0,
            },
        ],
    )
    return manifest, rollouts


def test_score_rollout_answer_likelihood_is_atomic_resumable_and_hides_target(
    tmp_path: Path,
) -> None:
    manifest, rollouts = _fixture(tmp_path)
    output = tmp_path / "scores.jsonl"
    seen: list[str] = []

    def fake_score(request):
        seen.append(request.target_answer)
        return AnswerLikelihoodScore(mean_nll=0.5, sum_nll=1.0, token_count=2)

    result = score_rollout_answer_likelihood(
        manifest=manifest,
        rollouts=rollouts,
        output=output,
        score_request=fake_score,
        expected_manifest_sha256=sha256_file(manifest),
        expected_rollouts_sha256=sha256_file(rollouts),
        checkpoint_interval=1,
        model="test-model",
        model_revision="revision",
        measurement_config={"dtype": "test"},
        code_revision="code",
        scientific_status="test",
    )
    assert result["decisions"] == 1
    assert result["records"] == 2
    assert seen == ["yes", "yes"]
    assert "yes" not in output.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["action_id"] for row in rows] == ["answer-now", "crop-0"]
    assert all(row["answer_mean_nll"] == 0.5 for row in rows)
    assert json.loads(output.with_suffix(".provenance.json").read_text())[
        "raw_targets_written"
    ] is False

    seen.clear()
    resumed = score_rollout_answer_likelihood(
        manifest=manifest,
        rollouts=rollouts,
        output=output,
        score_request=fake_score,
        expected_manifest_sha256=sha256_file(manifest),
        expected_rollouts_sha256=sha256_file(rollouts),
        checkpoint_interval=1,
        resume=True,
        model="test-model",
        model_revision="revision",
        measurement_config={"dtype": "test"},
        code_revision="code",
        scientific_status="test",
    )
    assert resumed["resumed_from_decisions"] == 1
    assert seen == []


def test_answer_likelihood_rejects_hash_and_partial_checkpoint(tmp_path: Path) -> None:
    manifest, rollouts = _fixture(tmp_path)
    output = tmp_path / "scores.jsonl"
    with pytest.raises(ValueError, match="manifest SHA-256"):
        score_rollout_answer_likelihood(
            manifest=manifest,
            rollouts=rollouts,
            output=output,
            score_request=lambda request: AnswerLikelihoodScore(0.5, 1.0, 2),
            expected_manifest_sha256="0" * 64,
            model="test-model",
            model_revision="revision",
            measurement_config={"dtype": "test"},
            code_revision="code",
            scientific_status="test",
        )

    output.write_text(
        json.dumps(
            {
                "state_id": "state-1",
                "replicate_id": "replicate-000",
                "action_id": "answer-now",
                "config_sha256": "wrong",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="within a decision"):
        score_rollout_answer_likelihood(
            manifest=manifest,
            rollouts=rollouts,
            output=output,
            score_request=lambda request: AnswerLikelihoodScore(0.5, 1.0, 2),
            resume=True,
            model="test-model",
            model_revision="revision",
            measurement_config={"dtype": "test"},
            code_revision="code",
            scientific_status="test",
        )


def test_screenqa_proxy_nll_smoke_slurm_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (root / "scripts/slurm_screenqa_proxy_nll_smoke.sh").read_text()
    submitter = (root / "scripts/submit_screenqa_proxy_nll_smoke.sh").read_text()
    assert "#SBATCH --gres=gpu:rtx_4090:1" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "--mail-user=\"${notify_email}\"" in submitter
    assert "--mail-type=ALL" in submitter
    assert "--shard-count 14511" in worker
    assert "--shard-index 0" in worker
    assert "--resume" in worker
    assert "raw_targets_written" in worker
    assert "protected role opened" in worker
