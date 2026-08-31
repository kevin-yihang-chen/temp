from __future__ import annotations

import json
from pathlib import Path

import pytest

from beyond_entropy.answer_likelihood import SCHEMA as SCORE_SCHEMA
from beyond_entropy.answer_likelihood import TARGET_RULE, sha256_file
from beyond_entropy.proxy_outcome_audit import (
    AUDIT_SCHEMA,
    MERGE_SCHEMA,
    analyze_proxy_outcomes,
    compare_proxy_nll_hardware,
    merge_answer_likelihood_shards,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _score_rows(decision_index: int, config_sha256: str) -> list[dict[str, object]]:
    patterns = (
        (0.0, (1.0, 0.0, 0.0, 0.0), (0.40, 0.20, -0.10, 0.00)),
        (0.0, (0.0, 0.0, 0.0, 0.0), (0.30, 0.10, 0.00, -0.10)),
        (1.0, (-1.0, 0.0, 0.0, 0.0), (0.50, 0.20, 0.00, -0.20)),
        (0.0, (1.0, 0.0, 0.0, 0.0), (-0.20, 0.30, 0.10, 0.00)),
    )
    before, gains, loss_gaps = patterns[decision_index % len(patterns)]
    state_id = f"state-{decision_index:02d}"
    source_id = "source-a" if decision_index < 4 else "source-b"
    common: dict[str, object] = {
        "schema": SCORE_SCHEMA,
        "config_sha256": config_sha256,
        "state_id": state_id,
        "replicate_id": "replicate-000",
        "source_id": source_id,
        "image_id": f"image-{decision_index:02d}",
        "target_answer_sha256": "a" * 64,
        "target_answer_index": 0,
        "target_answer_votes": 2,
        "target_answer_count": 2,
        "answer_token_count": 2,
        "entropy_before": 0.5,
        "correct_before": before,
    }
    rows = [
        {
            **common,
            "action_id": "answer-now",
            "action_type": "ANSWER",
            "answer_mean_nll": 1.0,
            "answer_sum_nll": 2.0,
            "entropy_after": 0.5,
            "correct_after": before,
            "tool_cost": 0.0,
        }
    ]
    entropy_gaps = (0.10, 0.30, -0.05, 0.00)
    for action_index, (gain, loss_gap, entropy_gap) in enumerate(
        zip(gains, loss_gaps, entropy_gaps, strict=True)
    ):
        mean_nll = 1.0 - loss_gap
        rows.append(
            {
                **common,
                "action_id": f"crop-{action_index}",
                "action_type": "ZOOM",
                "answer_mean_nll": mean_nll,
                "answer_sum_nll": 2.0 * mean_nll,
                "entropy_after": 0.5 - entropy_gap,
                "correct_after": before + gain,
                "tool_cost": 1.0,
            }
        )
    return rows


def _score_shards(tmp_path: Path) -> list[Path]:
    shards: list[Path] = []
    for shard_index in range(4):
        path = tmp_path / f"shard-{shard_index}.jsonl"
        config_sha256 = f"{shard_index + 1:064x}"
        rows: list[dict[str, object]] = []
        for decision_index in range(shard_index, 8, 4):
            rows.extend(_score_rows(decision_index, config_sha256))
        _write_jsonl(path, rows)
        _write_json(
            path.with_suffix(".provenance.json"),
            {
                "schema": SCORE_SCHEMA,
                "target_rule": TARGET_RULE,
                "manifest_sha256": "b" * 64,
                "rollouts_sha256": "c" * 64,
                "model": "test-model",
                "model_revision": "test-revision",
                "measurement_config": {
                    "accelerator_name": "test-device",
                    "requested_dtype": "bfloat16",
                },
                "code_revision": "test-code",
                "shard_count": 4,
                "shard_index": shard_index,
                "scientific_status": "test development audit",
                "config_sha256": config_sha256,
                "decisions": 2,
                "records": len(rows),
                "sources": len({str(row["source_id"]) for row in rows}),
                "output_sha256": sha256_file(path),
                "raw_targets_written": False,
            },
        )
        shards.append(path)
    return shards


def test_merge_and_audit_proxy_outcomes_are_deterministic(tmp_path: Path) -> None:
    shards = _score_shards(tmp_path)
    merged = tmp_path / "merged.jsonl"
    merge = merge_answer_likelihood_shards(
        shards=shards,
        output=merged,
        expected_decisions=8,
        expected_records=40,
        expected_sources=2,
    )
    assert merge["schema"] == MERGE_SCHEMA
    assert merge["raw_targets_written"] is False
    assert [entry["shard_index"] for entry in merge["input_shards"]] == [0, 1, 2, 3]
    assert "target_answer\"" not in merged.read_text(encoding="utf-8")

    protocol = tmp_path / "protocol.md"
    protocol.write_text("# frozen test protocol\n", encoding="utf-8")
    implementation = tmp_path / "implementation.md"
    implementation.write_text("# frozen implementation\n", encoding="utf-8")
    first = analyze_proxy_outcomes(
        scores=merged,
        protocol=protocol,
        implementation_contract=implementation,
        output_dir=tmp_path / "audit-1",
        expected_scores_sha256=sha256_file(merged),
        expected_protocol_sha256=sha256_file(protocol),
        expected_implementation_contract_sha256=sha256_file(implementation),
        expected_decisions=8,
        expected_sources=2,
        bootstrap_resamples=40,
        bootstrap_seed=17,
        code_revision="test-code",
    )
    second = analyze_proxy_outcomes(
        scores=merged,
        protocol=protocol,
        implementation_contract=implementation,
        output_dir=tmp_path / "audit-2",
        expected_scores_sha256=sha256_file(merged),
        expected_protocol_sha256=sha256_file(protocol),
        expected_implementation_contract_sha256=sha256_file(implementation),
        expected_decisions=8,
        expected_sources=2,
        bootstrap_resamples=40,
        bootstrap_seed=17,
        code_revision="test-code",
    )
    assert first == second
    assert first["schema"] == AUDIT_SCHEMA
    assert first["population"] == {
        "sources": 2,
        "decisions": 8,
        "score_records": 40,
        "zoom_actions": 32,
        "helpful_decisions": 4,
    }
    assert first["top_one"]["random_expected"]["definition"].startswith("exact")
    assert first["study"]["label"] == "ScreenQA"
    assert first["disagreements"]["loss_improves_task_falls"]["count"] == 2
    assert (
        first["disagreements"]["task_improves_without_positive_loss_gap"]["count"]
        == 2
    )
    assert first["outcome_use"]["calibration_or_formal_inputs_used"] is False
    assert (tmp_path / "audit-1/report.md").is_file()
    assert (tmp_path / "audit-1/audit.complete.json").is_file()

    docvqa = analyze_proxy_outcomes(
        scores=merged,
        protocol=protocol,
        implementation_contract=implementation,
        output_dir=tmp_path / "audit-docvqa",
        expected_scores_sha256=sha256_file(merged),
        expected_protocol_sha256=sha256_file(protocol),
        expected_implementation_contract_sha256=sha256_file(implementation),
        expected_decisions=8,
        expected_sources=2,
        bootstrap_resamples=40,
        bootstrap_seed=17,
        study_label="DocVQA ranker development",
        scientific_status="opened-bank cross-domain replication",
        interpretation_boundary="No protected DocVQA or ScreenQA role is an input.",
        code_revision="test-code",
    )
    assert docvqa["study"]["label"] == "DocVQA ranker development"
    assert docvqa["scientific_status"] == "opened-bank cross-domain replication"
    assert (tmp_path / "audit-docvqa/report.md").read_text().startswith(
        "# DocVQA ranker development proxy-to-outcome audit\n"
    )
    assert docvqa["outcome_use"] == {
        "opened_ranker_development_used": True,
        "candidate_search_reopened": False,
        "calibration_or_formal_inputs_used": False,
        "reserve_validation_or_test_inputs_used": False,
        "protected_role_inputs_used": False,
    }


def test_merge_rejects_tampered_shard_configuration(tmp_path: Path) -> None:
    shards = _score_shards(tmp_path)
    rows = [json.loads(line) for line in shards[0].read_text().splitlines()]
    rows[0]["config_sha256"] = "f" * 64
    _write_jsonl(shards[0], rows)
    provenance_path = shards[0].with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text())
    provenance["output_sha256"] = sha256_file(shards[0])
    _write_json(provenance_path, provenance)
    with pytest.raises(ValueError, match="configuration hash"):
        merge_answer_likelihood_shards(shards=shards, output=tmp_path / "merged.jsonl")


def test_screenqa_proxy_h800_job_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    benchmark = (root / "scripts/slurm_screenqa_proxy_nll_benchmark.sh").read_text()
    benchmark_submit = (
        root / "scripts/submit_screenqa_proxy_nll_benchmark_h800.sh"
    ).read_text()
    benchmark_4090_submit = (
        root / "scripts/submit_screenqa_proxy_nll_benchmark_4090.sh"
    ).read_text()
    full = (root / "scripts/slurm_screenqa_proxy_nll_full_4090.sh").read_text()
    full_submit = (
        root / "scripts/submit_screenqa_proxy_nll_full_4090.sh"
    ).read_text()
    assert "#SBATCH --partition=q-hgpu-small" in benchmark
    assert "#SBATCH --gres=gpu:h800:1" in benchmark
    assert "--shard-count 227" in benchmark
    assert '"${decisions}" -ne 64' in benchmark
    assert "#SBATCH --mail-type=ALL" in benchmark
    assert "BE_PROXY_BENCHMARK_GPU_TYPE" in benchmark
    assert "measurement_config.accelerator_name" in benchmark
    assert '--mail-user="${notify_email}"' in benchmark_submit
    assert "--mail-type=ALL" in benchmark_submit
    assert "--gres=gpu:rtx_4090:1" in benchmark_4090_submit
    assert "--partition=debug" in benchmark_4090_submit
    assert "--mail-type=ALL" in benchmark_4090_submit

    assert "#SBATCH --partition=debug" in full
    assert "#SBATCH --gres=gpu:rtx_4090:4" in full
    assert "#SBATCH --time=04:00:00" in full
    assert "#SBATCH --mail-type=ALL" in full
    assert "--shard-count 4" in full
    assert "--expected-decisions 14511" in full
    assert "--expected-records 72555" in full
    assert "--bootstrap-resamples 2000" in full
    assert "--implementation-contract" in full
    assert "frozen_implementation_sha256=" in full
    assert "frozen_hardware_activation_sha256=" in full
    assert "measurement_config.accelerator_name" in full
    assert "protected role opened" in full
    assert '--mail-user="${notify_email}"' in full_submit
    assert "--mail-type=ALL" in full_submit
    assert "BE_PROXY_FULL_RESUME" in full_submit


def _hardware_score_run(tmp_path: Path, gpu_type: str) -> tuple[Path, Path]:
    score_path = tmp_path / f"{gpu_type}.jsonl"
    config_sha256 = ("d" if gpu_type == "h800" else "e") * 64
    rows = [
        row
        for decision_index in range(8)
        for row in _score_rows(decision_index, config_sha256)
    ]
    _write_jsonl(score_path, rows)
    accelerator = "NVIDIA H800" if gpu_type == "h800" else "NVIDIA GeForce RTX 4090"
    capability = [9, 0] if gpu_type == "h800" else [8, 9]
    measurement = {
        "device_map": "cuda:0",
        "device_type": "cuda",
        "accelerator_name": accelerator,
        "compute_capability": capability,
        "requested_dtype": "bfloat16",
        "parameter_dtype": "torch.bfloat16",
        "attention_implementation": "sdpa",
        "actual_attention_implementation": "sdpa",
        "min_pixels": 200704,
        "max_pixels": 602112,
        "system_prompt": "You are a helpful assistant.",
        "torch_version": "test",
        "cuda_runtime_version": "test",
        "transformers_version": "test",
        "local_files_only": True,
    }
    _write_json(
        score_path.with_suffix(".provenance.json"),
        {
            "schema": SCORE_SCHEMA,
            "target_rule": TARGET_RULE,
            "manifest_sha256": "b" * 64,
            "rollouts_sha256": "c" * 64,
            "model": "test-model",
            "model_revision": "test-revision",
            "measurement_config": measurement,
            "code_revision": "test-code",
            "shard_count": 227,
            "shard_index": 0,
            "scientific_status": "engineering test",
            "config_sha256": config_sha256,
            "decisions": 8,
            "records": 40,
            "sources": 2,
            "output_sha256": sha256_file(score_path),
            "raw_targets_written": False,
        },
    )
    benchmark_path = tmp_path / f"{gpu_type}-benchmark.json"
    _write_json(
        benchmark_path,
        {
            "schema": "screenqa_proxy_nll_hardware_benchmark_v1",
            "gpu_type": gpu_type,
            "accelerator_name": accelerator,
            "elapsed_seconds": 10,
            "projected_four_gpu_full_wall_seconds": 1000,
            "projected_four_gpu_gpu_minutes": 100,
            "output_sha256": sha256_file(score_path),
            "code_revision": "test-code",
        },
    )
    return score_path, benchmark_path


def test_hardware_consistency_audit_applies_frozen_4090_preference(tmp_path: Path) -> None:
    h800_scores, h800_benchmark = _hardware_score_run(tmp_path, "h800")
    rtx_scores, rtx_benchmark = _hardware_score_run(tmp_path, "rtx_4090")
    protocol = tmp_path / "hardware-protocol.md"
    protocol.write_text("# frozen hardware protocol\n", encoding="utf-8")
    report = compare_proxy_nll_hardware(
        first_scores=h800_scores,
        first_benchmark=h800_benchmark,
        second_scores=rtx_scores,
        second_benchmark=rtx_benchmark,
        protocol=protocol,
        output_dir=tmp_path / "hardware-audit",
        expected_first_scores_sha256=sha256_file(h800_scores),
        expected_second_scores_sha256=sha256_file(rtx_scores),
        expected_protocol_sha256=sha256_file(protocol),
        expected_decisions=8,
        remaining_gpu_minutes=500,
        code_revision="test-code",
    )
    assert report["loss_gap_consistency"]["pearson"] == pytest.approx(1.0)
    assert report["loss_gap_consistency"]["spearman"] == pytest.approx(1.0)
    assert report["loss_gap_consistency"]["top_one_crop_agreement"] == 1.0
    assert report["hardware_decision"]["selected"] == "rtx_4090"
    assert report["hardware_decision"]["gates"]["h800_all_stability_gates"] is True
    assert (tmp_path / "hardware-audit/audit.complete.json").is_file()
