from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image
from beyond_entropy.schema import ActionRecord, BBox
import scripts.analyze_screenqa_formal_paper as paper_analysis

from scripts.screenqa_formal_one_shot import (
    COMPLETION_NAME,
    SCIENTIFIC_STATUS,
    build_contract,
    complete_shard,
    open_shard,
    verify_shard_completion,
)
from scripts.render_screenqa_formal import render
from scripts.analyze_screenqa_formal_paper import (
    PRIMARY,
    bootstrap_policy_table,
    holm_adjust,
    write_outputs,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path, count: int = 24) -> Path:
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "white").save(image)
    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "state_id": f"formal-state-{index:03d}",
                        "image_id": f"image-{index:03d}",
                        "source_id": f"screenqa:source-{index:03d}",
                        "image_path": str(image),
                        "question": "Question?",
                        "target": {"answers": ["yes"]},
                    }
                )
                + "\n"
            )
    return manifest


def _contract(tmp_path: Path):
    manifest = _manifest(tmp_path)
    shard_dir = tmp_path / "run" / "shard-00000-of-00004"
    contract = build_contract(
        shard_dir=shard_dir,
        manifest=manifest,
        expected_manifest_sha256=_sha(manifest),
        manifest_audit_sha256="a" * 64,
        candidate_bundle_sha256="b" * 64,
        calibration_bundle_sha256="c" * 64,
        code_revision="d" * 40,
        shard_index=0,
        shard_count=4,
        expected_total_states=24,
    )
    return manifest, shard_dir, contract


def _materialize_completed_shard(shard_dir: Path, contract: dict):
    records = int(contract["expected_shard_records"])
    rollouts = shard_dir / "rollouts.jsonl"
    rollouts.write_text("{}\n" * records, encoding="utf-8")
    rollout_sha = _sha(rollouts)
    provenance = {
        "scientific_status": SCIENTIFIC_STATUS,
        "code_revision": contract["code_revision"],
        "manifest_sha256": contract["manifest_sha256"],
        "manifest_limit": None,
        "manifest_examples_before_sharding": contract["expected_total_states"],
        "shard_algorithm": "sha256-state-id-v1",
        "shard_count": 4,
        "shard_index": 0,
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "scorer": "screenqa",
        "examples": contract["expected_shard_states"],
        "completed_examples": contract["expected_shard_states"],
        "candidate_count": 4,
        "proposer": "ug-grid",
        "visual_crop_ratio": 2.0,
        "visual_cost": 1.0,
        "generation_seeds": [0],
        "bootstrap_seed": 20260831,
        "max_new_tokens": 32,
        "min_pixels": 200704,
        "max_pixels": 602112,
        "attention_implementation": "sdpa",
        "system_prompt": "You are a helpful assistant.",
        "local_files_only": True,
        "output_sha256": rollout_sha,
        "resumed_from_records": records,
    }
    (shard_dir / "rollouts.provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    first = dict(provenance)
    first["resumed_from_records"] = 0
    (shard_dir / "rollouts.first-pass.provenance.json").write_text(
        json.dumps(first), encoding="utf-8"
    )
    (shard_dir / "rollouts.diagnostic.json").write_text("{}\n", encoding="utf-8")
    (shard_dir / "resume.audit.json").write_text(
        json.dumps(
            {
                "passed": True,
                "rollouts_sha256_before_resume": rollout_sha,
                "rollouts_sha256_after_resume": rollout_sha,
                "records": records,
                "examples": contract["expected_shard_states"],
                "resumed_from_records": records,
            }
        ),
        encoding="utf-8",
    )
    return rollouts


def test_screenqa_formal_ledger_allows_only_exact_checkpoint_recovery(tmp_path):
    _, shard_dir, contract = _contract(tmp_path)
    first = open_shard(shard_dir, contract)
    assert first["ledger_created"] is True
    resumed = open_shard(shard_dir, contract)
    assert resumed["exact_contract_checkpoint_recovery"] is True

    changed = dict(contract)
    changed["candidate_bundle_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="start ledger changed"):
        open_shard(shard_dir, changed)


def test_screenqa_formal_completion_is_frozen_and_tamper_evident(tmp_path):
    _, shard_dir, contract = _contract(tmp_path)
    open_shard(shard_dir, contract)
    rollouts = _materialize_completed_shard(shard_dir, contract)
    completed = complete_shard(shard_dir, contract)
    assert completed["one_shot_formal_shard_complete"] is True
    assert completed["formal_outcomes_used_for_tuning"] is False
    assert verify_shard_completion(shard_dir, contract) == completed
    assert (shard_dir / COMPLETION_NAME).is_file()

    with pytest.raises(FileExistsError, match="cannot be reopened"):
        open_shard(shard_dir, contract)
    rollouts.write_text(rollouts.read_text(encoding="utf-8") + "{}\n")
    with pytest.raises(ValueError, match="record count mismatch"):
        verify_shard_completion(shard_dir, contract)


def test_screenqa_formal_slurm_chain_is_one_shot_hashed_and_notified():
    root = Path(__file__).resolve().parents[1]
    worker_path = root / "scripts/slurm_screenqa_formal_bank.sh"
    merge_path = root / "scripts/slurm_screenqa_formal_bank_merge.sh"
    evaluate_path = root / "scripts/slurm_screenqa_formal_evaluate.sh"
    analysis_path = root / "scripts/slurm_screenqa_formal_paper_analysis.sh"
    submit_path = root / "scripts/submit_screenqa_formal_bank.sh"
    for path in (worker_path, merge_path, evaluate_path, analysis_path, submit_path):
        subprocess.run(["bash", "-n", str(path)], check=True)
    worker = worker_path.read_text(encoding="utf-8")
    merge = merge_path.read_text(encoding="utf-8")
    evaluate = evaluate_path.read_text(encoding="utf-8")
    analysis = analysis_path.read_text(encoding="utf-8")
    submit = submit_path.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-3" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "screenqa_formal_one_shot open" in worker
    assert "screenqa_formal_one_shot complete" in worker
    assert "already completed under the exact contract" in worker
    assert "--resume" in worker
    assert "no-generation proof" in worker
    assert "screenqa_formal_one_shot verify" in merge
    assert "--require-resume-audit" in merge
    assert "formal-bank.complete.json" in merge
    assert "evaluate_screenqa_formal" in evaluate
    assert "--bootstrap-resamples 20000" in evaluate
    assert "--bootstrap-confidence 0.975" in evaluate
    assert "--bootstrap-seed 20260831" in evaluate
    assert "formal-result.complete.json" in evaluate
    assert "one-shot formal evaluation was already finalized" in evaluate
    assert 'if [[ ! -e "${report}" ]]' in evaluate
    assert "analyze_screenqa_formal_paper" in analysis
    assert "--self-test" in analysis
    assert "formal_outcomes_used_for_tuning" in analysis
    assert "BE_SCREENQA_FORMAL_PAPER_PROTOCOL_SHA256" in analysis
    assert "--expected-protocol-sha256" in analysis
    assert "refusing to reuse existing ScreenQA one-shot formal outcomes" in submit
    assert '--dependency="afterok:${array_job_id}"' in submit
    assert '--dependency="afterok:${merge_job_id}"' in submit
    assert '--dependency="afterok:${evaluation_job_id}"' in submit
    assert "formal-paper-analysis-protocol-v1.md" in submit
    assert '--mail-user="${notify_email}"' in submit
    assert "--mail-type=ALL" in submit


def test_screenqa_formal_verifier_contract_has_exact_population_and_boundaries():
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "scripts/verify_screenqa_formal_manifest.py").read_text()
    rollouts = (root / "scripts/verify_screenqa_formal_rollouts.py").read_text()
    assert "EXPECTED_STATES = 14672" in manifest
    assert "EXPECTED_IMAGES = 6000" in manifest
    assert "EXPECTED_SOURCES = 1471" in manifest
    assert '"reserve_opened": False' in manifest
    assert "EXPECTED_RECORDS = 73360" in rollouts
    assert "one_shot_completed_shards" in rollouts
    assert '"formal_outcomes_used_for_tuning": False' in rollouts
    assert '"official_validation_test_opened": False' in rollouts


def test_screenqa_formal_renderer_marks_registered_decision_and_cost_controls():
    source = {
        "utility": 0.002,
        "call": 0.02,
        "random_always_call_utility": -0.01,
        "post_action_entropy_always_call_utility": -0.02,
        "ug_style_exhaustive_entropy_utility": -0.15,
    }
    report = {
        "passed": True,
        "threshold": 0.2,
        "n_sources": 1471,
        "n_decisions": 14672,
        "source_balanced": source,
        "question_weighted": {"utility": 0.0015},
        "source_bootstrap": {
            "metrics": {"utility": {"ci_low": 0.0001, "ci_high": 0.0039}}
        },
        "pass_rule": {"source_utility_positive": True},
        "baselines": {
            "matched_budget_entropy_gate_source_utility_learned_crop": 0.001,
            "matched_budget_entropy_gate_source_utility_random_crop": -0.001,
            "matched_budget_random_gate_source_utility_random_crop_expected": -0.002,
            "fixed_crop_source_utility_always_call": {
                "zoom-0": -0.01,
                "zoom-1": -0.02,
            },
        },
    }
    markdown = render(report)
    assert "Registered decision: **PASS**" in markdown
    assert "97.5% whole-source interval" in markdown
    assert "Exhaustive UG entropy, four calls charged" in markdown
    assert "Post-action entropy selection is explicitly idealized" in markdown


def test_screenqa_paired_bootstrap_is_deterministic_and_holm_adjusted():
    source_values = {
        PRIMARY: {"s0": 0.4, "s1": 0.3, "s2": 0.5, "s3": 0.4},
        "no_call": {"s0": 0.0, "s1": 0.0, "s2": 0.0, "s3": 0.0},
        "control": {"s0": 0.1, "s1": 0.0, "s2": 0.2, "s3": 0.1},
    }
    first = bootstrap_policy_table(
        source_values, n_resamples=1000, confidence_level=0.95, seed=19
    )
    second = bootstrap_policy_table(
        source_values, n_resamples=1000, confidence_level=0.95, seed=19
    )
    assert first == second
    assert all(row["paired_ci_low"] > 0.0 for row in first[1].values())
    assert holm_adjust({"a": 0.01, "b": 0.03, "c": 0.02}) == {
        "a": 0.03,
        "c": 0.04,
        "b": 0.04,
    }


def _paired_decision(state: str, source: str, entropy: float):
    common = {
        "state_id": state,
        "image_id": source,
        "source_id": source,
        "question": "question",
        "original_image": "/sealed/image.png",
        "replicate_id": "replicate-000",
        "generation_seed": 0,
        "entropy_before": entropy,
        "answer_before": "before",
        "correct_before": 0.0,
        "pre_action_features": {},
        "metadata": {},
    }
    records = [
        ActionRecord(
            **common,
            action_id="ANSWER",
            action_type="ANSWER",
            candidate_bbox=None,
            entropy_after=entropy,
            answer_after="before",
            correct_after=0.0,
            tool_cost=0.0,
        )
    ]
    for index, action_id in enumerate(("a", "b", "c", "d")):
        records.append(
            ActionRecord(
                **common,
                action_id=action_id,
                action_type="ZOOM",
                candidate_bbox=BBox(0.0, 0.0, 0.5, 0.5),
                entropy_after=0.1 + index,
                answer_after=action_id,
                correct_after=1.0 if action_id == "a" else 0.0,
                tool_cost=1.0,
            )
        )
    return records


def test_screenqa_paired_policy_reconstruction_charges_gate_and_exhaustive_cost(
    monkeypatch,
):
    records = _paired_decision("s0", "source-0", 3.0) + _paired_decision(
        "s1", "source-1", 1.0
    )
    keys = (("s0", "replicate-000"), ("s1", "replicate-000"))
    monkeypatch.setattr(
        paper_analysis,
        "predict_frozen_factorized_action_values",
        lambda model, records: (
            {keys[0]: "a", keys[1]: "a"},
            {keys[0]: 0.8, keys[1]: 0.2},
        ),
    )
    model = {
        "threshold": 0.5,
        "lambda_cost": 0.05,
        "risk_calibration": {
            "selection_status": "selected_non_degenerate_safe_threshold",
            "selected_threshold": 0.5,
        },
    }
    values, sources, executions, diagnostics = paper_analysis.policy_decision_values(
        model, records
    )
    assert sources == {keys[0]: "source-0", keys[1]: "source-1"}
    assert values[PRIMARY] == {keys[0]: 0.95, keys[1]: 0.0}
    assert values["matched_budget_entropy_gate_learned_crop"] == {
        keys[0]: 0.95,
        keys[1]: 0.0,
    }
    assert values["random_crop_always_call_expected"][keys[0]] == pytest.approx(0.2)
    assert values["ug_exhaustive_entropy_four_calls"][keys[0]] == pytest.approx(0.8)
    assert executions[PRIMARY] == 0.5
    assert executions["ug_exhaustive_entropy_four_calls"] == 4.0
    assert diagnostics["positive_net_action_prevalence"] == 1.0
    assert diagnostics["task_rescue_rate_per_call"] == 1.0
    assert diagnostics["task_harm_rate_per_call"] == 0.0


def test_screenqa_paper_analysis_bundle_is_atomic_hashed_and_exclusive(tmp_path):
    report = {
        "analysis_status": "locked",
        "primary_confirmation": {"passed": False},
        "policies": {
            PRIMARY: {
                "source_utility": 0.001,
                "ci_low": -0.001,
                "ci_high": 0.003,
                "question_utility": 0.001,
                "mean_candidate_executions": 0.02,
            }
        },
        "paired_comparisons": {},
    }
    output = tmp_path / "analysis"
    write_outputs(output, report)
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert _sha(output / name) == expected
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_outputs(output, report)
