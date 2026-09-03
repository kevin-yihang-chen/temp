from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from beyond_entropy.sibling_bank_inventory import (
    AUDIT_SCHEMA,
    SiblingBankSpec,
    audit_sibling_bank,
    build_n1_inventory,
)


def _row(
    *,
    state: str,
    action_id: str,
    action_type: str,
    model: str,
    revision: str,
    evidence_ready: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "state_id": state,
        "replicate_id": "replicate-000",
        "action_id": action_id,
        "action_type": action_type,
        "source_id": f"source-{state}",
        "image_id": f"image-{state}",
        "question": f"question-{state}",
        "original_image": f"{state}.png",
        "generation_seed": 0,
        "candidate_bbox": None if action_type == "ANSWER" else [0.0, 0.0, 0.5, 0.5],
        "entropy_before": 0.5,
        "entropy_after": 0.5,
        "answer_before": "a",
        "answer_after": "a",
        "correct_before": 0.0,
        "correct_after": 1.0 if action_type == "ZOOM" else 0.0,
        "tool_cost": 0.0 if action_type == "ANSWER" else 1.0,
        "pre_action_features": {},
        "metadata": {"baseline_backend": {"model": model, "model_revision": revision}},
    }
    if action_type == "ZOOM":
        row["metadata"]["action_backend"] = {
            "model": model,
            "model_revision": revision,
        }
        if evidence_ready:
            row.update(
                {
                    "action_prefix": "call crop",
                    "factual_observation": "real crop",
                    "counterfactual_observation": "blank crop",
                    "continuation_seed": 17,
                }
            )
    return row


def _bank(
    root: Path,
    *,
    name: str,
    dataset: str,
    model: str,
    revision: str,
    role: str = "main_development",
    evidence_ready: bool = False,
) -> SiblingBankSpec:
    directory = root / name
    directory.mkdir()
    rollouts = directory / "rollouts.jsonl"
    rows = []
    for state in ("s0", "s1"):
        rows.extend(
            [
                _row(
                    state=state,
                    action_id="answer-now",
                    action_type="ANSWER",
                    model=model,
                    revision=revision,
                ),
                _row(
                    state=state,
                    action_id="ug-grid-00",
                    action_type="ZOOM",
                    model=model,
                    revision=revision,
                    evidence_ready=evidence_ready,
                ),
                _row(
                    state=state,
                    action_id="ug-grid-01",
                    action_type="ZOOM",
                    model=model,
                    revision=revision,
                    evidence_ready=evidence_ready,
                ),
            ]
        )
    rollouts.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    provenance = directory / "rollouts.provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "model": model,
                "model_revision": revision,
                "proposer": "ug-grid",
                "candidate_count": 2,
                "generation_seeds": [0],
                "manifest_sha256": "a" * 64,
                "output_sha256": "b" * 64,
                "code_revision": "c" * 40,
            }
        ),
        encoding="utf-8",
    )
    return SiblingBankSpec(name, dataset, role, rollouts, provenance)


def test_bank_inventory_counts_complete_siblings_and_missing_interventions(
    tmp_path: Path,
) -> None:
    spec = _bank(
        tmp_path,
        name="bank-a",
        dataset="A",
        model="model-a",
        revision="rev-a",
    )
    audit = audit_sibling_bank(spec, repo_root=tmp_path)
    assert audit["records"] == 6
    assert audit["decisions"] == 2
    assert audit["candidate_count_counts"] == {"2": 2}
    assert audit["replicates_per_state_counts"] == {"1": 2}
    assert audit["evidence_use_ready_zoom_rows"] == 0
    assert all(audit["checks"].values())


def test_n1_gate_rejects_large_complete_banks_without_causal_contract(
    tmp_path: Path,
) -> None:
    specs = tuple(
        _bank(
            tmp_path,
            name=f"bank-{index}",
            dataset=dataset,
            model="model-a" if index < 2 else "model-b",
            revision=f"rev-{index}",
        )
        for index, dataset in enumerate(("A", "B", "C"))
    )
    report = build_n1_inventory(specs, repo_root=tmp_path)
    assert report["schema"] == AUDIT_SCHEMA
    assert report["summary"]["main_decisions"] == 6
    assert report["estimand_identifiability"]["stop_regret"]["identifiable"]
    assert report["estimand_identifiability"][
        "action_selection_regret_within_registered_bank"
    ]["identifiable"]
    assert not report["estimand_identifiability"]["evidence_use_regret"]["identifiable"]
    assert report["decision"] == (
        "n1_existing_assets_insufficient_for_top_tier_regret_benchmark"
    )


def test_evidence_use_contract_is_detected_on_every_zoom_row(tmp_path: Path) -> None:
    spec = _bank(
        tmp_path,
        name="ready",
        dataset="A",
        model="model-a",
        revision="rev-a",
        evidence_ready=True,
    )
    audit = audit_sibling_bank(spec, repo_root=tmp_path)
    assert audit["evidence_use_ready_zoom_rows"] == 4
    assert audit["intervention_signal_row_counts"]["action_prefix"] == 4
    assert audit["intervention_signal_row_counts"]["continuation_seed"] == 4


def test_inventory_rejects_duplicate_action_within_decision(tmp_path: Path) -> None:
    spec = _bank(
        tmp_path,
        name="duplicate",
        dataset="A",
        model="model-a",
        revision="rev-a",
    )
    rows = spec.rollouts.read_text(encoding="utf-8").splitlines()
    duplicate = json.loads(rows[1])
    rows[2] = json.dumps(duplicate)
    spec.rollouts.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate action"):
        audit_sibling_bank(spec, repo_root=tmp_path)


def test_inventory_rejects_duplicate_bank_names(tmp_path: Path) -> None:
    spec = _bank(
        tmp_path,
        name="same",
        dataset="A",
        model="model-a",
        revision="rev-a",
    )
    with pytest.raises(ValueError, match="names must be unique"):
        build_n1_inventory((spec, spec), repo_root=tmp_path)
