from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


AUDIT_SCHEMA = "n1_sibling_bank_inventory_v1"

_PREFIX_KEYS = frozenset({"action_prefix", "action_tokens", "tool_call_prefix"})
_FACTUAL_OBSERVATION_KEYS = frozenset(
    {"factual_observation", "real_observation", "tool_observation"}
)
_COUNTERFACTUAL_OBSERVATION_KEYS = frozenset(
    {"counterfactual_observation", "noop_observation", "alternative_observation"}
)
_CONTINUATION_KEYS = frozenset(
    {"continuation", "continuation_tokens", "continuation_seed"}
)
_TRACKED_INTERVENTION_KEYS = tuple(
    sorted(
        _PREFIX_KEYS
        | _FACTUAL_OBSERVATION_KEYS
        | _COUNTERFACTUAL_OBSERVATION_KEYS
        | _CONTINUATION_KEYS
    )
)


@dataclass(frozen=True)
class SiblingBankSpec:
    name: str
    dataset: str
    role: str
    rollouts: Path
    provenance: Path

    def __post_init__(self) -> None:
        if not self.name or not self.dataset or not self.role:
            raise ValueError("bank name, dataset, and role must be non-empty")


@dataclass
class _DecisionState:
    action_ids: set[str]
    answer_count: int
    zoom_count: int
    generation_seeds: set[int | None]
    identities: set[tuple[str, str, str, str]]


def _nested_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, item in current.items():
                keys.add(str(key))
                if isinstance(item, Mapping):
                    stack.append(item)
                elif isinstance(item, list):
                    stack.extend(
                        element for element in item if isinstance(element, Mapping)
                    )
    return keys


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def _provenance_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"provenance must be a JSON object: {path}")
    invariant = payload.get("invariant_provenance", payload)
    if not isinstance(invariant, Mapping):
        raise ValueError(f"invariant_provenance must be a JSON object: {path}")
    return {
        "model": invariant.get("model"),
        "model_revision": invariant.get("model_revision"),
        "proposer": invariant.get("proposer"),
        "candidate_count": invariant.get("candidate_count"),
        "generation_seeds": invariant.get("generation_seeds"),
        "manifest_sha256": payload.get(
            "manifest_sha256", invariant.get("manifest_sha256")
        ),
        "rollouts_sha256": payload.get(
            "merged_rollouts_sha256", payload.get("output_sha256")
        ),
        "code_revision": invariant.get("code_revision"),
        "scientific_status": payload.get("scientific_status"),
    }


def _read_state_ids(path: Path) -> set[str]:
    state_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping) or not payload.get("state_id"):
                raise ValueError(f"missing state_id at {path}:{line_number}")
            state_ids.add(str(payload["state_id"]))
    return state_ids


def audit_sibling_bank(spec: SiblingBankSpec, *, repo_root: Path) -> dict[str, Any]:
    rollouts = spec.rollouts.resolve()
    provenance = spec.provenance.resolve()
    if not rollouts.is_file():
        raise FileNotFoundError(f"rollout bank does not exist: {rollouts}")
    if not provenance.is_file():
        raise FileNotFoundError(f"rollout provenance does not exist: {provenance}")

    decisions: dict[tuple[str, str], _DecisionState] = {}
    state_replicates: dict[str, set[str]] = defaultdict(set)
    sources: set[str] = set()
    images: set[str] = set()
    action_type_counts: Counter[str] = Counter()
    action_id_counts: Counter[str] = Counter()
    bbox_counts: dict[str, set[tuple[float, ...]]] = defaultdict(set)
    row_models: set[str] = set()
    row_model_revisions: set[str] = set()
    signal_counts: Counter[str] = Counter()
    evidence_use_ready_zoom_rows = 0
    top_level_field_intersection: set[str] | None = None
    top_level_field_union: set[str] = set()
    records = 0

    with rollouts.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {rollouts}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"rollout row must be a JSON object at line {line_number}"
                )
            records += 1

            required = {
                "state_id",
                "replicate_id",
                "action_id",
                "action_type",
                "source_id",
                "image_id",
                "question",
                "original_image",
                "generation_seed",
                "correct_before",
                "correct_after",
            }
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"rollout row {line_number} is missing fields: {sorted(missing)}"
                )

            row_keys = {str(key) for key in row}
            top_level_field_union.update(row_keys)
            if top_level_field_intersection is None:
                top_level_field_intersection = set(row_keys)
            else:
                top_level_field_intersection.intersection_update(row_keys)

            state_id = str(row["state_id"])
            replicate_id = str(row["replicate_id"])
            action_id = str(row["action_id"])
            action_type = str(row["action_type"])
            source_id = str(row["source_id"])
            image_id = str(row["image_id"])
            identity = (
                source_id,
                image_id,
                str(row["question"]),
                str(row["original_image"]),
            )
            seed_value = row.get("generation_seed")
            generation_seed = None if seed_value is None else int(seed_value)
            key = (state_id, replicate_id)
            decision = decisions.setdefault(
                key,
                _DecisionState(set(), 0, 0, set(), set()),
            )
            if action_id in decision.action_ids:
                raise ValueError(
                    f"decision {key!r} contains duplicate action {action_id!r}"
                )
            decision.action_ids.add(action_id)
            decision.answer_count += int(action_type == "ANSWER")
            decision.zoom_count += int(action_type == "ZOOM")
            decision.generation_seeds.add(generation_seed)
            decision.identities.add(identity)
            state_replicates[state_id].add(replicate_id)
            sources.add(source_id)
            images.add(image_id)
            action_type_counts[action_type] += 1
            action_id_counts[action_id] += 1

            candidate_bbox = row.get("candidate_bbox")
            if candidate_bbox is not None:
                if not isinstance(candidate_bbox, list) or len(candidate_bbox) != 4:
                    raise ValueError(f"invalid candidate_bbox at line {line_number}")
                bbox_counts[action_id].add(
                    tuple(float(value) for value in candidate_bbox)
                )

            metadata = row.get("metadata", {})
            if isinstance(metadata, Mapping):
                for backend_name in ("baseline_backend", "action_backend"):
                    backend = metadata.get(backend_name)
                    if isinstance(backend, Mapping):
                        if backend.get("model"):
                            row_models.add(str(backend["model"]))
                        if backend.get("model_revision"):
                            row_model_revisions.add(str(backend["model_revision"]))

            nested_keys = _nested_mapping_keys(row)
            for signal in _TRACKED_INTERVENTION_KEYS:
                if signal in nested_keys:
                    signal_counts[signal] += 1
            if action_type == "ZOOM":
                has_prefix = bool(nested_keys & _PREFIX_KEYS)
                has_factual = bool(nested_keys & _FACTUAL_OBSERVATION_KEYS)
                has_counterfactual = bool(
                    nested_keys & _COUNTERFACTUAL_OBSERVATION_KEYS
                )
                has_continuation = bool(nested_keys & _CONTINUATION_KEYS)
                evidence_use_ready_zoom_rows += int(
                    has_prefix
                    and has_factual
                    and has_counterfactual
                    and has_continuation
                )

    if records == 0:
        raise ValueError(f"rollout bank is empty: {rollouts}")

    malformed_decisions = sum(
        decision.answer_count != 1
        or decision.zoom_count < 1
        or len(decision.generation_seeds) != 1
        or len(decision.identities) != 1
        for decision in decisions.values()
    )
    action_set_counts = Counter(
        ",".join(sorted(decision.action_ids)) for decision in decisions.values()
    )
    candidate_count_counts = Counter(
        decision.zoom_count for decision in decisions.values()
    )
    replicates_per_state_counts = Counter(
        len(replicates) for replicates in state_replicates.values()
    )
    zoom_records = action_type_counts["ZOOM"]
    provenance_contract = _provenance_contract(provenance)
    checks = {
        "all_decisions_have_complete_siblings": malformed_decisions == 0,
        "one_registered_action_set": len(action_set_counts) == 1,
        "one_candidate_count": len(candidate_count_counts) == 1,
        "source_ids_available": len(sources) > 0,
        "image_ids_available": len(images) > 0,
        "row_model_matches_provenance": (
            provenance_contract["model"] in row_models
            and provenance_contract["model_revision"] in row_model_revisions
        ),
        "immutable_rollout_hash_available": bool(
            provenance_contract["rollouts_sha256"]
        ),
        "immutable_manifest_hash_available": bool(
            provenance_contract["manifest_sha256"]
        ),
    }
    return {
        "name": spec.name,
        "dataset": spec.dataset,
        "role": spec.role,
        "rollouts": str(rollouts.relative_to(repo_root.resolve())),
        "provenance": str(provenance.relative_to(repo_root.resolve())),
        "records": records,
        "decisions": len(decisions),
        "states": len(state_replicates),
        "sources": len(sources),
        "images": len(images),
        "action_type_counts": _counter_dict(action_type_counts),
        "action_id_counts": _counter_dict(action_id_counts),
        "action_set_counts": _counter_dict(action_set_counts),
        "candidate_count_counts": _counter_dict(candidate_count_counts),
        "replicates_per_state_counts": _counter_dict(replicates_per_state_counts),
        "unique_bboxes_per_action": {
            action_id: len(boxes) for action_id, boxes in sorted(bbox_counts.items())
        },
        "row_models": sorted(row_models),
        "row_model_revisions": sorted(row_model_revisions),
        "intervention_signal_row_counts": {
            signal: signal_counts[signal] for signal in _TRACKED_INTERVENTION_KEYS
        },
        "evidence_use_ready_zoom_rows": evidence_use_ready_zoom_rows,
        "top_level_field_intersection": sorted(top_level_field_intersection or set()),
        "top_level_field_union": sorted(top_level_field_union),
        "malformed_decisions": malformed_decisions,
        "provenance_contract": provenance_contract,
        "checks": checks,
    }


def build_n1_inventory(
    specs: Sequence[SiblingBankSpec], *, repo_root: Path
) -> dict[str, Any]:
    if not specs:
        raise ValueError("at least one sibling bank is required")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("sibling bank names must be unique")

    banks = [audit_sibling_bank(spec, repo_root=repo_root) for spec in specs]
    main_banks = [bank for bank in banks if bank["role"] == "main_development"]
    diagnostic_banks = [bank for bank in banks if bank["role"] != "main_development"]
    main_datasets = sorted({str(bank["dataset"]) for bank in main_banks})
    main_models = sorted(
        {
            str(bank["provenance_contract"]["model"])
            for bank in main_banks
            if bank["provenance_contract"]["model"]
        }
    )
    main_proposers = sorted(
        {
            str(bank["provenance_contract"]["proposer"])
            for bank in main_banks
            if bank["provenance_contract"]["proposer"]
        }
    )
    dataset_to_main_models: dict[str, set[str]] = defaultdict(set)
    for bank in main_banks:
        model = bank["provenance_contract"]["model"]
        if model:
            dataset_to_main_models[str(bank["dataset"])].add(str(model))

    all_main_schema_complete = bool(main_banks) and all(
        all(bank["checks"].values()) for bank in main_banks
    )
    all_main_single_replicate = bool(main_banks) and all(
        bank["replicates_per_state_counts"] == {"1": bank["states"]}
        for bank in main_banks
    )
    all_main_multiple_replicates = bool(main_banks) and all(
        min(int(count) for count in bank["replicates_per_state_counts"]) >= 2
        for bank in main_banks
    )
    all_main_evidence_ready = bool(main_banks) and all(
        bank["evidence_use_ready_zoom_rows"]
        == bank["action_type_counts"].get("ZOOM", 0)
        for bank in main_banks
    )
    same_dataset_main_backbone_factor = any(
        len(models) >= 2 for models in dataset_to_main_models.values()
    )
    main_tool_action_types = sorted(
        {
            action_type
            for bank in main_banks
            for action_type in bank["action_type_counts"]
            if action_type != "ANSWER"
        }
    )
    cross_backbone_same_dataset_comparisons: list[dict[str, Any]] = []
    state_ids_by_name: dict[str, set[str]] = {}
    for left_index, left in enumerate(banks):
        for right in banks[left_index + 1 :]:
            if left["dataset"] != right["dataset"]:
                continue
            left_model = left["provenance_contract"]["model"]
            right_model = right["provenance_contract"]["model"]
            if not left_model or not right_model or left_model == right_model:
                continue
            for bank in (left, right):
                if bank["name"] not in state_ids_by_name:
                    spec = next(item for item in specs if item.name == bank["name"])
                    state_ids_by_name[bank["name"]] = _read_state_ids(spec.rollouts)
            overlap = state_ids_by_name[left["name"]] & state_ids_by_name[right["name"]]
            cross_backbone_same_dataset_comparisons.append(
                {
                    "dataset": left["dataset"],
                    "left_bank": left["name"],
                    "left_role": left["role"],
                    "left_model": left_model,
                    "right_bank": right["name"],
                    "right_role": right["role"],
                    "right_model": right_model,
                    "overlapping_states": len(overlap),
                }
            )

    estimands = {
        "stop_regret": {
            "identifiable": all_main_schema_complete,
            "reason": (
                "Every complete decision contains answer-now and observed tool outcomes."
            ),
        },
        "action_selection_regret_within_registered_bank": {
            "identifiable": all_main_schema_complete,
            "reason": (
                "Every complete decision contains outcomes for every registered sibling action."
            ),
        },
        "evidence_use_regret": {
            "identifiable": all_main_evidence_ready,
            "reason": (
                "Requires a fixed action prefix plus matched factual and counterfactual "
                "observations and a controlled continuation; those fields are absent."
            ),
        },
    }
    gate_checks = {
        "at_least_three_main_datasets": len(main_datasets) >= 3,
        "all_main_banks_schema_complete": all_main_schema_complete,
        "source_level_identifiers_available": bool(main_banks)
        and all(bank["sources"] > 0 for bank in main_banks),
        "immutable_reproduction_metadata_available": bool(main_banks)
        and all(
            bank["checks"]["immutable_rollout_hash_available"]
            and bank["checks"]["immutable_manifest_hash_available"]
            and bool(bank["provenance_contract"]["model_revision"])
            and bool(bank["provenance_contract"]["code_revision"])
            for bank in main_banks
        ),
        "same_dataset_multibackbone_main_factor": same_dataset_main_backbone_factor,
        "more_than_one_tool_action_family": len(main_tool_action_types) >= 2,
        "multiple_stochastic_replicates_per_state": all_main_multiple_replicates,
        "stop_regret_identifiable": estimands["stop_regret"]["identifiable"],
        "action_selection_regret_identifiable": estimands[
            "action_selection_regret_within_registered_bank"
        ]["identifiable"],
        "evidence_use_regret_identifiable": estimands["evidence_use_regret"][
            "identifiable"
        ],
    }
    decision = (
        "n1_existing_assets_pass_top_tier_regret_benchmark_gate"
        if all(gate_checks.values())
        else "n1_existing_assets_insufficient_for_top_tier_regret_benchmark"
    )
    return {
        "schema": AUDIT_SCHEMA,
        "banks": banks,
        "summary": {
            "main_bank_count": len(main_banks),
            "diagnostic_bank_count": len(diagnostic_banks),
            "main_datasets": main_datasets,
            "main_models": main_models,
            "main_proposers": main_proposers,
            "main_records": sum(int(bank["records"]) for bank in main_banks),
            "main_decisions": sum(int(bank["decisions"]) for bank in main_banks),
            "main_sources_summed_without_cross_dataset_deduplication": sum(
                int(bank["sources"]) for bank in main_banks
            ),
            "dataset_to_main_models": {
                dataset: sorted(models)
                for dataset, models in sorted(dataset_to_main_models.items())
            },
            "main_tool_action_types": main_tool_action_types,
            "all_main_single_replicate": all_main_single_replicate,
            "cross_backbone_same_dataset_comparisons": (
                cross_backbone_same_dataset_comparisons
            ),
        },
        "estimand_identifiability": estimands,
        "gate_checks": gate_checks,
        "decision": decision,
    }


def default_n1_bank_specs(repo_root: Path) -> tuple[SiblingBankSpec, ...]:
    root = repo_root.resolve()
    entries = (
        (
            "infographicvqa_qwen7b_full",
            "InfographicVQA",
            "main_development",
            "artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1/merged-rollouts/rollouts.jsonl",
            "artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1/merged-rollouts/rollouts.merge.json",
        ),
        (
            "screenqa_qwen3b_full",
            "ScreenQA",
            "main_development",
            "artifacts/screenqa-train-factorized-v1/ranker-rollouts-v1/merged/rollouts.jsonl",
            "artifacts/screenqa-train-factorized-v1/ranker-rollouts-v1/merged/rollouts.merge.json",
        ),
        (
            "docvqa_qwen3b_full",
            "DocVQA",
            "main_development",
            "artifacts/docvqa-train-factorized-v2/ranker-training/qwen3b-c4-seed0/rollouts.jsonl",
            "artifacts/docvqa-train-factorized-v2/ranker-training/qwen3b-c4-seed0/rollouts.provenance.json",
        ),
        (
            "textvqa_qwen3b_full",
            "TextVQA",
            "main_development",
            "artifacts/textvqa-train-scale-v1/ranker-training/qwen3b-c4-seed0/rollouts.jsonl",
            "artifacts/textvqa-train-scale-v1/ranker-training/qwen3b-c4-seed0/rollouts.provenance.json",
        ),
        (
            "screenqa_qwen7b_512",
            "ScreenQA",
            "opened_development_diagnostic",
            "artifacts/screenqa-train-factorized-v1/backbone-7b-diagnostic-v1/full-h800-v1/merged-rollouts/rollouts.jsonl",
            "artifacts/screenqa-train-factorized-v1/backbone-7b-diagnostic-v1/full-h800-v1/merged-rollouts/rollouts.merge.json",
        ),
        (
            "chartqa_qwen3b_4500",
            "ChartQA",
            "diagnostic_replication",
            "artifacts/replication-chartqa-train-4500/qwen3b-c4-concise-seed0/rollouts.jsonl",
            "artifacts/replication-chartqa-train-4500/qwen3b-c4-concise-seed0/rollouts.provenance.json",
        ),
    )
    return tuple(
        SiblingBankSpec(name, dataset, role, root / rollouts, root / provenance)
        for name, dataset, role, rollouts, provenance in entries
    )
