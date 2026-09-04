from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .predictability_audit import (
    AUDIT_BENCHMARKS,
    AUDIT_SEEDS,
    PREDICTOR_LEVELS,
    TARGET_FAMILIES,
    BinaryToolOutcome,
    SplitIdentity,
    SplitRole,
    audit_split_disjointness,
    collapse_fixed_entropy_tool,
    fixed_tool_headroom_summary,
    matrix_completion_report,
)
from .predictability_baselines import (
    FrozenStrongBaselinePolicy,
    apply_strong_baselines,
    fit_strong_baselines,
    strong_baseline_report,
    trace_by_name,
    validate_fixed_tool_outcomes,
)
from .predictability_evaluation import (
    align_policy_outcomes,
    calls_at_threshold,
    paired_source_bootstrap_policy_difference,
    policy_curve,
    policy_metrics,
    prediction_metrics,
)
from .predictability_modeling import (
    AuditExample,
    FrozenAuditCell,
    evaluate_frozen_audit_cell,
    fit_frozen_audit_cell,
)
from .predictability_post_action import (
    FrozenPostActionProbe,
    PostActionProbeExample,
    evaluate_frozen_post_action_probe,
    fit_frozen_post_action_probe,
)
from .schema import ActionRecord


STRONG_BASELINE_RANDOM_SEED = 20260903
FROZEN_MATRIX_FORMAT_VERSION = 2
CellKey = tuple[str, str, int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _commit_file_exclusive(staging: Path, destination: Path) -> None:
    """Atomically publish a complete file without overwriting a competitor."""

    linked = False
    try:
        os.link(staging, destination)
        linked = True
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if linked:
            destination.unlink(missing_ok=True)
        raise
    finally:
        staging.unlink(missing_ok=True)


def _validate_matrix_configuration(
    *,
    seeds: Sequence[int],
    predictor_levels: Sequence[str],
    target_families: Sequence[str],
    strong_baseline_random_seed: int,
    formal_claim_eligible: bool,
) -> None:
    if not seeds or len(seeds) > 3 or len(set(seeds)) != len(seeds):
        raise ValueError("matrix runner requires one to three unique seeds")
    if set(predictor_levels) - set(PREDICTOR_LEVELS):
        raise ValueError("matrix runner received an unregistered predictor level")
    if set(target_families) - set(TARGET_FAMILIES):
        raise ValueError("matrix runner received an unregistered target family")
    if formal_claim_eligible and (
        tuple(seeds) != AUDIT_SEEDS
        or tuple(predictor_levels) != PREDICTOR_LEVELS
        or tuple(target_families) != TARGET_FAMILIES
        or strong_baseline_random_seed != STRONG_BASELINE_RANDOM_SEED
    ):
        raise ValueError(
            "formal matrix requires the complete frozen levels, targets, and seeds"
        )


def _majority_calls(call_rows: Sequence[Sequence[bool]]) -> tuple[bool, ...]:
    """Aggregate fixed-seed policies, breaking an even tie toward no call."""

    if not call_rows:
        raise ValueError("majority aggregation requires at least one policy")
    width = len(call_rows[0])
    if width <= 0 or any(len(row) != width for row in call_rows):
        raise ValueError("majority policies must be aligned and non-empty")
    if any(not isinstance(call, bool) for row in call_rows for call in row):
        raise ValueError("majority policies must contain booleans")
    required = len(call_rows) // 2 + 1
    return tuple(
        sum(bool(row[index]) for row in call_rows) >= required for index in range(width)
    )


def _select_validation_cell_keys(
    cells: Sequence[FrozenAuditCell],
    *,
    seeds: Sequence[int],
    predictor_levels: Sequence[str],
    target_families: Sequence[str],
    required_level: str | None = None,
) -> tuple[CellKey, ...]:
    """Select exactly one validation-best registered cell for each fixed seed."""

    level_order = {name: index for index, name in enumerate(predictor_levels)}
    target_order = {name: index for index, name in enumerate(target_families)}
    selected: list[CellKey] = []
    for seed in seeds:
        candidates = [
            cell
            for cell in cells
            if cell.seed == seed
            and (required_level is None or cell.level == required_level)
        ]
        if not candidates:
            raise ValueError(f"no validation candidate exists for seed {seed}")

        def key(cell: FrozenAuditCell) -> tuple[float, int, int, int]:
            return (
                -float(cell.validation_metrics["validation_utility"]),
                int(cell.validation_metrics["validation_calls"]),
                level_order[cell.level],
                target_order[cell.target],
            )

        cell = min(candidates, key=key)
        selected.append((cell.level, cell.target, cell.seed))
    return tuple(selected)


def _validation_calls_for_cells(
    cells: Sequence[FrozenAuditCell],
    keys: Sequence[CellKey],
    validation: Sequence[AuditExample],
    reference_outcomes: Sequence[BinaryToolOutcome],
) -> tuple[bool, ...]:
    cell_map = {(cell.level, cell.target, cell.seed): cell for cell in cells}
    calls = _majority_calls(
        [
            calls_at_threshold(
                cell_map[key].predict(validation), cell_map[key].threshold
            )
            for key in keys
        ]
    )
    _, aligned = align_policy_outcomes(
        reference_outcomes,
        [item.outcome for item in validation],
        calls,
    )
    return tuple(aligned)


def _post_action_validation_calls(
    probes: Sequence[FrozenPostActionProbe],
    validation: Sequence[PostActionProbeExample],
    reference_outcomes: Sequence[BinaryToolOutcome],
) -> tuple[bool, ...]:
    calls = _majority_calls(
        [
            calls_at_threshold(probe.predict(validation), probe.threshold)
            for probe in probes
        ]
    )
    _, aligned = align_policy_outcomes(
        reference_outcomes,
        [item.outcome for item in validation],
        calls,
    )
    return tuple(aligned)


def _validate_pre_post_role(
    role: str,
    pre_action: Sequence[AuditExample],
    post_action: Sequence[PostActionProbeExample],
) -> None:
    if not pre_action or not post_action:
        raise ValueError(f"{role} pre- and post-action roles must be non-empty")
    pre = {item.outcome.decision_id: item for item in pre_action}
    post = {item.outcome.decision_id: item for item in post_action}
    if len(pre) != len(pre_action) or len(post) != len(post_action):
        raise ValueError(f"{role} predictability decision IDs must be unique")
    if set(pre) != set(post):
        raise ValueError(f"{role} pre- and post-action coverage differs")
    for decision_id, item in pre.items():
        counterpart = post[decision_id]
        if (
            item.outcome != counterpart.outcome
            or item.image_rgb_sha256 != counterpart.image_rgb_sha256
        ):
            raise ValueError(f"{role} pre- and post-action labels or images differ")
        for level in PREDICTOR_LEVELS:
            item.inputs.feature_vector(level)
        counterpart.inputs.feature_vector()


def _role_identities(
    role: SplitRole, examples: Sequence[AuditExample]
) -> tuple[list[SplitIdentity], dict[str, SplitRole]]:
    identities: list[SplitIdentity] = []
    assignments: dict[str, SplitRole] = {}
    for index, example in enumerate(examples):
        item_id = (
            f"{role}:{index}:{example.outcome.state_id}:"
            f"{example.outcome.replicate_id}"
        )
        identities.append(
            SplitIdentity(
                item_id=item_id,
                source_id=example.outcome.source_id,
                image_rgb_sha256=example.image_rgb_sha256,
            )
        )
        assignments[item_id] = role
    return identities, assignments


@dataclass(frozen=True)
class DevelopmentIdentityIndex:
    sources: tuple[str, ...]
    image_rgb_sha256: tuple[str, ...]
    decisions: int

    def __post_init__(self) -> None:
        if not self.sources or not self.image_rgb_sha256 or self.decisions <= 0:
            raise ValueError("development identity index must be non-empty")
        if tuple(sorted(set(self.sources))) != self.sources:
            raise ValueError("development sources must be sorted and unique")
        if tuple(sorted(set(self.image_rgb_sha256))) != self.image_rgb_sha256:
            raise ValueError("development RGB hashes must be sorted and unique")

    @classmethod
    def from_examples(
        cls, examples: Sequence[AuditExample]
    ) -> "DevelopmentIdentityIndex":
        return cls(
            sources=tuple(sorted({item.outcome.source_id for item in examples})),
            image_rgb_sha256=tuple(
                sorted({item.image_rgb_sha256 for item in examples})
            ),
            decisions=len(examples),
        )

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                "sources": self.sources,
                "image_rgb_sha256": self.image_rgb_sha256,
                "decisions": self.decisions,
            }
        )


@dataclass(frozen=True)
class BenchmarkDevelopmentData:
    """Train/validation-only input accepted by the freeze phase."""

    train: Sequence[AuditExample]
    validation: Sequence[AuditExample]
    post_action_train: Sequence[PostActionProbeExample]
    post_action_validation: Sequence[PostActionProbeExample]
    validation_siblings: Sequence[ActionRecord]

    def validate(self) -> dict[str, Any]:
        _validate_pre_post_role("train", self.train, self.post_action_train)
        _validate_pre_post_role(
            "validation", self.validation, self.post_action_validation
        )
        if not self.validation_siblings:
            raise ValueError("validation strong-baseline sibling records are empty")
        validate_fixed_tool_outcomes(
            [item.outcome for item in self.validation],
            collapse_fixed_entropy_tool(self.validation_siblings),
        )
        train_identities, train_assignments = _role_identities("train", self.train)
        validation_identities, validation_assignments = _role_identities(
            "validation", self.validation
        )
        return audit_split_disjointness(
            train_identities + validation_identities,
            {**train_assignments, **validation_assignments},
        )

    def identity_index(self) -> DevelopmentIdentityIndex:
        return DevelopmentIdentityIndex.from_examples(
            tuple(self.train) + tuple(self.validation)
        )


@dataclass(frozen=True)
class BenchmarkTestData:
    """Held-out input accepted only by the evaluation phase."""

    test: Sequence[AuditExample]
    post_action_test: Sequence[PostActionProbeExample]
    test_siblings: Sequence[ActionRecord]

    def validate(self) -> dict[str, Any]:
        _validate_pre_post_role("test", self.test, self.post_action_test)
        if not self.test_siblings:
            raise ValueError("test strong-baseline sibling records are empty")
        validate_fixed_tool_outcomes(
            [item.outcome for item in self.test],
            collapse_fixed_entropy_tool(self.test_siblings),
        )
        return {
            "schema": "predictability_test_role_validation_v1",
            "passed": True,
            "decisions": len(self.test),
            "sources": len({item.outcome.source_id for item in self.test}),
            "decoded_rgb": len({item.image_rgb_sha256 for item in self.test}),
        }


@dataclass(frozen=True)
class BenchmarkAuditData:
    """Convenience container retained for dependency-light synthetic smoke tests."""

    train: Sequence[AuditExample]
    validation: Sequence[AuditExample]
    test: Sequence[AuditExample]
    post_action_train: Sequence[PostActionProbeExample]
    post_action_validation: Sequence[PostActionProbeExample]
    post_action_test: Sequence[PostActionProbeExample]
    validation_siblings: Sequence[ActionRecord]
    test_siblings: Sequence[ActionRecord]

    def development(self) -> BenchmarkDevelopmentData:
        return BenchmarkDevelopmentData(
            train=self.train,
            validation=self.validation,
            post_action_train=self.post_action_train,
            post_action_validation=self.post_action_validation,
            validation_siblings=self.validation_siblings,
        )

    def held_out_test(self) -> BenchmarkTestData:
        return BenchmarkTestData(
            test=self.test,
            post_action_test=self.post_action_test,
            test_siblings=self.test_siblings,
        )

    def validate(self) -> dict[str, Any]:
        self.development().validate()
        self.held_out_test().validate()
        identities: list[SplitIdentity] = []
        assignments: dict[str, SplitRole] = {}
        role_examples: tuple[tuple[SplitRole, Sequence[AuditExample]], ...] = (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        )
        for role, examples in role_examples:
            role_identities, role_assignments = _role_identities(role, examples)
            identities.extend(role_identities)
            assignments.update(role_assignments)
        return audit_split_disjointness(identities, assignments)


@dataclass(frozen=True)
class FrozenBenchmarkAudit:
    baseline: FrozenStrongBaselinePolicy
    post_action_probes: tuple[FrozenPostActionProbe, ...]
    cells: tuple[FrozenAuditCell, ...]
    selected_deployable_cell_keys: tuple[tuple[str, str, int], ...]
    selected_l3_cell_keys: tuple[tuple[str, str, int], ...]
    selected_deployable_validation_calls: tuple[bool, ...]
    selected_l3_validation_calls: tuple[bool, ...]
    post_action_validation_calls: tuple[bool, ...]
    development_identities: DevelopmentIdentityIndex
    development_split_audit: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenPredictabilityMatrix:
    """All validation-selected state; this object contains no test examples."""

    format_version: int
    lambda_cost: float
    seeds: tuple[int, ...]
    predictor_levels: tuple[str, ...]
    target_families: tuple[str, ...]
    strong_baseline_random_seed: int
    formal_claim_eligible: bool
    benchmarks: Mapping[str, FrozenBenchmarkAudit]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.format_version != FROZEN_MATRIX_FORMAT_VERSION:
            raise ValueError("unsupported frozen matrix format")
        _validate_matrix_configuration(
            seeds=self.seeds,
            predictor_levels=self.predictor_levels,
            target_families=self.target_families,
            strong_baseline_random_seed=self.strong_baseline_random_seed,
            formal_claim_eligible=self.formal_claim_eligible,
        )
        if set(self.benchmarks) != set(AUDIT_BENCHMARKS):
            raise ValueError("frozen matrix requires exactly the three benchmarks")
        expected_cells = {
            (level, target, seed)
            for level in self.predictor_levels
            for target in self.target_families
            for seed in self.seeds
        }
        for benchmark, frozen in self.benchmarks.items():
            if frozen.baseline.lambda_cost != self.lambda_cost:
                raise ValueError(f"{benchmark} baseline lambda differs from matrix")
            if tuple(item.seed for item in frozen.post_action_probes) != self.seeds:
                raise ValueError(f"{benchmark} post-action probe seeds differ")
            actual_cells = {
                (item.level, item.target, item.seed) for item in frozen.cells
            }
            if actual_cells != expected_cells or len(frozen.cells) != len(
                expected_cells
            ):
                raise ValueError(f"{benchmark} frozen audit cells are incomplete")
            if (
                tuple(key[2] for key in frozen.selected_deployable_cell_keys)
                != self.seeds
            ):
                raise ValueError(f"{benchmark} selected deployable seeds differ")
            expected_l3_seeds = (
                self.seeds if "l3_frozen_qwen" in self.predictor_levels else ()
            )
            if (
                tuple(key[2] for key in frozen.selected_l3_cell_keys)
                != expected_l3_seeds
            ):
                raise ValueError(f"{benchmark} selected L3 seeds differ")
            if any(
                key not in expected_cells
                for key in frozen.selected_deployable_cell_keys
            ):
                raise ValueError(f"{benchmark} selected deployable cell is absent")
            if any(
                key not in expected_cells or key[0] != "l3_frozen_qwen"
                for key in frozen.selected_l3_cell_keys
            ):
                raise ValueError(f"{benchmark} selected L3 cell is absent")
            validation_count = len(frozen.baseline.validation_traces[0].outcomes)
            for calls in (
                frozen.selected_deployable_validation_calls,
                frozen.post_action_validation_calls,
            ):
                if len(calls) != validation_count or not all(
                    isinstance(call, bool) for call in calls
                ):
                    raise ValueError(f"{benchmark} frozen validation calls differ")
            if expected_l3_seeds:
                if len(frozen.selected_l3_validation_calls) != validation_count:
                    raise ValueError(f"{benchmark} frozen L3 validation calls differ")
            elif frozen.selected_l3_validation_calls:
                raise ValueError(f"{benchmark} has unexpected frozen L3 calls")
        try:
            json.dumps(self.provenance, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("frozen matrix provenance must be strict JSON") from exc


def fit_predictability_matrix(
    datasets: Mapping[str, BenchmarkDevelopmentData],
    *,
    lambda_cost: float,
    seeds: Sequence[int] = AUDIT_SEEDS,
    predictor_levels: Sequence[str] = PREDICTOR_LEVELS,
    target_families: Sequence[str] = TARGET_FAMILIES,
    strong_baseline_random_seed: int = STRONG_BASELINE_RANDOM_SEED,
    formal_claim_eligible: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> FrozenPredictabilityMatrix:
    """Fit and freeze every validation-selected choice without accepting test data."""

    if set(datasets) != set(AUDIT_BENCHMARKS):
        raise ValueError("matrix freeze requires exactly ChartQA, DocVQA, and HRBench")
    _validate_matrix_configuration(
        seeds=seeds,
        predictor_levels=predictor_levels,
        target_families=target_families,
        strong_baseline_random_seed=strong_baseline_random_seed,
        formal_claim_eligible=formal_claim_eligible,
    )
    frozen_benchmarks: dict[str, FrozenBenchmarkAudit] = {}
    for benchmark in AUDIT_BENCHMARKS:
        data = datasets[benchmark]
        split_audit = data.validate()
        baseline = fit_strong_baselines(
            data.validation_siblings,
            lambda_cost=lambda_cost,
            random_gate_seed=strong_baseline_random_seed,
        )
        probes = tuple(
            fit_frozen_post_action_probe(
                data.post_action_train,
                data.post_action_validation,
                seed=seed,
                lambda_cost=lambda_cost,
            )
            for seed in seeds
        )
        cells = tuple(
            fit_frozen_audit_cell(
                data.train,
                data.validation,
                level=level,
                target=target,
                seed=seed,
                lambda_cost=lambda_cost,
            )
            for level in predictor_levels
            for target in target_families
            for seed in seeds
        )
        selected_deployable = _select_validation_cell_keys(
            cells,
            seeds=seeds,
            predictor_levels=predictor_levels,
            target_families=target_families,
        )
        selected_l3 = (
            _select_validation_cell_keys(
                cells,
                seeds=seeds,
                predictor_levels=predictor_levels,
                target_families=target_families,
                required_level="l3_frozen_qwen",
            )
            if "l3_frozen_qwen" in predictor_levels
            else ()
        )
        validation_reference = trace_by_name(
            baseline.validation_traces, "answer_now"
        ).outcomes
        frozen_benchmarks[benchmark] = FrozenBenchmarkAudit(
            baseline=baseline,
            post_action_probes=probes,
            cells=cells,
            selected_deployable_cell_keys=selected_deployable,
            selected_l3_cell_keys=selected_l3,
            selected_deployable_validation_calls=_validation_calls_for_cells(
                cells, selected_deployable, data.validation, validation_reference
            ),
            selected_l3_validation_calls=(
                _validation_calls_for_cells(
                    cells, selected_l3, data.validation, validation_reference
                )
                if selected_l3
                else ()
            ),
            post_action_validation_calls=_post_action_validation_calls(
                probes, data.post_action_validation, validation_reference
            ),
            development_identities=data.identity_index(),
            development_split_audit=split_audit,
        )
    safe_provenance = json.loads(
        json.dumps(dict(provenance or {}), allow_nan=False, sort_keys=True)
    )
    return FrozenPredictabilityMatrix(
        format_version=FROZEN_MATRIX_FORMAT_VERSION,
        lambda_cost=float(lambda_cost),
        seeds=tuple(seeds),
        predictor_levels=tuple(predictor_levels),
        target_families=tuple(target_families),
        strong_baseline_random_seed=strong_baseline_random_seed,
        formal_claim_eligible=formal_claim_eligible,
        benchmarks=frozen_benchmarks,
        provenance=safe_provenance,
    )


def frozen_predictability_matrix_report(
    frozen: FrozenPredictabilityMatrix,
) -> dict[str, Any]:
    """Return a strict-JSON inventory proving what was selected before test."""

    benchmarks: dict[str, Any] = {}
    for benchmark in AUDIT_BENCHMARKS:
        item = frozen.benchmarks[benchmark]
        benchmarks[benchmark] = {
            "development_identity_sha256": item.development_identities.sha256,
            "development_decisions": item.development_identities.decisions,
            "development_sources": len(item.development_identities.sources),
            "development_decoded_rgb": len(
                item.development_identities.image_rgb_sha256
            ),
            "development_split_audit": item.development_split_audit,
            "strong_baseline": {
                "strongest_name": item.baseline.strongest_name,
                "entropy_gate_threshold": item.baseline.entropy_gate_threshold,
                "random_gate_threshold": item.baseline.random_gate_threshold,
                "fixed_crop_action_id": item.baseline.fixed_crop_action_id,
                "validation": {
                    trace.name: trace.metrics(lambda_cost=frozen.lambda_cost)
                    for trace in item.baseline.validation_traces
                },
            },
            "post_action_probes": [
                {
                    "seed": probe.seed,
                    "threshold": probe.threshold,
                    "input_dimension": probe.input_dimension,
                    "validation": probe.validation_metrics,
                }
                for probe in item.post_action_probes
            ],
            "frozen_policy_selection": {
                "rule": (
                    "per_seed_max_validation_utility_then_fewer_calls_then_"
                    "predictor_level_order_then_target_order; strict_majority_"
                    "across_seeds_with_even_tie_as_no_call"
                ),
                "selected_deployable_cell_keys": [
                    {"level": level, "target": target, "seed": seed}
                    for level, target, seed in item.selected_deployable_cell_keys
                ],
                "selected_l3_cell_keys": [
                    {"level": level, "target": target, "seed": seed}
                    for level, target, seed in item.selected_l3_cell_keys
                ],
                "selected_deployable_validation_calls": list(
                    item.selected_deployable_validation_calls
                ),
                "selected_l3_validation_calls": list(item.selected_l3_validation_calls),
                "post_action_validation_calls": list(item.post_action_validation_calls),
            },
            "cells": [
                {
                    "level": cell.level,
                    "target": cell.target,
                    "seed": cell.seed,
                    "selected_variant": cell.variant,
                    "threshold": cell.threshold,
                    "validation": cell.validation_metrics,
                }
                for cell in item.cells
            ],
        }
    report = {
        "schema": "predictability_matrix_freeze_report_v2",
        "test_data_present": False,
        "format_version": frozen.format_version,
        "formal_claim_eligible": frozen.formal_claim_eligible,
        "lambda_cost": frozen.lambda_cost,
        "seeds": list(frozen.seeds),
        "predictor_levels": list(frozen.predictor_levels),
        "target_families": list(frozen.target_families),
        "strong_baseline_random_seed": frozen.strong_baseline_random_seed,
        "provenance": frozen.provenance,
        "benchmarks": benchmarks,
    }
    json.dumps(report, allow_nan=False, sort_keys=True)
    return report


def save_frozen_predictability_matrix(
    frozen: FrozenPredictabilityMatrix,
    *,
    model_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Atomically persist the trusted local model bundle and its JSON inventory."""

    destination = Path(model_path).resolve()
    report_destination = Path(report_path).resolve()
    if destination.exists() or report_destination.exists():
        raise FileExistsError("refusing to overwrite frozen matrix artifacts")
    if destination == report_destination:
        raise ValueError("frozen model and report paths must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = destination.with_name(destination.name + ".tmp")
    temporary_report = report_destination.with_name(report_destination.name + ".tmp")
    if temporary_model.exists() or temporary_report.exists():
        raise FileExistsError("frozen matrix staging artifact already exists")
    model_committed = False
    report_committed = False
    try:
        with temporary_model.open("xb") as handle:
            pickle.dump(frozen, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        _commit_file_exclusive(temporary_model, destination)
        model_committed = True
        report = frozen_predictability_matrix_report(frozen)
        report.update(
            {
                "model_path": str(destination),
                "model_sha256": _sha256_file(destination),
                "serialization": "python_pickle_trusted_local_only",
            }
        )
        with temporary_report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _commit_file_exclusive(temporary_report, report_destination)
        report_committed = True
        return report
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
        if model_committed and not report_committed:
            destination.unlink(missing_ok=True)


def load_frozen_predictability_matrix(
    model_path: str | Path, *, expected_sha256: str
) -> FrozenPredictabilityMatrix:
    """Load only a trusted local bundle whose frozen SHA-256 is supplied."""

    source = Path(model_path).resolve()
    if not expected_sha256 or _sha256_file(source) != expected_sha256:
        raise ValueError("frozen matrix model SHA-256 mismatch")
    with source.open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - exact hash + trusted local only
    if not isinstance(value, FrozenPredictabilityMatrix):
        raise ValueError("frozen matrix artifact has the wrong type")
    value.__post_init__()
    return value


def _audit_development_test_disjointness(
    development: DevelopmentIdentityIndex, test: Sequence[AuditExample]
) -> dict[str, Any]:
    test_sources = {item.outcome.source_id for item in test}
    test_images = {item.image_rgb_sha256 for item in test}
    source_overlap = sorted(set(development.sources) & test_sources)
    image_overlap = sorted(set(development.image_rgb_sha256) & test_images)
    if source_overlap or image_overlap:
        raise ValueError(
            "development/test source or decoded-RGB leakage detected: "
            f"source={source_overlap[:3]}, rgb={image_overlap[:3]}"
        )
    return {
        "schema": "predictability_frozen_development_test_audit_v1",
        "passed": True,
        "development_identity_sha256": development.sha256,
        "development_decisions": development.decisions,
        "test_decisions": len(test),
        "source_overlap": 0,
        "image_rgb_sha256_overlap": 0,
    }


def _cell_key_report(key: CellKey) -> dict[str, str | int]:
    return {"level": key[0], "target": key[1], "seed": key[2]}


def _required_metric(metrics: Mapping[str, float | int | None], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, (float, int)):
        raise ValueError(f"required policy metric {name!r} is missing")
    return float(value)


def _zero_if_no_calls(metrics: Mapping[str, float | int | None], name: str) -> float:
    value = metrics.get(name)
    if value is None:
        if metrics.get("calls") != 0:
            raise ValueError(f"{name} is missing for a policy with calls")
        return 0.0
    if not isinstance(value, (float, int)):
        raise ValueError(f"policy metric {name!r} has the wrong type")
    return float(value)


def _operating_point_comparison(
    candidate: Mapping[str, float | int | None],
    baseline: Mapping[str, float | int | None],
) -> dict[str, Any]:
    candidate_accuracy = _required_metric(candidate, "accuracy")
    baseline_accuracy = _required_metric(baseline, "accuracy")
    candidate_cost = _required_metric(candidate, "cost")
    baseline_cost = _required_metric(baseline, "cost")
    candidate_rescue = _zero_if_no_calls(candidate, "rescue_precision")
    baseline_rescue = _zero_if_no_calls(baseline, "rescue_precision")
    candidate_harm = _zero_if_no_calls(candidate, "harm_rate_per_call")
    baseline_harm = _zero_if_no_calls(baseline, "harm_rate_per_call")
    weak_accuracy = candidate_accuracy >= baseline_accuracy
    weak_cost = candidate_cost <= baseline_cost
    pareto = (
        weak_accuracy
        and weak_cost
        and (candidate_accuracy > baseline_accuracy or candidate_cost < baseline_cost)
    )
    return {
        "candidate_accuracy": candidate_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "candidate_cost": candidate_cost,
        "baseline_cost": baseline_cost,
        "accuracy_cost_pareto": pareto,
        "candidate_rescue_precision": candidate_rescue,
        "baseline_rescue_precision": baseline_rescue,
        "rescue_precision_higher": candidate_rescue > baseline_rescue,
        "candidate_harm_rate_per_call": candidate_harm,
        "baseline_harm_rate_per_call": baseline_harm,
        "harm_rate_not_higher": candidate_harm <= baseline_harm,
        "zero_call_rate_convention": (
            "undefined rescue_precision and harm_rate_per_call are compared as 0.0"
        ),
    }


def evaluate_frozen_predictability_matrix(
    frozen: FrozenPredictabilityMatrix,
    datasets: Mapping[str, BenchmarkTestData],
    *,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
    call_rates: Sequence[float],
) -> dict[str, Any]:
    """Evaluate held-out data once without fitting or selecting any choices."""

    if set(datasets) != set(AUDIT_BENCHMARKS):
        raise ValueError("matrix evaluation requires exactly the three benchmarks")
    split_audits: dict[str, Any] = {}
    baseline_test_traces: dict[str, Any] = {}
    baseline_reports: dict[str, Any] = {}
    for benchmark in AUDIT_BENCHMARKS:
        data = datasets[benchmark]
        test_role_audit = data.validate()
        benchmark_frozen = frozen.benchmarks[benchmark]
        disjoint = _audit_development_test_disjointness(
            benchmark_frozen.development_identities, data.test
        )
        disjoint["test_role_validation"] = test_role_audit
        split_audits[benchmark] = disjoint
        traces = apply_strong_baselines(benchmark_frozen.baseline, data.test_siblings)
        baseline_test_traces[benchmark] = traces
        baseline_reports[benchmark] = strong_baseline_report(
            benchmark_frozen.baseline, traces
        )

    headroom_reports: dict[str, Any] = {}
    post_action_reports: dict[str, Any] = {}
    primary_reports: dict[str, Any] = {}
    representation_reports: dict[str, Any] = {}
    cell_reports: list[dict[str, Any]] = []
    completed_cells: list[tuple[str, str, str]] = []
    for benchmark_index, benchmark in enumerate(AUDIT_BENCHMARKS):
        data = datasets[benchmark]
        benchmark_frozen = frozen.benchmarks[benchmark]
        test_outcomes = [item.outcome for item in data.test]
        post_test_outcomes = [item.outcome for item in data.post_action_test]
        strongest_name = benchmark_frozen.baseline.strongest_name
        test_traces = baseline_test_traces[benchmark]
        answer_now = trace_by_name(test_traces, "answer_now")
        strongest = trace_by_name(test_traces, strongest_name)
        validation_answer_now = trace_by_name(
            benchmark_frozen.baseline.validation_traces, "answer_now"
        )
        validation_strongest = trace_by_name(
            benchmark_frozen.baseline.validation_traces, strongest_name
        )
        seed_base = bootstrap_seed + benchmark_index * 10_000

        headroom = fixed_tool_headroom_summary(
            test_outcomes, lambda_cost=frozen.lambda_cost
        )
        oracle_calls = tuple(
            item.incremental_utility(frozen.lambda_cost) > 0.0 for item in test_outcomes
        )
        headroom["always_call"]["paired_vs_answer_now"] = (
            paired_source_bootstrap_policy_difference(
                test_outcomes,
                (True,) * len(test_outcomes),
                answer_now.outcomes,
                answer_now.calls,
                lambda_cost=frozen.lambda_cost,
                resamples=bootstrap_resamples,
                confidence_level=bootstrap_confidence,
                seed=seed_base + 5_000,
            )
        )
        headroom["privileged_binary_oracle"]["paired_vs_answer_now"] = (
            paired_source_bootstrap_policy_difference(
                test_outcomes,
                oracle_calls,
                answer_now.outcomes,
                answer_now.calls,
                lambda_cost=frozen.lambda_cost,
                resamples=bootstrap_resamples,
                confidence_level=bootstrap_confidence,
                seed=seed_base + 5_001,
            )
        )
        headroom_reports[benchmark] = headroom

        post_seed_reports: list[dict[str, Any]] = []
        post_test_call_rows: list[tuple[bool, ...]] = []
        for seed_index, probe in enumerate(benchmark_frozen.post_action_probes):
            predictions, metrics = evaluate_frozen_post_action_probe(
                probe, data.post_action_test, lambda_cost=frozen.lambda_cost
            )
            candidate_calls = tuple(calls_at_threshold(predictions, probe.threshold))
            post_test_call_rows.append(candidate_calls)
            post_seed_reports.append(
                {
                    "seed": probe.seed,
                    "model": "fixed_two_layer_mlp",
                    "target": "direct_gain",
                    "deployable": False,
                    "input_dimension": probe.input_dimension,
                    "validation": probe.validation_metrics,
                    "test_policy": metrics,
                    "test_prediction": prediction_metrics(
                        post_test_outcomes,
                        predictions,
                    ),
                    "test_curve": policy_curve(
                        post_test_outcomes,
                        predictions,
                        lambda_cost=frozen.lambda_cost,
                        call_rates=call_rates,
                    ),
                    "paired_vs_answer_now": paired_source_bootstrap_policy_difference(
                        post_test_outcomes,
                        candidate_calls,
                        answer_now.outcomes,
                        answer_now.calls,
                        lambda_cost=frozen.lambda_cost,
                        resamples=bootstrap_resamples,
                        confidence_level=bootstrap_confidence,
                        seed=seed_base + 4_000 + seed_index,
                    ),
                    "paired_vs_strongest_baseline": paired_source_bootstrap_policy_difference(
                        post_test_outcomes,
                        candidate_calls,
                        strongest.outcomes,
                        strongest.calls,
                        lambda_cost=frozen.lambda_cost,
                        resamples=bootstrap_resamples,
                        confidence_level=bootstrap_confidence,
                        seed=seed_base + 4_010 + seed_index,
                    ),
                }
            )
        post_test_calls = _majority_calls(post_test_call_rows)
        post_validation_metrics = policy_metrics(
            validation_answer_now.outcomes,
            benchmark_frozen.post_action_validation_calls,
            lambda_cost=frozen.lambda_cost,
        )
        post_test_metrics = policy_metrics(
            post_test_outcomes, post_test_calls, lambda_cost=frozen.lambda_cost
        )
        post_action_reports[benchmark] = {
            "schema": "predictability_post_action_probe_report_v2",
            "role": "diagnostic_only_never_deployable",
            "selection_role": "validation_only",
            "seeds": post_seed_reports,
            "mean_test_incremental_utility": mean(
                float(item["test_policy"]["incremental_utility"])
                for item in post_seed_reports
            ),
            "ensemble": {
                "aggregation": "strict_seed_majority_with_even_tie_as_no_call",
                "validation_policy": post_validation_metrics,
                "test_policy": post_test_metrics,
                "paired_vs_answer_now": paired_source_bootstrap_policy_difference(
                    post_test_outcomes,
                    post_test_calls,
                    answer_now.outcomes,
                    answer_now.calls,
                    lambda_cost=frozen.lambda_cost,
                    resamples=bootstrap_resamples,
                    confidence_level=bootstrap_confidence,
                    seed=seed_base + 4_100,
                ),
                "paired_vs_strongest_baseline": paired_source_bootstrap_policy_difference(
                    post_test_outcomes,
                    post_test_calls,
                    strongest.outcomes,
                    strongest.calls,
                    lambda_cost=frozen.lambda_cost,
                    resamples=bootstrap_resamples,
                    confidence_level=bootstrap_confidence,
                    seed=seed_base + 4_101,
                ),
            },
        }

        cell_map = {
            (item.level, item.target, item.seed): item
            for item in benchmark_frozen.cells
        }
        test_calls_by_cell: dict[CellKey, tuple[bool, ...]] = {}
        deployable_lower_bounds: list[float] = []
        for level_index, level in enumerate(frozen.predictor_levels):
            for target_index, target in enumerate(frozen.target_families):
                seed_reports: list[dict[str, Any]] = []
                for seed_index, seed in enumerate(frozen.seeds):
                    cell = cell_map[(level, target, seed)]
                    predictions, metrics = evaluate_frozen_audit_cell(
                        cell, data.test, lambda_cost=frozen.lambda_cost
                    )
                    candidate_calls = tuple(
                        calls_at_threshold(predictions, cell.threshold)
                    )
                    test_calls_by_cell[(level, target, seed)] = candidate_calls
                    paired = paired_source_bootstrap_policy_difference(
                        test_outcomes,
                        candidate_calls,
                        strongest.outcomes,
                        strongest.calls,
                        lambda_cost=frozen.lambda_cost,
                        resamples=bootstrap_resamples,
                        confidence_level=bootstrap_confidence,
                        seed=(
                            seed_base
                            + level_index * 100
                            + target_index * 10
                            + seed_index
                        ),
                    )
                    deployable_lower_bounds.append(float(paired["lower"]))
                    seed_reports.append(
                        {
                            "seed": seed,
                            "selected_variant": cell.variant,
                            "validation": cell.validation_metrics,
                            "test_policy": metrics,
                            "test_prediction": prediction_metrics(
                                test_outcomes,
                                predictions,
                            ),
                            "test_curve": policy_curve(
                                test_outcomes,
                                predictions,
                                lambda_cost=frozen.lambda_cost,
                                call_rates=call_rates,
                            ),
                            "paired_vs_strongest_baseline": paired,
                        }
                    )
                cell_reports.append(
                    {
                        "benchmark": benchmark,
                        "predictor_level": level,
                        "target": target,
                        "strongest_baseline": strongest_name,
                        "seeds": seed_reports,
                        "mean_test_incremental_utility": mean(
                            float(item["test_policy"]["incremental_utility"])
                            for item in seed_reports
                        ),
                    }
                )
                completed_cells.append((benchmark, level, target))

        primary_test_calls = _majority_calls(
            [
                test_calls_by_cell[key]
                for key in benchmark_frozen.selected_deployable_cell_keys
            ]
        )
        primary_validation_metrics = policy_metrics(
            validation_answer_now.outcomes,
            benchmark_frozen.selected_deployable_validation_calls,
            lambda_cost=frozen.lambda_cost,
        )
        primary_test_metrics = policy_metrics(
            test_outcomes, primary_test_calls, lambda_cost=frozen.lambda_cost
        )
        primary_paired = paired_source_bootstrap_policy_difference(
            test_outcomes,
            primary_test_calls,
            strongest.outcomes,
            strongest.calls,
            lambda_cost=frozen.lambda_cost,
            resamples=bootstrap_resamples,
            confidence_level=bootstrap_confidence,
            seed=seed_base + 1_000,
        )
        deployable_lower_bounds.append(float(primary_paired["lower"]))
        primary_reports[benchmark] = {
            "schema": "predictability_primary_deployable_policy_v1",
            "selection_role": "validation_only",
            "selected_cell_keys": [
                _cell_key_report(key)
                for key in benchmark_frozen.selected_deployable_cell_keys
            ],
            "aggregation": "strict_seed_majority_with_even_tie_as_no_call",
            "strongest_baseline": strongest_name,
            "validation_policy": primary_validation_metrics,
            "validation_paired_vs_strongest_baseline": (
                paired_source_bootstrap_policy_difference(
                    validation_answer_now.outcomes,
                    benchmark_frozen.selected_deployable_validation_calls,
                    validation_strongest.outcomes,
                    validation_strongest.calls,
                    lambda_cost=frozen.lambda_cost,
                    resamples=bootstrap_resamples,
                    confidence_level=bootstrap_confidence,
                    seed=seed_base + 3_100,
                )
            ),
            "test_policy": primary_test_metrics,
            "paired_vs_answer_now": paired_source_bootstrap_policy_difference(
                test_outcomes,
                primary_test_calls,
                answer_now.outcomes,
                answer_now.calls,
                lambda_cost=frozen.lambda_cost,
                resamples=bootstrap_resamples,
                confidence_level=bootstrap_confidence,
                seed=seed_base + 1_001,
            ),
            "paired_vs_strongest_baseline": primary_paired,
            "operating_point_vs_strongest_baseline": _operating_point_comparison(
                primary_test_metrics,
                strongest.metrics(lambda_cost=frozen.lambda_cost),
            ),
            "maximum_lower_ci_across_all_deployable_cells_and_primary": max(
                deployable_lower_bounds
            ),
        }

        if benchmark_frozen.selected_l3_cell_keys:
            l3_test_calls = _majority_calls(
                [
                    test_calls_by_cell[key]
                    for key in benchmark_frozen.selected_l3_cell_keys
                ]
            )
            representation_reports[benchmark] = {
                "schema": "predictability_l3_representation_diagnostic_v1",
                "selected_cell_keys": [
                    _cell_key_report(key)
                    for key in benchmark_frozen.selected_l3_cell_keys
                ],
                "selection_role": "validation_only",
                "aggregation": "strict_seed_majority_with_even_tie_as_no_call",
                "validation_policy": policy_metrics(
                    validation_answer_now.outcomes,
                    benchmark_frozen.selected_l3_validation_calls,
                    lambda_cost=frozen.lambda_cost,
                ),
                "validation_paired_vs_strongest_baseline": (
                    paired_source_bootstrap_policy_difference(
                        validation_answer_now.outcomes,
                        benchmark_frozen.selected_l3_validation_calls,
                        validation_strongest.outcomes,
                        validation_strongest.calls,
                        lambda_cost=frozen.lambda_cost,
                        resamples=bootstrap_resamples,
                        confidence_level=bootstrap_confidence,
                        seed=seed_base + 3_000,
                    )
                ),
                "test_policy": policy_metrics(
                    test_outcomes, l3_test_calls, lambda_cost=frozen.lambda_cost
                ),
                "test_paired_vs_strongest_baseline": (
                    paired_source_bootstrap_policy_difference(
                        test_outcomes,
                        l3_test_calls,
                        strongest.outcomes,
                        strongest.calls,
                        lambda_cost=frozen.lambda_cost,
                        resamples=bootstrap_resamples,
                        confidence_level=bootstrap_confidence,
                        seed=seed_base + 2_000,
                    )
                ),
            }
        else:
            representation_reports[benchmark] = {
                "schema": "predictability_l3_representation_diagnostic_v1",
                "available": False,
                "reason": "L3 was not included in this non-formal partial smoke",
            }
    return {
        "schema": "predictability_matrix_report_v3",
        "formal_claim_eligible": frozen.formal_claim_eligible,
        "frozen_before_test": True,
        "freeze_inventory_sha256": _canonical_sha256(
            frozen_predictability_matrix_report(frozen)
        ),
        "lambda_cost": frozen.lambda_cost,
        "seeds": list(frozen.seeds),
        "split_audits": split_audits,
        "strong_baselines": baseline_reports,
        "oracle_headroom": headroom_reports,
        "primary_deployable": primary_reports,
        "post_action_probe": post_action_reports,
        "representation_diagnostic": representation_reports,
        "matrix": matrix_completion_report(completed_cells),
        "cells": cell_reports,
        "bootstrap_seed_schedule": {
            "base": bootstrap_seed,
            "benchmark_stride": 10_000,
            "individual_cells": "level_index*100+target_index*10+seed_index",
            "primary_test_vs_strongest": 1_000,
            "primary_test_vs_answer_now": 1_001,
            "l3_test_vs_strongest": 2_000,
            "l3_validation_vs_strongest": 3_000,
            "primary_validation_vs_strongest": 3_100,
            "post_action_seed_vs_answer_now": "4000+seed_index",
            "post_action_seed_vs_strongest": "4010+seed_index",
            "post_action_ensemble_vs_answer_now": 4_100,
            "post_action_ensemble_vs_strongest": 4_101,
            "always_call_vs_answer_now": 5_000,
            "privileged_oracle_vs_answer_now": 5_001,
        },
    }


def run_predictability_matrix(
    datasets: Mapping[str, BenchmarkAuditData],
    *,
    lambda_cost: float,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
    call_rates: Sequence[float],
    seeds: Sequence[int] = AUDIT_SEEDS,
    predictor_levels: Sequence[str] = PREDICTOR_LEVELS,
    target_families: Sequence[str] = TARGET_FAMILIES,
    strong_baseline_random_seed: int = STRONG_BASELINE_RANDOM_SEED,
    formal_claim_eligible: bool = False,
) -> dict[str, Any]:
    """Convenience smoke wrapper; formal work must persist the freeze first."""

    if set(datasets) != set(AUDIT_BENCHMARKS):
        raise ValueError("matrix runner requires exactly ChartQA, DocVQA, and HRBench")
    _validate_matrix_configuration(
        seeds=seeds,
        predictor_levels=predictor_levels,
        target_families=target_families,
        strong_baseline_random_seed=strong_baseline_random_seed,
        formal_claim_eligible=formal_claim_eligible,
    )
    if formal_claim_eligible:
        raise ValueError(
            "formal one-shot evaluation is forbidden; persist the complete frozen "
            "matrix before loading test data"
        )
    for data in datasets.values():
        data.validate()
    frozen = fit_predictability_matrix(
        {name: data.development() for name, data in datasets.items()},
        lambda_cost=lambda_cost,
        seeds=seeds,
        predictor_levels=predictor_levels,
        target_families=target_families,
        strong_baseline_random_seed=strong_baseline_random_seed,
        formal_claim_eligible=False,
        provenance={"purpose": "non_scientific_one_shot_smoke"},
    )
    return evaluate_frozen_predictability_matrix(
        frozen,
        {name: data.held_out_test() for name, data in datasets.items()},
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
        call_rates=call_rates,
    )
