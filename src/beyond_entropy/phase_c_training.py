"""Freeze matched three-domain selector-training configurations for Phase C."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .predictability_matrix_artifacts import atomic_json_write_exclusive, sha256_file


METHODS = (
    "outcome_only",
    "counterfactual_utility",
    "factorized_potential_outcomes",
)
SEEDS = (17, 29, 47)
BENCHMARKS = ("chartqa", "docvqa", "hrbench")


def _resolve_repository_file(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Phase-C training file must stay inside the repository") from exc
    if "heldout" in path.parts:
        raise ValueError("Phase-C selector training must not reference held-out data")
    return path


def _count_jsonl(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{number}")
            count += 1
    if not count:
        raise ValueError(f"empty JSONL: {path}")
    return count


def validate_phase_c_training_matrix(
    matrix: dict[str, Any], repository_root: str | Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Validate the frozen matrix without reading any held-out identity or outcome."""

    root = Path(repository_root).resolve()
    if (
        matrix.get("schema") != "factorized_phase_c_training_matrix_v1"
        or matrix.get("stage") != "phase_c_training"
        or tuple(matrix.get("methods", ())) != METHODS
        or tuple(matrix.get("seeds", ())) != SEEDS
        or matrix.get("test_authorized") is not False
        or matrix.get("formal_heldout_referenced") is not False
        or matrix.get("development_monitor_only") is not True
        or matrix.get("steps") != 3072
        or matrix.get("schedule") != "domain_balanced_1024_draws_per_domain"
    ):
        raise ValueError("invalid frozen Phase-C training matrix")
    if tuple(sorted(matrix.get("datasets", {}))) != BENCHMARKS:
        raise ValueError("Phase-C matrix requires the three frozen domains")

    resolved: dict[str, dict[str, dict[str, Any]]] = {}
    for benchmark in BENCHMARKS:
        domain = matrix["datasets"][benchmark]
        if set(domain) != {"train", "validation"}:
            raise ValueError("Phase-C matrix requires train and development monitor roles")
        resolved[benchmark] = {}
        for role in ("train", "validation"):
            spec = domain[role]
            if set(spec) != {"states", "manifest", "rollouts"}:
                raise ValueError("invalid Phase-C dataset role specification")
            materialized: dict[str, Any] = {"states": int(spec["states"])}
            for field in ("manifest", "rollouts"):
                file_spec = spec[field]
                if set(file_spec) != {"path", "sha256"}:
                    raise ValueError("invalid Phase-C file specification")
                path = _resolve_repository_file(root, str(file_spec["path"]))
                if sha256_file(path) != file_spec["sha256"]:
                    raise ValueError(f"{benchmark}.{role}.{field} hash mismatch")
                materialized[field] = {
                    "path": str(path), "sha256": str(file_spec["sha256"]),
                }
            manifest_states = _count_jsonl(Path(materialized["manifest"]["path"]))
            if (
                (role == "train" and manifest_states != materialized["states"])
                or (role == "validation" and manifest_states < materialized["states"])
            ):
                raise ValueError(f"{benchmark}.{role} manifest state count mismatch")
            if _count_jsonl(Path(materialized["rollouts"]["path"])) != materialized["states"]:
                raise ValueError(f"{benchmark}.{role} rollout state count mismatch")
            resolved[benchmark][role] = materialized
    return resolved


def materialize_phase_c_seed_configs(
    *, matrix_path: str | Path, repository_root: str | Path,
    seed: int, output_dir: str | Path,
) -> dict[str, Any]:
    """Write matched arm/evaluation configs for one pre-registered seed."""

    root = Path(repository_root).resolve()
    source = Path(matrix_path).resolve()
    matrix = json.loads(source.read_text())
    datasets = validate_phase_c_training_matrix(matrix, root)
    if seed not in SEEDS:
        raise ValueError("seed is not in the frozen Phase-C matrix")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    matrix_hash = sha256_file(source)
    omitted = {
        "schema", "methods", "seeds", "schedule", "formal_heldout_referenced",
        "development_monitor_only", "datasets", "evaluation",
    }
    common = {key: value for key, value in matrix.items() if key not in omitted}
    common.update({
        "schema": "cv_method_post_training_config_v1",
        "seed": seed,
        "datasets": datasets,
        "formal_claim_eligible": False,
        "matrix_config_path": str(source),
        "matrix_config_sha256": matrix_hash,
        "validation_role": "previously_seen_development_monitor_only",
    })
    config_paths = {}
    for method in METHODS:
        path = destination / f"{method}.json"
        atomic_json_write_exclusive(path, {**common, "method": method})
        config_paths[method] = str(path)
    evaluation = matrix["evaluation"]
    evaluation_path = destination / "evaluation.json"
    atomic_json_write_exclusive(evaluation_path, {
        "schema": "cv_method_evaluation_config_v1",
        "stage": "phase_c_training",
        "test_authorized": False,
        "formal_claim_eligible": False,
        "validation_role": "previously_seen_development_monitor_only",
        "matrix_config_path": str(source),
        "matrix_config_sha256": matrix_hash,
        "seed": seed,
        "bootstrap_samples": evaluation["bootstrap_samples"],
        "bootstrap_seed": evaluation["bootstrap_seed"],
        "rates": evaluation["rates"],
        "lambdas": evaluation["lambdas"],
        "validation_rollouts": {
            benchmark: datasets[benchmark]["validation"]["rollouts"]
            for benchmark in BENCHMARKS
        },
    })
    report = {
        "schema": "factorized_phase_c_seed_configs_v1",
        "seed": seed,
        "matrix_config_sha256": matrix_hash,
        "formal_heldout_referenced": False,
        "configs": config_paths,
        "evaluation": str(evaluation_path),
    }
    report_path = destination / "materialization.json"
    atomic_json_write_exclusive(report_path, report)
    return {**report, "materialization": str(report_path)}
