from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .benchmarks import load_manifest
from .dataset import read_jsonl
from .predictability_audit import (
    AUDIT_BENCHMARKS,
    collapse_fixed_entropy_tool,
)
from .predictability_baselines import validate_fixed_tool_outcomes
from .predictability_features import (
    load_predictability_feature_dataset,
    post_action_probe_examples_from_feature_dataset,
)
from .predictability_matrix import (
    BenchmarkDevelopmentData,
    BenchmarkTestData,
)
from .predictability_modeling import AuditExample
from .predictability_post_action import PostActionProbeExample
from .schema import ActionRecord


DEVELOPMENT_INPUT_SCHEMA = "predictability_matrix_development_inputs_v1"
TEST_INPUT_SCHEMA = "predictability_matrix_test_inputs_v1"
TEST_TRANSACTION_PLAN_SCHEMA = "predictability_matrix_test_transaction_plan_v1"
TEST_ACCESS_SCHEMA = "predictability_matrix_test_access_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_write_exclusive(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"artifact staging path exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link commit is atomic and refuses an existing destination. Using
        # os.replace here would permit a check-then-rename race to overwrite an
        # already-created one-shot access ledger.
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} keys differ; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_hashed_json(
    path: str | Path, *, expected_sha256: str, schema: str
) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"{schema} SHA-256 mismatch")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{schema} is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"unexpected {schema} document")
    return source, value


def current_clean_revision(repo_root: str | Path) -> str:
    root = Path(repo_root).resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("tracked worktree must be clean for a formal matrix phase")
    if len(revision) != 40:
        raise ValueError("could not resolve a full Git revision")
    return revision


@dataclass(frozen=True)
class LoadedRole:
    examples: tuple[AuditExample, ...]
    post_action_examples: tuple[PostActionProbeExample, ...]
    siblings: tuple[ActionRecord, ...]
    hashes: Mapping[str, str]


def _hashed_artifact(value: Any, *, name: str) -> tuple[Path, str]:
    spec = _mapping(value, name=name)
    _exact_keys(spec, {"path", "sha256"}, name=name)
    path = Path(str(spec["path"])).resolve()
    expected = str(spec["sha256"])
    if len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    return path, expected


def load_role_artifacts(
    value: Any,
    *,
    benchmark: str,
    role: str,
    code_revision: str,
    protocol: Mapping[str, Any] | None = None,
) -> LoadedRole:
    spec = _mapping(value, name=f"{benchmark}.{role}")
    _exact_keys(
        spec,
        {"manifest", "rollouts", "rollout_provenance", "features"},
        name=f"{benchmark}.{role}",
    )
    manifest_path, manifest_sha256 = _hashed_artifact(
        spec["manifest"], name=f"{benchmark}.{role}.manifest"
    )
    rollout_path, rollout_sha256 = _hashed_artifact(
        spec["rollouts"], name=f"{benchmark}.{role}.rollouts"
    )
    feature_path, feature_sha256 = _hashed_artifact(
        spec["features"], name=f"{benchmark}.{role}.features"
    )
    provenance_path, provenance_sha256 = _hashed_artifact(
        spec["rollout_provenance"],
        name=f"{benchmark}.{role}.rollout_provenance",
    )
    try:
        rollout_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("rollout provenance is not valid JSON") from exc
    if not isinstance(rollout_provenance, Mapping):
        raise ValueError("rollout provenance must be a mapping")
    payload, examples = load_predictability_feature_dataset(feature_path)
    metadata = _mapping(payload.get("metadata"), name="feature metadata")
    required_metadata = {
        "dataset_role": role,
        "manifest_sha256": manifest_sha256,
        "rollouts_sha256": rollout_sha256,
        "code_revision": code_revision,
    }
    if protocol is not None:
        feature_contract = _mapping(
            protocol.get("feature_extraction"), name="feature extraction contract"
        )
        required_metadata.update(
            {
                "model": feature_contract["model"],
                "model_revision": feature_contract["model_revision"],
                "dtype": feature_contract["dtype"],
                "attention_implementation": feature_contract[
                    "attention_implementation"
                ],
                "min_pixels": feature_contract["min_pixels"],
                "max_pixels": feature_contract["max_pixels"],
                "local_files_only": feature_contract["local_files_only"],
                "require_prompt_hash": feature_contract["require_prompt_hash"],
            }
        )
        expected_rollout = {
            "manifest_sha256": manifest_sha256,
            "output_sha256": rollout_sha256,
            "code_revision": code_revision,
            "model": feature_contract["model"],
            "model_revision": feature_contract["model_revision"],
            "generation_seeds": feature_contract["generation_seeds"],
            "max_new_tokens": feature_contract["max_new_tokens"],
            "proposer": feature_contract["proposer"],
            "visual_crop_ratio": feature_contract["visual_crop_ratio"],
            "visual_cost": feature_contract["visual_cost_per_crop"],
            "dtype": feature_contract["dtype"],
            "attention_implementation": feature_contract["attention_implementation"],
            "min_pixels": feature_contract["min_pixels"],
            "max_pixels": feature_contract["max_pixels"],
            "local_files_only": feature_contract["local_files_only"],
        }
        for field, expected in expected_rollout.items():
            if rollout_provenance.get(field) != expected:
                raise ValueError(
                    f"{benchmark}.{role} rollout provenance mismatch for {field}"
                )
    for field, expected in required_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"{benchmark}.{role} feature metadata mismatch for {field}"
            )
    siblings = tuple(read_jsonl(rollout_path))
    manifest = load_manifest(manifest_path)
    for field, expected in {
        "examples": len(manifest),
        "completed_examples": len(manifest),
        "scorer": benchmark,
    }.items():
        if rollout_provenance.get(field) != expected:
            raise ValueError(
                f"{benchmark}.{role} rollout provenance mismatch for {field}"
            )
    examples_by_state = {item.outcome.state_id: item for item in examples}
    manifest_by_state = {item.state.state_id: item for item in manifest}
    if (
        len(examples_by_state) != len(examples)
        or len(manifest_by_state) != len(manifest)
        or set(examples_by_state) != set(manifest_by_state)
    ):
        raise ValueError(f"{benchmark}.{role} manifest/feature state coverage differs")
    for state_id, example in examples_by_state.items():
        state = manifest_by_state[state_id].state
        if (
            example.outcome.image_id != state.image_id
            or example.outcome.source_id != state.source_id
        ):
            raise ValueError(f"{benchmark}.{role} manifest/feature identities differ")
    validate_fixed_tool_outcomes(
        [item.outcome for item in examples],
        collapse_fixed_entropy_tool(siblings),
    )
    post_action_examples = tuple(
        post_action_probe_examples_from_feature_dataset(payload)
    )
    if protocol is not None:
        feature_contract = _mapping(
            protocol.get("feature_extraction"), name="feature extraction contract"
        )
        expected_dimensions = dict(feature_contract["pre_action_dimensions"])
        actual_dimensions = {
            level: sorted({len(item.inputs.feature_vector(level)) for item in examples})
            for level in expected_dimensions
        }
        if any(
            values != [int(expected_dimensions[level])]
            for level, values in actual_dimensions.items()
        ):
            raise ValueError(f"{benchmark}.{role} pre-action feature dimensions differ")
        post_dimensions = {
            len(item.inputs.feature_vector()) for item in post_action_examples
        }
        if post_dimensions != {int(feature_contract["post_action_probe_dimension"])}:
            raise ValueError(
                f"{benchmark}.{role} post-action feature dimension differs"
            )
    return LoadedRole(
        examples=tuple(examples),
        post_action_examples=post_action_examples,
        siblings=siblings,
        hashes={
            "manifest": manifest_sha256,
            "rollouts": rollout_sha256,
            "features": feature_sha256,
            "rollout_provenance": provenance_sha256,
            "manifest_path": str(manifest_path),
            "rollouts_path": str(rollout_path),
            "features_path": str(feature_path),
            "rollout_provenance_path": str(provenance_path),
        },
    )


def validate_protocol_artifact(value: Any) -> tuple[Path, str, dict[str, Any]]:
    path, expected = _hashed_artifact(value, name="protocol")
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("protocol is not valid JSON") from exc
    if (
        not isinstance(protocol, dict)
        or protocol.get("schema") != "predictability_audit_protocol_v1"
    ):
        raise ValueError("unexpected predictability protocol")
    return path, expected, protocol


def load_development_input_spec(
    path: str | Path,
    *,
    expected_sha256: str,
    repo_root: str | Path,
) -> tuple[
    dict[str, BenchmarkDevelopmentData],
    dict[str, Any],
    dict[str, Any],
]:
    source, spec = load_hashed_json(
        path, expected_sha256=expected_sha256, schema=DEVELOPMENT_INPUT_SCHEMA
    )
    _exact_keys(
        spec,
        {"schema", "code_revision", "protocol", "benchmarks"},
        name="development input spec",
    )
    revision = current_clean_revision(repo_root)
    if spec.get("code_revision") != revision:
        raise ValueError("development input code revision differs from clean HEAD")
    protocol_path, protocol_sha256, protocol = validate_protocol_artifact(
        spec["protocol"]
    )
    raw_benchmarks = _mapping(spec["benchmarks"], name="benchmarks")
    if set(raw_benchmarks) != set(AUDIT_BENCHMARKS):
        raise ValueError("development input spec requires exactly three benchmarks")
    datasets: dict[str, BenchmarkDevelopmentData] = {}
    artifact_hashes: dict[str, Any] = {}
    for benchmark in AUDIT_BENCHMARKS:
        benchmark_spec = _mapping(raw_benchmarks[benchmark], name=benchmark)
        _exact_keys(benchmark_spec, {"train", "validation"}, name=benchmark)
        train = load_role_artifacts(
            benchmark_spec["train"],
            benchmark=benchmark,
            role="train",
            code_revision=revision,
            protocol=protocol,
        )
        validation = load_role_artifacts(
            benchmark_spec["validation"],
            benchmark=benchmark,
            role="validation",
            code_revision=revision,
            protocol=protocol,
        )
        datasets[benchmark] = BenchmarkDevelopmentData(
            train=train.examples,
            validation=validation.examples,
            post_action_train=train.post_action_examples,
            post_action_validation=validation.post_action_examples,
            validation_siblings=validation.siblings,
        )
        artifact_hashes[benchmark] = {
            "train": train.hashes,
            "validation": validation.hashes,
        }
    provenance = {
        "development_input_spec": str(source),
        "development_input_spec_sha256": expected_sha256,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "code_revision": revision,
        "development_artifacts": artifact_hashes,
    }
    return datasets, protocol, provenance


def load_test_input_spec_header(
    path: str | Path,
    *,
    expected_sha256: str,
    repo_root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    source, spec = load_hashed_json(
        path, expected_sha256=expected_sha256, schema=TEST_INPUT_SCHEMA
    )
    _exact_keys(
        spec,
        {
            "schema",
            "code_revision",
            "protocol",
            "allocation_report",
            "frozen",
            "benchmarks",
            "access_ledger",
            "test_transaction_plan_sha256",
            "output",
        },
        name="test input spec",
    )
    plan_sha256 = spec["test_transaction_plan_sha256"]
    if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
        raise ValueError("test transaction plan SHA-256 is invalid")
    if not isinstance(spec["output"], str) or not spec["output"].strip():
        raise ValueError("test input spec output must be a non-empty path")
    ledger_spec = _mapping(spec["access_ledger"], name="access_ledger")
    _exact_keys(ledger_spec, {"path", "sha256"}, name="access_ledger")
    if Path(str(ledger_spec["path"])).resolve() == Path(spec["output"]).resolve():
        raise ValueError("test access ledger and output paths must differ")
    allocation_spec = _mapping(spec["allocation_report"], name="allocation_report")
    _exact_keys(allocation_spec, {"path", "sha256"}, name="allocation_report")
    revision = current_clean_revision(repo_root)
    if spec.get("code_revision") != revision:
        raise ValueError("test input code revision differs from clean HEAD")
    _, _, protocol = validate_protocol_artifact(spec["protocol"])
    raw_frozen = _mapping(spec["frozen"], name="frozen")
    _exact_keys(raw_frozen, {"model", "report"}, name="frozen")
    raw_benchmarks = _mapping(spec["benchmarks"], name="benchmarks")
    if set(raw_benchmarks) != set(AUDIT_BENCHMARKS):
        raise ValueError("test input spec requires exactly three benchmarks")
    for benchmark in AUDIT_BENCHMARKS:
        benchmark_spec = _mapping(raw_benchmarks[benchmark], name=benchmark)
        _exact_keys(benchmark_spec, {"test"}, name=benchmark)
        test_spec = _mapping(benchmark_spec["test"], name=f"{benchmark}.test")
        _exact_keys(
            test_spec,
            {"manifest", "rollouts", "rollout_provenance", "features"},
            name=f"{benchmark}.test",
        )
        # Deliberately do not touch test artifact paths before the access ledger.
        for artifact in (
            "manifest",
            "rollouts",
            "rollout_provenance",
            "features",
        ):
            artifact_spec = _mapping(
                test_spec[artifact], name=f"{benchmark}.test.{artifact}"
            )
            _exact_keys(
                artifact_spec,
                {"path", "sha256"},
                name=f"{benchmark}.test.{artifact}",
            )
    return source, spec, protocol, revision


def load_test_datasets_after_access_ledger(
    spec: Mapping[str, Any], *, code_revision: str, protocol: Mapping[str, Any]
) -> tuple[dict[str, BenchmarkTestData], dict[str, Any]]:
    raw_benchmarks = _mapping(spec["benchmarks"], name="benchmarks")
    datasets: dict[str, BenchmarkTestData] = {}
    artifact_hashes: dict[str, Any] = {}
    for benchmark in AUDIT_BENCHMARKS:
        benchmark_spec = _mapping(raw_benchmarks[benchmark], name=benchmark)
        loaded = load_role_artifacts(
            _mapping(benchmark_spec["test"], name=f"{benchmark}.test"),
            benchmark=benchmark,
            role="test",
            code_revision=code_revision,
            protocol=protocol,
        )
        datasets[benchmark] = BenchmarkTestData(
            test=loaded.examples,
            post_action_test=loaded.post_action_examples,
            test_siblings=loaded.siblings,
        )
        artifact_hashes[benchmark] = {"test": loaded.hashes}
    return datasets, artifact_hashes
