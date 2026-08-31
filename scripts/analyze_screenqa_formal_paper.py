#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from beyond_entropy.action_value import predict_frozen_factorized_action_values
from beyond_entropy.dataset import group_by_decision, read_jsonl
from beyond_entropy.rescue_gate import DecisionKey


SCHEMA = "screenqa_formal_paper_analysis_v1"
ANALYSIS_STATUS = (
    "secondary paired analysis implementation locked before ScreenQA risk-"
    "calibration or formal outcomes were opened"
)
PRIMARY = "risk_calibrated_counterfactual_value"
BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_CONFIDENCE = 0.975
BOOTSTRAP_SEED = 20260831
EXPECTED_SOURCES = 1471
EXPECTED_DECISIONS = 14672


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_revision() -> str:
    repo_dir = Path(__file__).resolve().parents[1]
    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), "tracked worktree must be clean")
    return subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_formal_evaluation(
    evaluation_dir: Path,
) -> tuple[dict[str, Any], Path, Path]:
    sums = evaluation_dir / "SHA256SUMS"
    if not sums.is_file():
        raise FileNotFoundError("ScreenQA formal evaluation SHA256SUMS is missing")
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = evaluation_dir / relative.strip()
        _require(path.is_file(), f"formal evaluation file is missing: {path}")
        _require(sha256_file(path) == expected, f"formal evaluation hash mismatch: {path}")
    report_path = evaluation_dir / "report.json"
    completion_path = evaluation_dir / "formal-result.complete.json"
    report = _load_object(report_path, "formal report")
    completion = _load_object(completion_path, "formal completion")
    _require(
        completion.get("one_shot_formal_evaluation_complete") is True,
        "formal evaluation is not complete",
    )
    _require(
        completion.get("formal_outcomes_used_for_tuning") is False,
        "formal completion permits outcome tuning",
    )
    _require(
        completion.get("report_sha256") == sha256_file(report_path),
        "formal completion report binding mismatch",
    )
    run = report.get("run")
    _require(isinstance(run, Mapping), "formal report lacks run provenance")
    assert isinstance(run, Mapping)
    expected_run: dict[str, object] = {
        "formal_outcomes_used": True,
        "no_target_derived_tuning": True,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    for key, expected_run_value in expected_run.items():
        _require(
            run.get(key) == expected_run_value,
            f"formal report {key} mismatch",
        )
    revision = _git_revision()
    _require(run.get("code_revision") == revision, "formal report code revision mismatch")
    _require(completion.get("code_revision") == revision, "formal completion revision mismatch")
    _require(report.get("n_sources") == EXPECTED_SOURCES, "formal source count mismatch")
    _require(
        report.get("n_decisions") == EXPECTED_DECISIONS,
        "formal decision count mismatch",
    )
    model_path = Path(str(run.get("calibrated_model", ""))).resolve()
    rollouts_path = Path(str(run.get("rollouts", ""))).resolve()
    for path, hash_key in (
        (model_path, "calibrated_model_sha256"),
        (rollouts_path, "rollouts_sha256"),
    ):
        _require(path.is_file(), f"formal input does not exist: {path}")
        expected_hash = run.get(hash_key)
        _require(
            isinstance(expected_hash, str) and sha256_file(path) == expected_hash,
            f"formal input hash mismatch: {path}",
        )
    return report, model_path, rollouts_path


def _source_means(
    decision_values: Mapping[DecisionKey, float],
    source_by_key: Mapping[DecisionKey, str],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for key, value in decision_values.items():
        grouped.setdefault(source_by_key[key], []).append(float(value))
    return {source: mean(values) for source, values in grouped.items()}


def holm_adjust(raw_p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw_p_values, key=lambda name: (raw_p_values[name], name))
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(ordered):
        candidate = min(1.0, (total - index) * float(raw_p_values[name]))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def bootstrap_policy_table(
    policy_source_values: Mapping[str, Mapping[str, float]],
    *,
    primary: str = PRIMARY,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence_level: float = BOOTSTRAP_CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
    batch_size: int = 128,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    import numpy as np  # type: ignore[import-not-found]

    _require(primary in policy_source_values, "primary policy is absent")
    _require(n_resamples > 0 and batch_size > 0, "invalid bootstrap size")
    _require(0.0 < confidence_level < 1.0, "invalid confidence level")
    policies = list(policy_source_values)
    sources = sorted(policy_source_values[primary])
    _require(len(sources) >= 2, "source bootstrap needs at least two sources")
    for policy in policies:
        _require(
            sorted(policy_source_values[policy]) == sources,
            f"source population mismatch for {policy}",
        )
    arrays = {
        policy: np.asarray(
            [float(policy_source_values[policy][source]) for source in sources],
            dtype=np.float64,
        )
        for policy in policies
    }
    _require(
        all(bool(np.isfinite(values).all()) for values in arrays.values()),
        "non-finite policy utility",
    )
    draws = {policy: np.empty(n_resamples, dtype=np.float64) for policy in policies}
    rng = np.random.default_rng(seed)
    completed = 0
    while completed < n_resamples:
        current = min(batch_size, n_resamples - completed)
        indices = rng.integers(0, len(sources), size=(current, len(sources)))
        for policy in policies:
            draws[policy][completed : completed + current] = arrays[policy][
                indices
            ].mean(axis=1)
        completed += current
    alpha = 1.0 - confidence_level
    policy_intervals = {
        policy: {
            "source_utility": float(arrays[policy].mean()),
            "ci_low": float(np.quantile(draws[policy], alpha / 2.0)),
            "ci_high": float(np.quantile(draws[policy], 1.0 - alpha / 2.0)),
        }
        for policy in policies
    }
    raw_p_values: dict[str, float] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    for policy in policies:
        if policy == primary:
            continue
        difference_draws = draws[primary] - draws[policy]
        name = f"{primary}_minus_{policy}"
        p_value = float(
            (1 + int((difference_draws <= 0.0).sum())) / (n_resamples + 1)
        )
        raw_p_values[name] = p_value
        comparisons[name] = {
            "primary": primary,
            "comparator": policy,
            "source_utility_difference": (
                policy_intervals[primary]["source_utility"]
                - policy_intervals[policy]["source_utility"]
            ),
            "paired_ci_low": float(np.quantile(difference_draws, alpha / 2.0)),
            "paired_ci_high": float(
                np.quantile(difference_draws, 1.0 - alpha / 2.0)
            ),
            "bootstrap_p_one_sided": p_value,
        }
    adjusted = holm_adjust(raw_p_values)
    for name, comparison in comparisons.items():
        comparison.update(
            {
                "holm_adjusted_p_one_sided": adjusted[name],
                "paired_interval_strictly_above_zero": (
                    float(comparison["paired_ci_low"]) > 0.0
                ),
                "holm_reject_at_0_025": adjusted[name] < 0.025,
                "status": "secondary_analysis_locked_before_formal_outcomes",
            }
        )
    return policy_intervals, comparisons


def policy_decision_values(
    model: Mapping[str, Any],
    records: Sequence[Any],
) -> tuple[
    dict[str, dict[DecisionKey, float]],
    dict[DecisionKey, str],
    dict[str, float],
    dict[str, Any],
]:
    threshold = float(model["threshold"])
    calibration = model.get("risk_calibration")
    _require(isinstance(calibration, Mapping), "model lacks risk calibration")
    assert isinstance(calibration, Mapping)
    _require(
        calibration.get("selection_status")
        == "selected_non_degenerate_safe_threshold",
        "model is not safely calibrated",
    )
    _require(
        float(calibration.get("selected_threshold", math.nan)) == threshold,
        "model threshold differs from calibration choice",
    )
    lambda_cost = float(model["lambda_cost"])
    _require(lambda_cost == 0.05, "unexpected formal utility cost")
    actions, scores = predict_frozen_factorized_action_values(model, records)
    grouped = group_by_decision(records)
    _require(set(actions) == set(scores) == set(grouped), "prediction coverage mismatch")
    policy_names = (
        PRIMARY,
        "no_call",
        "learned_crop_always_call",
        "random_crop_always_call_expected",
        "post_action_entropy_always_call_idealized",
        "ug_exhaustive_entropy_four_calls",
        "random_crop_same_learned_gate_expected",
        "post_action_entropy_same_learned_gate_idealized",
        "ug_exhaustive_entropy_same_learned_gate",
        "matched_budget_entropy_gate_learned_crop",
        "matched_budget_entropy_gate_random_crop",
        "matched_budget_random_gate_random_crop_expected",
    )
    values: dict[str, dict[DecisionKey, float]] = {
        name: {} for name in policy_names
    }
    source_by_key: dict[DecisionKey, str] = {}
    diagnostics: dict[DecisionKey, dict[str, Any]] = {}
    calls = 0
    positive_net_available = 0
    calls_with_positive_net_available = 0
    selected_positive_net_calls = 0
    rescue_calls = 0
    harm_calls = 0
    unchanged_calls = 0
    action_ids: set[str] = set()
    for key, siblings in grouped.items():
        answers = [record for record in siblings if record.action_type == "ANSWER"]
        zooms = sorted(
            (record for record in siblings if record.action_type == "ZOOM"),
            key=lambda record: record.action_id,
        )
        _require(len(answers) == 1 and len(zooms) == 4, f"invalid decision {key!r}")
        baseline = answers[0]
        matches = [zoom for zoom in zooms if zoom.action_id == actions[key]]
        _require(len(matches) == 1, f"selected crop absent for {key!r}")
        selected = matches[0]
        called = float(scores[key]) >= threshold
        selected_utility = float(selected.voi(lambda_cost))
        zoom_utilities = [float(zoom.voi(lambda_cost)) for zoom in zooms]
        random_utility = mean(zoom_utilities)
        entropy_action = min(
            zooms, key=lambda zoom: (zoom.entropy_after, zoom.action_id)
        )
        entropy_utility = float(entropy_action.voi(lambda_cost))
        exhaustive_utility = float(entropy_action.delta_success) - lambda_cost * sum(
            float(zoom.tool_cost) for zoom in zooms
        )
        values[PRIMARY][key] = selected_utility if called else 0.0
        values["no_call"][key] = 0.0
        values["learned_crop_always_call"][key] = selected_utility
        values["random_crop_always_call_expected"][key] = random_utility
        values["post_action_entropy_always_call_idealized"][key] = entropy_utility
        values["ug_exhaustive_entropy_four_calls"][key] = exhaustive_utility
        values["random_crop_same_learned_gate_expected"][key] = (
            random_utility if called else 0.0
        )
        values["post_action_entropy_same_learned_gate_idealized"][key] = (
            entropy_utility if called else 0.0
        )
        values["ug_exhaustive_entropy_same_learned_gate"][key] = (
            exhaustive_utility if called else 0.0
        )
        source_by_key[key] = baseline.source_id
        calls += int(called)
        has_positive = max(zoom_utilities) > 0.0
        positive_net_available += int(has_positive)
        calls_with_positive_net_available += int(called and has_positive)
        selected_positive_net_calls += int(called and selected_utility > 0.0)
        task_effect = float(selected.delta_success) if called else 0.0
        rescue_calls += int(called and task_effect > 0.0)
        harm_calls += int(called and task_effect < 0.0)
        unchanged_calls += int(called and task_effect == 0.0)
        fixed = {zoom.action_id: float(zoom.voi(lambda_cost)) for zoom in zooms}
        action_ids.update(fixed)
        diagnostics[key] = {
            "entropy_before": float(baseline.entropy_before),
            "selected_utility": selected_utility,
            "random_utility": random_utility,
            "fixed": fixed,
        }
    entropy_order = sorted(
        diagnostics,
        key=lambda key: (float(diagnostics[key]["entropy_before"]), key),
        reverse=True,
    )
    entropy_gate = set(entropy_order[:calls])
    random_gate_probability = calls / len(grouped) if grouped else 0.0
    for action_id in sorted(action_ids):
        values[f"fixed_crop_{action_id}_always_call"] = {}
        values[f"fixed_crop_{action_id}_matched_entropy_gate"] = {}
    for key, diagnostic in diagnostics.items():
        entropy_called = key in entropy_gate
        selected_utility = float(diagnostic["selected_utility"])
        random_utility = float(diagnostic["random_utility"])
        values["matched_budget_entropy_gate_learned_crop"][key] = (
            selected_utility if entropy_called else 0.0
        )
        values["matched_budget_entropy_gate_random_crop"][key] = (
            random_utility if entropy_called else 0.0
        )
        values["matched_budget_random_gate_random_crop_expected"][key] = (
            random_gate_probability * random_utility
        )
        fixed = diagnostic["fixed"]
        _require(isinstance(fixed, Mapping), "invalid fixed-crop map")
        for action_id in sorted(action_ids):
            fixed_utility = float(fixed[action_id])
            values[f"fixed_crop_{action_id}_always_call"][key] = fixed_utility
            values[f"fixed_crop_{action_id}_matched_entropy_gate"][key] = (
                fixed_utility if entropy_called else 0.0
            )
    call_rate = calls / len(grouped) if grouped else 0.0
    executions = {
        PRIMARY: call_rate,
        "no_call": 0.0,
        "learned_crop_always_call": 1.0,
        "random_crop_always_call_expected": 1.0,
        "post_action_entropy_always_call_idealized": 1.0,
        "ug_exhaustive_entropy_four_calls": 4.0,
        "random_crop_same_learned_gate_expected": call_rate,
        "post_action_entropy_same_learned_gate_idealized": call_rate,
        "ug_exhaustive_entropy_same_learned_gate": 4.0 * call_rate,
        "matched_budget_entropy_gate_learned_crop": call_rate,
        "matched_budget_entropy_gate_random_crop": call_rate,
        "matched_budget_random_gate_random_crop_expected": call_rate,
    }
    for action_id in sorted(action_ids):
        executions[f"fixed_crop_{action_id}_always_call"] = 1.0
        executions[f"fixed_crop_{action_id}_matched_entropy_gate"] = call_rate
    diagnostics_summary = {
        "status": "locked_secondary_diagnostics_without_policy_tuning",
        "necessity_definition": (
            "at least one enumerated crop has gain minus 0.05 cost above zero"
        ),
        "positive_net_action_prevalence": positive_net_available / len(grouped),
        "call_precision_for_positive_net_availability": (
            calls_with_positive_net_available / calls if calls else 0.0
        ),
        "selected_positive_net_precision_per_call": (
            selected_positive_net_calls / calls if calls else 0.0
        ),
        "task_rescue_rate_per_call": rescue_calls / calls if calls else 0.0,
        "task_harm_rate_per_call": harm_calls / calls if calls else 0.0,
        "task_unchanged_rate_per_call": unchanged_calls / calls if calls else 0.0,
        "counts": {
            "decisions": len(grouped),
            "calls": calls,
            "positive_net_available": positive_net_available,
            "rescue_calls": rescue_calls,
            "harm_calls": harm_calls,
            "unchanged_calls": unchanged_calls,
        },
    }
    return values, source_by_key, executions, diagnostics_summary


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# ScreenQA formal paired paper analysis",
        "",
        f"> {report['analysis_status']}",
        "",
        f"Registered formal pass: **{report['primary_confirmation']['passed']}**.",
        "",
        "## Policy utilities",
        "",
        "| Policy | Source utility | 97.5% CI | Question utility | Executions |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in report["policies"].items():
        lines.append(
            "| {} | {} | [{}, {}] | {} | {} |".format(
                name,
                _fmt(float(row["source_utility"])),
                _fmt(float(row["ci_low"])),
                _fmt(float(row["ci_high"])),
                _fmt(float(row["question_utility"])),
                _fmt(float(row["mean_candidate_executions"])),
            )
        )
    lines.extend(
        [
            "",
            "## Paired whole-source comparisons",
            "",
            "These locked secondary comparisons do not replace the formal gate.",
            "",
            "| Comparator | Primary minus comparator | 97.5% paired CI | Holm p |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["paired_comparisons"].values():
        lines.append(
            "| {} | {} | [{}, {}] | {} |".format(
                row["comparator"],
                _fmt(float(row["source_utility_difference"])),
                _fmt(float(row["paired_ci_low"])),
                _fmt(float(row["paired_ci_high"])),
                _fmt(float(row["holm_adjusted_p_one_sided"])),
            )
        )
    lines.extend(
        [
            "",
            "Formal outcomes were used only for one-shot evaluation and this "
            "predeclared secondary analysis; no threshold, candidate, crop family, "
            "sample, or comparator was selected from them.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    _require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    _require(not temporary.exists(), f"temporary path exists: {temporary}")
    temporary.mkdir()
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "report.md").write_text(render_markdown(report), encoding="utf-8")
        with (temporary / "policy-table.csv").open(
            "x", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ("policy", "source_utility", "ci_low", "ci_high", "question_utility", "mean_candidate_executions")
            )
            for name, row in report["policies"].items():
                writer.writerow(
                    (
                        name,
                        row["source_utility"],
                        row["ci_low"],
                        row["ci_high"],
                        row["question_utility"],
                        row["mean_candidate_executions"],
                    )
                )
        with (temporary / "paired-comparisons.csv").open(
            "x", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ("comparator", "difference", "paired_ci_low", "paired_ci_high", "bootstrap_p_one_sided", "holm_adjusted_p_one_sided")
            )
            for row in report["paired_comparisons"].values():
                writer.writerow(
                    (
                        row["comparator"],
                        row["source_utility_difference"],
                        row["paired_ci_low"],
                        row["paired_ci_high"],
                        row["bootstrap_p_one_sided"],
                        row["holm_adjusted_p_one_sided"],
                    )
                )
        files = ("report.json", "report.md", "policy-table.csv", "paired-comparisons.csv")
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "screenqa_formal_paper_analysis_manifest_v1",
                    "files": {name: sha256_file(temporary / name) for name in files},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def analyze(
    evaluation_dir: Path,
    output_dir: Path,
    *,
    protocol: Path,
    expected_protocol_sha256: str,
) -> dict[str, Any]:
    _require(protocol.is_file(), "ScreenQA paper-analysis protocol is missing")
    _require(
        sha256_file(protocol) == expected_protocol_sha256,
        "ScreenQA paper-analysis protocol SHA-256 mismatch",
    )
    formal_report, model_path, rollouts_path = verify_formal_evaluation(evaluation_dir)
    model = _load_object(model_path, "calibrated model")
    records = read_jsonl(rollouts_path)
    values, source_by_key, executions, diagnostics = policy_decision_values(model, records)
    source_values = {
        policy: _source_means(decisions, source_by_key)
        for policy, decisions in values.items()
    }
    _require(len(values[PRIMARY]) == EXPECTED_DECISIONS, "reanalysis decision mismatch")
    _require(len(source_values[PRIMARY]) == EXPECTED_SOURCES, "reanalysis source mismatch")
    formal_source = formal_report.get("source_balanced")
    formal_question = formal_report.get("question_weighted")
    _require(
        isinstance(formal_source, Mapping) and isinstance(formal_question, Mapping),
        "formal report lacks metric maps",
    )
    assert isinstance(formal_source, Mapping)
    assert isinstance(formal_question, Mapping)
    crosschecks = {
        PRIMARY: "utility",
        "random_crop_always_call_expected": "random_always_call_utility",
        "post_action_entropy_always_call_idealized": (
            "post_action_entropy_always_call_utility"
        ),
        "ug_exhaustive_entropy_four_calls": "ug_style_exhaustive_entropy_utility",
        "matched_budget_entropy_gate_learned_crop": (
            "matched_budget_entropy_gate_learned_crop_utility"
        ),
        "matched_budget_entropy_gate_random_crop": (
            "matched_budget_entropy_gate_random_crop_utility"
        ),
        "matched_budget_random_gate_random_crop_expected": (
            "matched_budget_random_gate_random_crop_expected_utility"
        ),
    }
    for policy, metric in crosschecks.items():
        _require(
            math.isclose(
                mean(source_values[policy].values()),
                float(formal_source[metric]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"source crosscheck failed for {policy}",
        )
    _require(
        math.isclose(
            mean(values[PRIMARY].values()),
            float(formal_question["utility"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "question utility crosscheck failed",
    )
    intervals, comparisons = bootstrap_policy_table(source_values)
    policies = {
        policy: {
            **intervals[policy],
            "question_utility": mean(values[policy].values()),
            "mean_candidate_executions": executions[policy],
        }
        for policy in values
    }
    required = (
        "no_call",
        "random_crop_always_call_expected",
        "post_action_entropy_always_call_idealized",
        "ug_exhaustive_entropy_four_calls",
        "matched_budget_entropy_gate_learned_crop",
        "matched_budget_entropy_gate_random_crop",
        "matched_budget_random_gate_random_crop_expected",
    )
    evidence_gates = {
        "registered_formal_pass": formal_report.get("passed") is True,
        **{
            f"paired_interval_above_{name}": comparisons[
                f"{PRIMARY}_minus_{name}"
            ]["paired_interval_strictly_above_zero"]
            is True
            for name in required
        },
    }
    evidence_gates["all_required_secondary_comparisons_holm_0_025"] = all(
        comparisons[f"{PRIMARY}_minus_{name}"]["holm_reject_at_0_025"] is True
        for name in required
    )
    report = {
        "schema": SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "population": {"n_sources": EXPECTED_SOURCES, "n_decisions": EXPECTED_DECISIONS},
        "primary_confirmation": {
            "passed": formal_report.get("passed") is True,
            "pass_rule": formal_report.get("pass_rule"),
            "scientific_status": formal_report.get("scientific_status"),
        },
        "bootstrap": {
            "method": "paired_iid_whole_source_percentile_bootstrap",
            "n_resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": BOOTSTRAP_CONFIDENCE,
            "seed": BOOTSTRAP_SEED,
            "multiplicity": "Holm over all primary-minus-comparator one-sided bootstrap p-values",
        },
        "policies": policies,
        "paired_comparisons": comparisons,
        "evidence_gates": evidence_gates,
        "necessity_and_tool_effect": diagnostics,
        "inputs": {
            "protocol": str(protocol.resolve()),
            "protocol_sha256": expected_protocol_sha256,
            "formal_evaluation_dir": str(evaluation_dir.resolve()),
            "formal_report_sha256": sha256_file(evaluation_dir / "report.json"),
            "formal_completion_sha256": sha256_file(
                evaluation_dir / "formal-result.complete.json"
            ),
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
            "code_revision": _git_revision(),
        },
        "outcome_use": {
            "formal_outcomes_used_for_evaluation": True,
            "formal_outcomes_used_for_tuning": False,
            "threshold_changed": False,
            "candidate_changed": False,
            "sample_changed": False,
            "comparator_selected_from_formal_outcomes": False,
        },
    }
    write_outputs(output_dir, report)
    return report


def self_test() -> None:
    synthetic = {
        PRIMARY: {"s0": 0.4, "s1": 0.3, "s2": 0.5, "s3": 0.4},
        "no_call": {"s0": 0.0, "s1": 0.0, "s2": 0.0, "s3": 0.0},
        "baseline": {"s0": 0.1, "s1": 0.0, "s2": 0.2, "s3": 0.1},
    }
    first = bootstrap_policy_table(
        synthetic, n_resamples=2000, confidence_level=0.95, seed=17
    )
    second = bootstrap_policy_table(
        synthetic, n_resamples=2000, confidence_level=0.95, seed=17
    )
    _require(first == second, "bootstrap is not deterministic")
    policies, comparisons = first
    _require(policies[PRIMARY]["source_utility"] == 0.4, "primary mean is wrong")
    _require(
        all(row["paired_ci_low"] > 0.0 for row in comparisons.values()),
        "positive synthetic comparison did not pass",
    )
    _require(
        holm_adjust({"a": 0.01, "b": 0.03, "c": 0.02})
        == {"a": 0.03, "c": 0.04, "b": 0.04},
        "Holm adjustment mismatch",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the locked ScreenQA paired formal paper analysis"
    )
    parser.add_argument("--formal-evaluation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print(json.dumps({"self_test": "passed"}))
        return
    if (
        args.formal_evaluation_dir is None
        or args.output_dir is None
        or args.protocol is None
        or args.expected_protocol_sha256 is None
    ):
        raise ValueError(
            "formal evaluation, protocol, protocol hash, and output are required"
        )
    report = analyze(
        args.formal_evaluation_dir.resolve(),
        args.output_dir.resolve(),
        protocol=args.protocol.resolve(),
        expected_protocol_sha256=args.expected_protocol_sha256,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "registered_formal_pass": report["primary_confirmation"]["passed"],
                "paired_comparisons": len(report["paired_comparisons"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
