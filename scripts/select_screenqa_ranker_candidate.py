#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Mapping


EXPECTED_FEATURE_MODES = ("context-geometry", "spatial-context-geometry")
EXPECTED_TAIL_RATES = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
EXPECTED_ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
UTILITY_TIE_MARGIN = 0.00025
GIT_REVISION = re.compile(r"[0-9a-f]{40,64}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _candidate(
    *,
    feature_mode: str,
    report_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    report = _load_json(report_path)
    model = _load_json(model_path)
    if feature_mode not in EXPECTED_FEATURE_MODES:
        raise ValueError(f"unregistered ScreenQA feature mode: {feature_mode}")
    expected_report = {
        "feature_mode": feature_mode,
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "seed": 20260831,
        "n_folds": 5,
        "lambda_cost": 0.05,
        "domains": ["screenqa"],
        "development_decisions": 14511,
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            raise ValueError(
                f"{feature_mode} report {key} mismatch: "
                f"expected {expected!r}, got {report.get(key)!r}"
            )
    for key in (
        "feature_mode",
        "model_type",
        "training_protocol",
        "sample_weighting",
        "seed",
        "n_folds",
        "lambda_cost",
        "domains",
    ):
        if model.get(key) != report.get(key):
            raise ValueError(f"{feature_mode} model/report {key} mismatch")
    if (
        model.get("selected_alpha") != report.get("selected_alpha")
        or model.get("threshold") != report.get("selected_threshold")
    ):
        raise ValueError(f"{feature_mode} model/report selection mismatch")
    candidate_metrics = report.get("candidate_oof_metrics")
    if not isinstance(candidate_metrics, list):
        raise ValueError(f"{feature_mode} candidate metrics are missing")
    observed_alpha_grid = sorted(float(row["alpha"]) for row in candidate_metrics)
    if observed_alpha_grid != EXPECTED_ALPHA_GRID:
        raise ValueError(f"{feature_mode} alpha grid mismatch")
    run = report.get("run")
    if not isinstance(run, Mapping) or run.get("formal_outcomes_used") is not False:
        raise ValueError(f"{feature_mode} formal outcome exclusion is not proven")
    development_inputs = run.get("development_inputs")
    if not isinstance(development_inputs, Mapping) or set(development_inputs) != {
        "screenqa"
    }:
        raise ValueError(f"{feature_mode} development input binding is malformed")
    input_payload = development_inputs["screenqa"]
    if not isinstance(input_payload, Mapping) or input_payload.get("records") != 72555:
        raise ValueError(f"{feature_mode} ranker record count mismatch")
    code_revision = run.get("code_revision")
    rollouts_sha256 = input_payload.get("sha256")
    if not isinstance(code_revision, str) or GIT_REVISION.fullmatch(code_revision) is None:
        raise ValueError(f"{feature_mode} code revision is malformed")
    if not isinstance(rollouts_sha256, str) or SHA256.fullmatch(rollouts_sha256) is None:
        raise ValueError(f"{feature_mode} rollout hash is malformed")

    tail = report.get("development_tail_risk_diagnostic")
    if not isinstance(tail, Mapping):
        raise ValueError(f"{feature_mode} tail risk diagnostic is missing")
    expected_tail = {
        "family_error": 0.05,
        "lambda_cost": 0.05,
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "n_decisions": 14511,
        "n_sources": 1510,
        "selection_objective": "source_utility",
        "valid_for_formal_selection": False,
    }
    for key, expected in expected_tail.items():
        if tail.get(key) != expected:
            raise ValueError(f"{feature_mode} tail diagnostic {key} mismatch")
    requested = tail.get("requested_thresholds")
    if not isinstance(requested, list) or not all(
        isinstance(row, Mapping) for row in requested
    ):
        raise ValueError(f"{feature_mode} tail call-rate grid is malformed")
    if [float(row["target_pooled_call_rate"]) for row in requested] != EXPECTED_TAIL_RATES:
        raise ValueError(f"{feature_mode} tail call-rate grid mismatch")
    raw_thresholds = [float(row["threshold"]) for row in requested]
    if any(not math.isfinite(value) for value in raw_thresholds):
        raise ValueError(f"{feature_mode} tail threshold grid is non-finite")
    threshold_grid = list(dict.fromkeys(raw_thresholds))
    if any(
        strict <= permissive
        for strict, permissive in zip(threshold_grid, threshold_grid[1:])
    ):
        raise ValueError(f"{feature_mode} tail threshold grid is not descending")
    constraints = tail.get("constraints")
    if constraints != [
        {"kind": "induced_harm", "limit": 0.005},
        {"kind": "net_negative_call_mass", "limit": 0.02},
    ]:
        raise ValueError(f"{feature_mode} risk constraints mismatch")
    selected = tail.get("selected")
    selected_payload = selected if isinstance(selected, Mapping) else {}
    source_utility = float(selected_payload.get("source_utility", float("-inf")))
    source_call_rate = float(selected_payload.get("source_call_rate", 0.0))
    selected_tail_threshold = selected_payload.get("threshold")
    if selected_payload.get("answer_now_only") is False and (
        not isinstance(selected_tail_threshold, (int, float))
        or not math.isfinite(float(selected_tail_threshold))
        or float(selected_tail_threshold) not in threshold_grid
    ):
        raise ValueError(f"{feature_mode} selected safe threshold is not registered")
    eligible = (
        tail.get("selection_status") == "selected_non_degenerate_safe_threshold"
        and selected_payload.get("answer_now_only") is False
        and selected_payload.get("risk_accepted") is True
        and source_call_rate >= 0.01
        and source_utility >= 0.001
    )
    return {
        "feature_mode": feature_mode,
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "code_revision": code_revision,
        "rollouts_sha256": rollouts_sha256,
        "selected_alpha": report["selected_alpha"],
        "model_threshold": report["selected_threshold"],
        "tail_selection_status": tail.get("selection_status"),
        "tail_selected": dict(selected_payload),
        "threshold_grid": threshold_grid,
        "source_utility": source_utility,
        "source_call_rate": source_call_rate,
        "eligible": eligible,
    }


def select_candidate(
    *,
    context_report: Path,
    context_model: Path,
    spatial_report: Path,
    spatial_model: Path,
    protocol: Path,
    expected_protocol_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate output: {output_dir}")
    actual_protocol_sha256 = sha256_file(protocol)
    if actual_protocol_sha256 != expected_protocol_sha256:
        raise ValueError("ScreenQA ranker-development protocol hash mismatch")
    candidates = [
        _candidate(
            feature_mode="context-geometry",
            report_path=context_report,
            model_path=context_model,
        ),
        _candidate(
            feature_mode="spatial-context-geometry",
            report_path=spatial_report,
            model_path=spatial_model,
        ),
    ]
    if len({candidate["code_revision"] for candidate in candidates}) != 1:
        raise ValueError("ScreenQA candidate code revisions differ")
    if len({candidate["rollouts_sha256"] for candidate in candidates}) != 1:
        raise ValueError("ScreenQA candidate rollout inputs differ")
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    winner: dict[str, Any] | None = None
    selection_reason = "no_registered_candidate_is_eligible"
    if len(eligible) == 1:
        winner = eligible[0]
        selection_reason = "only_registered_eligible_candidate"
    elif len(eligible) == 2:
        context = next(
            candidate
            for candidate in eligible
            if candidate["feature_mode"] == "context-geometry"
        )
        spatial = next(
            candidate
            for candidate in eligible
            if candidate["feature_mode"] == "spatial-context-geometry"
        )
        if abs(context["source_utility"] - spatial["source_utility"]) < UTILITY_TIE_MARGIN:
            winner = context
            selection_reason = "utility_tie_within_margin_choose_lower_capacity"
        else:
            winner = max(eligible, key=lambda candidate: candidate["source_utility"])
            selection_reason = "largest_registered_source_utility"

    output_dir.mkdir(parents=True, exist_ok=False)
    audit: dict[str, Any] = {
        "protocol_applied": True,
        "protocol": str(protocol.resolve()),
        "protocol_sha256": actual_protocol_sha256,
        "registered_feature_modes": list(EXPECTED_FEATURE_MODES),
        "utility_tie_margin": UTILITY_TIE_MARGIN,
        "candidates": candidates,
        "selection_reason": selection_reason,
        "candidate_frozen": winner is not None,
        "semantic_escalation_required": winner is None,
        "calibration_outcomes_opened": False,
        "formal_outcomes_opened": False,
        "reserve_outcomes_opened": False,
    }
    if winner is not None:
        selected_model = output_dir / "model.json"
        selected_report = output_dir / "selected-ranker-report.json"
        raw_model = _load_json(Path(winner["model_path"]))
        frozen_model = dict(raw_model)
        frozen_model["development_oof_threshold"] = float(raw_model["threshold"])
        frozen_model["threshold"] = None
        frozen_model["decision_rule"] = (
            "factorized_expected_net_value_above_fixed_sequence_calibrated_margin"
        )
        frozen_model["threshold_grid"] = list(winner["threshold_grid"])
        frozen_model["calibration_contract"] = {
            "method": "fixed_sequence_bounded_mean_kl_ltt_v1",
            "threshold_order": "strict_to_permissive_descending",
            "threshold_rate_weighting": "pooled_development_decisions",
            "target_pooled_development_call_rates": list(EXPECTED_TAIL_RATES),
            "constraints": [
                {"kind": "induced_harm", "limit": 0.005},
                {"kind": "net_negative_call_mass", "limit": 0.02},
            ],
            "family_error": 0.05,
            "per_step_p_cutoff": 0.025,
            "min_source_call_rate": 0.01,
            "min_source_utility": 0.001,
            "calibration_sources": 1016,
            "calibration_decisions": 9951,
            "formal_sources": 1471,
            "formal_decisions": 14672,
        }
        frozen_model["candidate_selection"] = {
            "protocol_sha256": actual_protocol_sha256,
            "selected_feature_mode": winner["feature_mode"],
            "selected_development_safe_threshold": winner["tail_selected"].get(
                "threshold"
            ),
            "selected_development_source_utility": winner["source_utility"],
            "selected_development_source_call_rate": winner["source_call_rate"],
            "ranker_rollouts_sha256": winner["rollouts_sha256"],
            "ranker_fit_code_revision": winner["code_revision"],
            "raw_model_sha256": winner["model_sha256"],
            "ranker_report_sha256": winner["report_sha256"],
            "ranker_training_outcomes_used": True,
            "calibration_outcomes_used": False,
            "formal_outcomes_used": False,
            "reserve_outcomes_used": False,
        }
        selected_model.write_text(
            json.dumps(frozen_model, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(
            context_report
            if winner["feature_mode"] == "context-geometry"
            else spatial_report,
            selected_report,
        )
        audit["selected_feature_mode"] = winner["feature_mode"]
        audit["selected_model"] = str(selected_model.resolve())
        audit["selected_model_sha256"] = sha256_file(selected_model)
        audit["raw_selected_model_sha256"] = winner["model_sha256"]
        audit["selected_ranker_report"] = str(selected_report.resolve())
        audit["selected_ranker_report_sha256"] = sha256_file(selected_report)
        audit["threshold_grid"] = list(winner["threshold_grid"])
        audit["threshold_status"] = (
            "execution threshold unset; fresh calibration may select only from the "
            "frozen threshold grid"
        )
    audit_path = output_dir / "candidate.audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.name}\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen ScreenQA low-capacity ranker selection rule"
    )
    parser.add_argument("--context-report", type=Path, required=True)
    parser.add_argument("--context-model", type=Path, required=True)
    parser.add_argument("--spatial-report", type=Path, required=True)
    parser.add_argument("--spatial-model", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = select_candidate(
        context_report=args.context_report,
        context_model=args.context_model,
        spatial_report=args.spatial_report,
        spatial_model=args.spatial_model,
        protocol=args.protocol,
        expected_protocol_sha256=args.expected_protocol_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
