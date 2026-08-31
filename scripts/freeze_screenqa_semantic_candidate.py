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


FEATURE_MODE = "hybrid-context-semantic"
TAIL_RATES = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_REVISION = re.compile(r"[0-9a-f]{40,64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def require_hash(path: Path, expected: str, name: str) -> str:
    if HEX_SHA256.fullmatch(expected) is None:
        raise ValueError(f"malformed expected SHA-256 for {name}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"ScreenQA semantic candidate {name} SHA-256 mismatch")
    return actual


def verify_semantic_candidate(
    *,
    report_path: Path,
    model_path: Path,
    features_path: Path,
    expected_features_sha256: str,
    label_free_audit_path: Path,
    expected_label_free_audit_sha256: str,
    activation_path: Path,
    expected_activation_sha256: str,
    protocol_path: Path,
    expected_protocol_sha256: str,
    expected_rollouts_sha256: str,
) -> dict[str, Any]:
    feature_sha256 = require_hash(
        features_path, expected_features_sha256, "feature file"
    )
    label_free_audit_sha256 = require_hash(
        label_free_audit_path,
        expected_label_free_audit_sha256,
        "label-free audit",
    )
    activation_sha256 = require_hash(
        activation_path, expected_activation_sha256, "activation audit"
    )
    protocol_sha256 = require_hash(
        protocol_path, expected_protocol_sha256, "v2 protocol"
    )
    activation = load_json(activation_path)
    expected_activation = {
        "passed": True,
        "semantic_escalation_activated": True,
        "ranker_rollouts_sha256": expected_rollouts_sha256,
        "v2_protocol_sha256": protocol_sha256,
        "records": 72_555,
        "decisions": 14_511,
        "sources": 1_510,
        "calibration_outcomes_opened": False,
        "formal_outcomes_opened": False,
        "reserve_outcomes_opened": False,
    }
    for key, expected in expected_activation.items():
        if activation.get(key) != expected:
            raise ValueError(f"ScreenQA semantic activation {key} mismatch")
    feature_revision = activation.get("semantic_code_revision")
    if not isinstance(feature_revision, str) or GIT_REVISION.fullmatch(feature_revision) is None:
        raise ValueError("ScreenQA semantic feature revision is malformed")

    label_free = load_json(label_free_audit_path)
    expected_label_free = {
        "features_sha256": feature_sha256,
        "rollouts_sha256": expected_rollouts_sha256,
        "decisions": 14_511,
        "outcome_fields_present": [],
        "outcomes_included_metadata": False,
    }
    for key, expected in expected_label_free.items():
        if label_free.get(key) != expected:
            raise ValueError(f"ScreenQA semantic label-free audit {key} mismatch")

    report = load_json(report_path)
    model = load_json(model_path)
    expected_report = {
        "feature_mode": FEATURE_MODE,
        "model_type": "multidomain_factorized_action_value",
        "training_protocol": "source_grouped_oof_domain_source_balanced_v2",
        "sample_weighting": "equal_domain_then_equal_source_then_equal_row",
        "seed": 20260831,
        "n_folds": 5,
        "lambda_cost": 0.05,
        "domains": ["screenqa"],
        "development_decisions": 14_511,
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            raise ValueError(f"ScreenQA semantic report {key} mismatch")
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
            raise ValueError(f"ScreenQA semantic model/report {key} mismatch")
    if (
        model.get("selected_alpha") != report.get("selected_alpha")
        or model.get("threshold") != report.get("selected_threshold")
    ):
        raise ValueError("ScreenQA semantic model/report selection mismatch")
    candidate_metrics = report.get("candidate_oof_metrics")
    if not isinstance(candidate_metrics, list) or sorted(
        float(row["alpha"]) for row in candidate_metrics
    ) != ALPHA_GRID:
        raise ValueError("ScreenQA semantic alpha grid mismatch")

    run = report.get("run")
    if not isinstance(run, Mapping) or run.get("formal_outcomes_used") is not False:
        raise ValueError("ScreenQA semantic formal outcome exclusion is not proven")
    code_revision = run.get("code_revision")
    if not isinstance(code_revision, str) or GIT_REVISION.fullmatch(code_revision) is None:
        raise ValueError("ScreenQA semantic fit revision is malformed")
    development = run.get("development_inputs")
    if not isinstance(development, Mapping) or set(development) != {"screenqa"}:
        raise ValueError("ScreenQA semantic development binding is malformed")
    if development["screenqa"].get("records") != 72_555 or development["screenqa"].get(
        "sha256"
    ) != expected_rollouts_sha256:
        raise ValueError("ScreenQA semantic ranker rollout binding mismatch")
    semantic_features = run.get("semantic_features")
    if not isinstance(semantic_features, Mapping) or set(semantic_features) != {
        "screenqa"
    }:
        raise ValueError("ScreenQA semantic feature binding is malformed")
    if semantic_features["screenqa"].get("sha256") != feature_sha256:
        raise ValueError("ScreenQA semantic fit used unexpected features")

    tail = report.get("development_tail_risk_diagnostic")
    if not isinstance(tail, Mapping):
        raise ValueError("ScreenQA semantic tail diagnostic is missing")
    expected_tail = {
        "family_error": 0.05,
        "lambda_cost": 0.05,
        "min_source_call_rate": 0.01,
        "min_source_utility": 0.001,
        "n_decisions": 14_511,
        "n_sources": 1_510,
        "selection_objective": "source_utility",
        "valid_for_formal_selection": False,
    }
    for key, expected in expected_tail.items():
        if tail.get(key) != expected:
            raise ValueError(f"ScreenQA semantic tail {key} mismatch")
    if tail.get("constraints") != [
        {"kind": "induced_harm", "limit": 0.005},
        {"kind": "net_negative_call_mass", "limit": 0.02},
    ]:
        raise ValueError("ScreenQA semantic risk constraints mismatch")
    requested = tail.get("requested_thresholds")
    if not isinstance(requested, list) or [
        float(row["target_pooled_call_rate"]) for row in requested
    ] != TAIL_RATES:
        raise ValueError("ScreenQA semantic tail call-rate grid mismatch")
    raw_thresholds = [float(row["threshold"]) for row in requested]
    if any(not math.isfinite(value) for value in raw_thresholds):
        raise ValueError("ScreenQA semantic threshold grid is non-finite")
    threshold_grid = list(dict.fromkeys(raw_thresholds))
    if any(
        strict <= permissive
        for strict, permissive in zip(threshold_grid, threshold_grid[1:])
    ):
        raise ValueError("ScreenQA semantic threshold grid is not descending")
    raw_selected = tail.get("selected")
    selected = dict(raw_selected) if isinstance(raw_selected, Mapping) else {}
    eligible = (
        tail.get("selection_status") == "selected_non_degenerate_safe_threshold"
        and selected.get("answer_now_only") is False
        and selected.get("risk_accepted") is True
        and float(selected.get("source_call_rate", 0.0)) >= 0.01
        and float(selected.get("source_utility", 0.0)) >= 0.001
        and isinstance(selected.get("threshold"), (int, float))
        and float(selected["threshold"]) in threshold_grid
    )
    return {
        "eligible": eligible,
        "feature_mode": FEATURE_MODE,
        "feature_revision": feature_revision,
        "fit_code_revision": code_revision,
        "features": str(features_path.resolve()),
        "features_sha256": feature_sha256,
        "label_free_audit": str(label_free_audit_path.resolve()),
        "label_free_audit_sha256": label_free_audit_sha256,
        "activation": str(activation_path.resolve()),
        "activation_sha256": activation_sha256,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha256,
        "report": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "raw_model": str(model_path.resolve()),
        "raw_model_sha256": sha256_file(model_path),
        "selected_alpha": report.get("selected_alpha"),
        "development_model_threshold": report.get("selected_threshold"),
        "tail_selection_status": tail.get("selection_status"),
        "tail_selected": selected,
        "threshold_grid": threshold_grid,
    }


def freeze_candidate(
    *,
    report_path: Path,
    model_path: Path,
    features_path: Path,
    expected_features_sha256: str,
    label_free_audit_path: Path,
    expected_label_free_audit_sha256: str,
    activation_path: Path,
    expected_activation_sha256: str,
    protocol_path: Path,
    expected_protocol_sha256: str,
    expected_rollouts_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite semantic candidate: {output_dir}")
    candidate = verify_semantic_candidate(
        report_path=report_path,
        model_path=model_path,
        features_path=features_path,
        expected_features_sha256=expected_features_sha256,
        label_free_audit_path=label_free_audit_path,
        expected_label_free_audit_sha256=expected_label_free_audit_sha256,
        activation_path=activation_path,
        expected_activation_sha256=expected_activation_sha256,
        protocol_path=protocol_path,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_rollouts_sha256=expected_rollouts_sha256,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    audit: dict[str, Any] = {
        "protocol_applied": True,
        "protocol_sha256": candidate["protocol_sha256"],
        "activation_sha256": candidate["activation_sha256"],
        "registered_feature_modes": [FEATURE_MODE],
        "candidate": candidate,
        "candidate_frozen": candidate["eligible"],
        "semantic_escalation_required": False,
        "further_ranker_search_allowed": False,
        "ranker_development_stopped": not candidate["eligible"],
        "calibration_outcomes_opened": False,
        "formal_outcomes_opened": False,
        "reserve_outcomes_opened": False,
    }
    if candidate["eligible"]:
        raw_model = load_json(model_path)
        frozen_model = dict(raw_model)
        frozen_model["development_oof_threshold"] = float(raw_model["threshold"])
        frozen_model["threshold"] = None
        frozen_model["decision_rule"] = (
            "factorized_expected_net_value_above_fixed_sequence_calibrated_margin"
        )
        frozen_model["threshold_grid"] = list(candidate["threshold_grid"])
        frozen_model["calibration_contract"] = {
            "method": "fixed_sequence_bounded_mean_kl_ltt_v1",
            "threshold_order": "strict_to_permissive_descending",
            "threshold_rate_weighting": "pooled_development_decisions",
            "target_pooled_development_call_rates": TAIL_RATES,
            "constraints": [
                {"kind": "induced_harm", "limit": 0.005},
                {"kind": "net_negative_call_mass", "limit": 0.02},
            ],
            "family_error": 0.05,
            "per_step_p_cutoff": 0.025,
            "min_source_call_rate": 0.01,
            "min_source_utility": 0.001,
            "calibration_sources": 1_016,
            "calibration_decisions": 9_951,
            "formal_sources": 1_471,
            "formal_decisions": 14_672,
        }
        frozen_model["candidate_selection"] = {
            "protocol_sha256": candidate["protocol_sha256"],
            "activation_sha256": candidate["activation_sha256"],
            "selected_feature_mode": FEATURE_MODE,
            "selected_development_safe_threshold": candidate["tail_selected"].get(
                "threshold"
            ),
            "selected_development_source_utility": candidate["tail_selected"].get(
                "source_utility"
            ),
            "selected_development_source_call_rate": candidate["tail_selected"].get(
                "source_call_rate"
            ),
            "ranker_rollouts_sha256": expected_rollouts_sha256,
            "semantic_features_sha256": candidate["features_sha256"],
            "semantic_feature_code_revision": candidate["feature_revision"],
            "ranker_fit_code_revision": candidate["fit_code_revision"],
            "raw_model_sha256": candidate["raw_model_sha256"],
            "ranker_report_sha256": candidate["report_sha256"],
            "ranker_training_outcomes_used": True,
            "calibration_outcomes_used": False,
            "formal_outcomes_used": False,
            "reserve_outcomes_used": False,
        }
        selected_model = output_dir / "model.json"
        selected_model.write_text(
            json.dumps(frozen_model, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        selected_report = output_dir / "selected-ranker-report.json"
        shutil.copyfile(report_path, selected_report)
        audit.update(
            {
                "selected_feature_mode": FEATURE_MODE,
                "selected_model": str(selected_model.resolve()),
                "selected_model_sha256": sha256_file(selected_model),
                "selected_ranker_report": str(selected_report.resolve()),
                "selected_ranker_report_sha256": sha256_file(selected_report),
                "threshold_grid": list(candidate["threshold_grid"]),
                "threshold_status": (
                    "execution threshold unset; fresh calibration may select only "
                    "from the frozen threshold grid"
                ),
            }
        )
    audit_path = output_dir / "candidate.audit.json"
    audit_path.write_text(
        json.dumps(audit, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.name}\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or terminate the sole ScreenQA semantic ranker candidate"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--label-free-audit", type=Path, required=True)
    parser.add_argument("--expected-label-free-audit-sha256", required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--expected-activation-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-rollouts-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = freeze_candidate(
        report_path=args.report,
        model_path=args.model,
        features_path=args.features,
        expected_features_sha256=args.expected_features_sha256,
        label_free_audit_path=args.label_free_audit,
        expected_label_free_audit_sha256=args.expected_label_free_audit_sha256,
        activation_path=args.activation,
        expected_activation_sha256=args.expected_activation_sha256,
        protocol_path=args.protocol,
        expected_protocol_sha256=args.expected_protocol_sha256,
        expected_rollouts_sha256=args.expected_rollouts_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
