from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "render_docvqa_train_factorized_v2_formal.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("docvqa_formal_renderer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report():
    digest = "a" * 64
    return {
        "passed": True,
        "threshold": 0.1,
        "lambda_cost": 0.05,
        "n_sources": 3500,
        "n_decisions": 4000,
        "source_balanced": {
            "utility": 0.01,
            "call": 0.1,
            "gain": 0.015,
            "baseline_accuracy": 0.7,
            "policy_accuracy": 0.715,
            "oracle_utility": 0.04,
        },
        "question_weighted": {"utility": 0.01},
        "source_bootstrap": {
            "n_resamples": 20000,
            "confidence_level": 0.975,
            "seed": 20260829,
            "metrics": {"utility": {"ci_low": 0.001, "ci_high": 0.02}},
        },
        "risk_diagnostics": {
            "source_balanced_induced_harm_mass": 0.001,
            "source_balanced_net_negative_call_mass": 0.01,
        },
        "ranking": {
            "top1_rescue_rate_within_helpful_states": 0.5,
            "random_rescue_rate_within_helpful_states": 0.25,
        },
        "baselines": {
            "ug_style_exhaustive_candidate_count": 4,
            "ug_style_exhaustive_search_charged_all_candidate_costs": True,
            "ug_style_exhaustive_entropy_source_gain": 0.02,
            "ug_style_exhaustive_entropy_source_utility": -0.18,
            "matched_budget_call_count": 400,
            "matched_budget_entropy_gate_source_utility_learned_crop": 0.001,
            "matched_budget_entropy_gate_source_utility_random_crop": -0.002,
            "matched_budget_random_gate_source_utility_random_crop_expected": -0.003,
            "fixed_crop_source_utility_entropy_gate": {},
            "fixed_crop_source_utility_always_call": {},
            "fixed_crop_source_utility_same_gate": {},
        },
        "selection": {
            "calls": 400,
            "source_balanced_raw_gain_per_call": 0.15,
            "unnecessary_call_rate": 0.3,
            "positive_utility_call_precision": 0.7,
            "correct_stopping_rate": 0.9,
        },
        "oracle_regret": 0.03,
        "pass_rule": {
            "source_utility_positive": True,
            "source_utility_97_5pct_ci_low_positive": True,
            "question_weighted_utility_positive": True,
            "source_call_rate_at_least_0_01": True,
            "threshold_matches_calibration_choice": True,
            "all_frozen_hashes_and_identity_audits_match": True,
        },
        "run": {
            "formal_outcomes_used": True,
            "no_target_derived_tuning": True,
            "feature_outcomes_included": False,
            "bootstrap_resamples": 20000,
            "bootstrap_confidence": 0.975,
            "bootstrap_seed": 20260829,
            "policy_freeze_sha256": digest,
            "model_sha256": digest,
            "manifest_sha256": digest,
            "manifest_provenance_sha256": digest,
            "formal_audit_sha256": digest,
            "rollouts_sha256": digest,
            "rollout_audit_sha256": digest,
            "features_sha256": digest,
            "code_revision": "b" * 40,
        },
    }


def test_docvqa_formal_renderer_covers_primary_diagnostics_and_scope():
    module = _module()
    rendered = module.render_report(_report())
    assert "Decision: **PASS**" in rendered
    assert "Source-balanced utility" in rendered
    assert "UG-style exhaustive entropy" in rendered
    assert "cross-benchmark claim" in rendered
    assert "Formal outcomes used for tuning: `false`" in rendered


def test_docvqa_formal_renderer_rejects_bootstrap_or_decision_drift():
    module = _module()
    report = _report()
    report["source_bootstrap"]["seed"] = 0
    with pytest.raises(ValueError, match="bootstrap"):
        module.validate_formal_report(report)
    report = _report()
    report["passed"] = False
    with pytest.raises(ValueError, match="decision"):
        module.validate_formal_report(report)


def test_docvqa_formal_renderer_rejects_missing_baseline():
    module = _module()
    report = _report()
    del report["baselines"]["fixed_crop_source_utility_same_gate"]
    with pytest.raises(ValueError, match="baselines"):
        module.validate_formal_report(report)
