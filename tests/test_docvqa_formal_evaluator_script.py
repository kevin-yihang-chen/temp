from pathlib import Path


def test_docvqa_formal_evaluator_freezes_population_bootstrap_and_pass_rule():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluate_docvqa_train_factorized_v2_formal.py"
    ).read_text(encoding="utf-8")
    assert "validate_materialized_formal_gate(" in source
    assert "audit_sibling_rollout_bank(" in source
    assert "validate_semantic_feature_dataset(features, records)" in source
    assert 'metadata.get("outcomes_included"), False' in source
    assert 'model.get("domains"), ["docvqa"]' in source
    assert "bootstrap_resamples=args.bootstrap_resamples" in source
    assert "BOOTSTRAP_RESAMPLES" in source
    assert "BOOTSTRAP_CONFIDENCE" in source
    assert "BOOTSTRAP_SEED" in source
    assert 'pass_rule["threshold_matches_calibration_choice"] = True' in source
    assert (
        'pass_rule["all_frozen_hashes_and_identity_audits_match"] = True'
        in source
    )
    assert 'report["passed"] = all(' in source
    assert 'with output_path.open("x"' in source


def test_docvqa_formal_evaluator_requires_every_bound_hash_argument():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluate_docvqa_train_factorized_v2_formal.py"
    ).read_text(encoding="utf-8")
    for flag in (
        "--expected-policy-freeze-sha256",
        "--expected-model-sha256",
        "--expected-manifest-sha256",
        "--expected-manifest-provenance-sha256",
        "--expected-formal-audit-sha256",
        "--expected-rollouts-sha256",
        "--expected-rollout-audit-sha256",
        "--expected-features-sha256",
    ):
        assert flag in source
