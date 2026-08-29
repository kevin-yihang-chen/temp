from pathlib import Path


def test_docvqa_formal_rollout_audit_rebinds_the_complete_gate():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_docvqa_train_factorized_v2_formal_rollouts.py"
    ).read_text(encoding="utf-8")
    assert "validate_materialized_formal_gate(" in source
    assert "audit_sibling_rollout_bank(" in source
    assert '"records": args.expected_states * 5' in source
    assert '"unique_sources": FORMAL_SOURCES' in source
    assert '"unique_images": FORMAL_SOURCES' in source
    assert '"formal_outcomes_collected": True' in source
    assert '"formal_outcomes_used_for_tuning": False' in source
    assert 'with path.open("x"' in source
    for flag in (
        "--expected-policy-freeze-sha256",
        "--expected-model-sha256",
        "--expected-manifest-sha256",
        "--expected-manifest-provenance-sha256",
        "--expected-formal-audit-sha256",
        "--expected-code-revision",
    ):
        assert flag in source


def test_docvqa_formal_rollout_audit_resume_requires_exact_bytes():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_docvqa_train_factorized_v2_formal_rollouts.py"
    ).read_text(encoding="utf-8")
    assert "if not resume" in source
    assert "existing DocVQA formal rollout audit differs" in source
