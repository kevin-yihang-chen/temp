from scripts.render_scaled_textvqa_formal_result import (
    render_scaled_formal_markdown,
)


def test_renderer_reports_frozen_pass_rule_and_integrity_hashes():
    source = {
        "utility": 0.02,
        "gain": 0.03,
        "call": 0.2,
        "induced_harm": 0.001,
        "net_negative_call": 0.01,
        "negative_net_value": 0.002,
        "oracle_utility": 0.1,
        "random_utility": -0.01,
        "entropy_search_utility": -0.2,
    }
    question = dict(source)
    evaluation = {
        "passed": True,
        "threshold": 0.1,
        "n_sources": 5000,
        "n_decisions": 7912,
        "source_balanced": source,
        "question_weighted": question,
        "source_bootstrap": {
            "n_resamples": 20000,
            "metrics": {"utility": {"ci_low": 0.01, "ci_high": 0.03}},
        },
        "pass_rule": {
            "source_utility_positive": True,
            "source_utility_97_5pct_ci_low_positive": True,
            "question_weighted_utility_positive": True,
            "source_call_rate_at_least_0_01": True,
        },
        "selection": {
            "positive_utility_call_precision": 0.8,
            "unnecessary_call_rate": 0.2,
            "correct_stopping_rate": 0.9,
            "source_balanced_raw_gain_per_call": 0.15,
        },
        "ranking": {
            "top1_rescue_rate_within_helpful_states": 0.7,
            "random_rescue_rate_within_helpful_states": 0.25,
            "fixed_crop_source_utilities": {"zoom-0": -0.01},
        },
        "oracle_regret": 0.08,
        "run": {
            "model_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "rollouts_sha256": "3" * 64,
            "features_sha256": "4" * 64,
            "protocol_sha256": "5" * 64,
            "evaluator_module_sha256": "6" * 64,
            "evaluator_script_sha256": "7" * 64,
        },
    }
    calibration = {
        "selected": {
            "source_call_rate": 0.2,
            "source_utility": 0.02,
            "risks": {
                "induced_harm": {
                    "source_balanced_mean": 0.001,
                    "limit": 0.005,
                    "passed": True,
                },
                "net_negative_call_mass": {
                    "source_balanced_mean": 0.01,
                    "limit": 0.02,
                    "passed": True,
                },
            },
        }
    }
    rendered = render_scaled_formal_markdown(
        evaluation,
        calibration,
        evaluation_sha256="a" * 64,
        calibration_sha256="b" * 64,
        policy_freeze_sha256="c" * 64,
    )
    assert "Preregistered verdict: PASS" in rendered
    assert "[0.010000, 0.030000]" in rendered
    assert "Earlier TextVQA and DocVQA formal failures remain" in rendered
    assert "`" + "a" * 64 + "`" in rendered
