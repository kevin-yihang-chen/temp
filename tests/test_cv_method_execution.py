import pytest

from scripts.execute_cv_method_stage import canonical_plan_methods


def test_plan_method_order_is_canonical_not_json_key_order():
    configs = {
        "counterfactual_utility": {},
        "evaluation": {},
        "factorized_potential_outcomes": {},
        "outcome_only": {},
    }
    assert canonical_plan_methods(configs) == (
        "outcome_only", "counterfactual_utility",
        "factorized_potential_outcomes",
    )


def test_plan_method_set_rejects_missing_control():
    with pytest.raises(ValueError, match="unsupported matched-arm plan"):
        canonical_plan_methods({
            "evaluation": {}, "factorized_potential_outcomes": {},
        })
