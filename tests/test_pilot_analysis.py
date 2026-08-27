from beyond_entropy.pilot_analysis import (
    analyze_counterfactual_pilot,
    build_pilot_markdown,
)
from beyond_entropy.simulate import simulate_counterfactual_dataset


def test_pilot_analysis_reports_strata_policies_and_transitions():
    records = simulate_counterfactual_dataset(
        n_states=12,
        num_candidates=4,
        questions_per_image=1,
        seed=13,
    )
    state_ids = sorted({record.state_id for record in records})
    strata = {
        state_id: "first" if index % 2 == 0 else "second"
        for index, state_id in enumerate(state_ids)
    }
    report = analyze_counterfactual_pilot(
        records,
        state_strata=strata,
        bootstrap_resamples=20,
        seed=3,
    )
    overall = report["overall"]
    assert overall["n_states"] == 12
    assert set(report["by_stratum"]) == {"first", "second"}
    assert sum(overall["transition_counts"].values()) == 48
    markdown = build_pilot_markdown(report)
    assert "Entropy search" in markdown
    assert "overall" in markdown
