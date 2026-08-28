import pytest

from scripts.fit_scaled_textvqa_learning_curves import registered_source_prefixes


def _allocation():
    return {
        "allocation": {
            "roles": {
                "ranker_training": {
                    "assignments": [
                        {
                            "source_group_id": f"source-{index}",
                            "source_rank": index,
                        }
                        for index in range(8)
                    ]
                }
            }
        }
    }


def test_registered_learning_curve_prefixes_are_nested_by_source_rank():
    prefixes = registered_source_prefixes(_allocation(), (2, 5, 8))
    assert prefixes[2] == ("source-0", "source-1")
    assert prefixes[5][:2] == prefixes[2]
    assert prefixes[8][:5] == prefixes[5]


def test_registered_learning_curve_rejects_non_increasing_counts():
    with pytest.raises(ValueError, match="unique and increasing"):
        registered_source_prefixes(_allocation(), (5, 2))
