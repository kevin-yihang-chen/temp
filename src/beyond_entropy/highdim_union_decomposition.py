from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .action_value import _validate_domains
from .decoupled_loss_gate import DECOUPLED_LAMBDA_COST, _evaluate
from .rescue_gate import DecisionKey
from .schema import ActionRecord


HIGHDIM_DECOMPOSITION_SEED = 20260910
HIGHDIM_DECOMPOSITION_BOOTSTRAP_RESAMPLES = 20_000
HIGHDIM_DECOMPOSITION_TARGET_CALLS = 225
HIGHDIM_DECOMPOSITION_DECISIONS = 13_580
HIGHDIM_DECOMPOSITION_SOURCES = 3_500
HIGHDIM_DECOMPOSITION_EQUAL_PAIRS = 4_875
HIGHDIM_DECOMPOSITION_UNEQUAL_PAIRS = 8_705
HIGHDIM_DECOMPOSITION_UNION_ROWS = 22_285


_FORBIDDEN_SCORE_FIELDS = {
    "correct_before",
    "correct_after",
    "target",
    "reward",
    "gain",
    "harm",
    "answer_before",
    "answer_after",
    "oracle_action_id",
    "entropy_after",
    "delta_success",
    "utility",
}


def _rename_candidate(value: Any, candidate_name: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key).replace("decoupled", candidate_name): _rename_candidate(
                item, candidate_name
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_candidate(item, candidate_name) for item in value]
    return value


def _score_index(
    rows: Sequence[Mapping[str, Any]],
    keys: set[DecisionKey],
) -> dict[str, dict[DecisionKey, Any]]:
    result: dict[str, dict[DecisionKey, Any]] = {
        "incumbent_action": {},
        "highdim_action": {},
        "incumbent_call": {},
        "highdim_call": {},
        "incumbent_score": {},
        "highdim_score": {},
        "union": {},
    }
    for row in rows:
        if _FORBIDDEN_SCORE_FIELDS.intersection(row):
            raise ValueError("highdim decomposition score rows leak outcomes")
        key = (str(row.get("state_id", "")), str(row.get("replicate_id", "")))
        if not all(key) or key in result["incumbent_action"]:
            raise ValueError("highdim decomposition score identity is invalid")
        incumbent_action = str(row.get("incumbent_action_id", ""))
        highdim_action = str(row.get("highdim_diagonal_bilinear_action_id", ""))
        incumbent_proposal = str(row.get("incumbent_proposal_action_id", ""))
        loss_proposal = str(row.get("loss_proposal_action_id", ""))
        if not all(
            (incumbent_action, highdim_action, incumbent_proposal, loss_proposal)
        ):
            raise ValueError("highdim decomposition action identity is invalid")
        incumbent_called = row.get("incumbent_called")
        highdim_called = row.get("highdim_diagonal_bilinear_called")
        if not isinstance(incumbent_called, bool) or not isinstance(
            highdim_called, bool
        ):
            raise ValueError("highdim decomposition call flag is invalid")
        incumbent_score = float(row.get("incumbent_score", math.nan))
        highdim_score = float(
            row.get("highdim_diagonal_bilinear_score", math.nan)
        )
        if not math.isfinite(incumbent_score) or not math.isfinite(highdim_score):
            raise ValueError("highdim decomposition score is non-finite")
        union = tuple(sorted({incumbent_proposal, loss_proposal}))
        if len(union) not in (1, 2) or highdim_action not in union:
            raise ValueError("highdim decomposition union is invalid")
        result["incumbent_action"][key] = incumbent_action
        result["highdim_action"][key] = highdim_action
        result["incumbent_call"][key] = incumbent_called
        result["highdim_call"][key] = highdim_called
        result["incumbent_score"][key] = incumbent_score
        result["highdim_score"][key] = highdim_score
        result["union"][key] = union
    if any(set(items) != keys for items in result.values()):
        raise ValueError("highdim decomposition scores do not cover decisions")
    return result


def _union_oracle_actions(
    union: Mapping[DecisionKey, Sequence[str]],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
) -> dict[DecisionKey, str]:
    selected: dict[DecisionKey, str] = {}
    for key, action_ids in union.items():
        candidates = [
            action
            for action in zooms[key]
            if action.action_id in set(action_ids)
        ]
        if len(candidates) != len(action_ids):
            raise ValueError(f"union action coverage is invalid for {key!r}")
        selected[key] = min(
            candidates,
            key=lambda action: (-action.voi(DECOUPLED_LAMBDA_COST), action.action_id),
        ).action_id
    return selected


def _positive_utility_calls(
    actions: Mapping[DecisionKey, str],
    zooms: Mapping[DecisionKey, Sequence[ActionRecord]],
) -> dict[DecisionKey, bool]:
    calls: dict[DecisionKey, bool] = {}
    for key, action_id in actions.items():
        matches = [action for action in zooms[key] if action.action_id == action_id]
        if len(matches) != 1:
            raise ValueError(f"selected action coverage is invalid for {key!r}")
        calls[key] = matches[0].voi(DECOUPLED_LAMBDA_COST) > 0.0
    return calls


def decompose_highdim_union(
    records: Sequence[ActionRecord],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = HIGHDIM_DECOMPOSITION_BOOTSTRAP_RESAMPLES,
    seed: int = HIGHDIM_DECOMPOSITION_SEED,
) -> dict[str, Any]:
    if bootstrap_resamples != HIGHDIM_DECOMPOSITION_BOOTSTRAP_RESAMPLES:
        raise ValueError("highdim decomposition bootstrap count is frozen")
    if seed != HIGHDIM_DECOMPOSITION_SEED:
        raise ValueError("highdim decomposition seed is frozen")
    _domains, baselines, zooms = _validate_domains({"docvqa": records})
    keys = set(baselines)
    if (
        len(keys) != HIGHDIM_DECOMPOSITION_DECISIONS
        or len({record.source_id for record in baselines.values()})
        != HIGHDIM_DECOMPOSITION_SOURCES
        or any(len(actions) != 4 for actions in zooms.values())
    ):
        raise ValueError("highdim decomposition population contract changed")
    indexed = _score_index(score_rows, keys)
    incumbent_calls = indexed["incumbent_call"]
    highdim_calls = indexed["highdim_call"]
    if (
        sum(bool(value) for value in incumbent_calls.values())
        != HIGHDIM_DECOMPOSITION_TARGET_CALLS
        or sum(bool(value) for value in highdim_calls.values())
        != HIGHDIM_DECOMPOSITION_TARGET_CALLS
    ):
        raise ValueError("highdim decomposition matched-call contract changed")
    union = indexed["union"]
    equal_pairs = sum(len(actions) == 1 for actions in union.values())
    union_rows = sum(len(actions) for actions in union.values())
    if (
        equal_pairs != HIGHDIM_DECOMPOSITION_EQUAL_PAIRS
        or len(union) - equal_pairs != HIGHDIM_DECOMPOSITION_UNEQUAL_PAIRS
        or union_rows != HIGHDIM_DECOMPOSITION_UNION_ROWS
    ):
        raise ValueError("highdim decomposition union contract changed")
    incumbent_actions = indexed["incumbent_action"]
    highdim_actions = indexed["highdim_action"]
    union_oracle_actions = _union_oracle_actions(union, zooms)
    oracle_highdim_calls = _positive_utility_calls(highdim_actions, zooms)
    oracle_union_calls = _positive_utility_calls(union_oracle_actions, zooms)

    candidates = {
        "highdim_full": (highdim_actions, highdim_calls),
        "incumbent_call_highdim_action": (highdim_actions, incumbent_calls),
        "highdim_call_incumbent_action": (incumbent_actions, highdim_calls),
        "incumbent_call_union_oracle_action": (
            union_oracle_actions,
            incumbent_calls,
        ),
        "highdim_call_union_oracle_action": (
            union_oracle_actions,
            highdim_calls,
        ),
        "oracle_call_highdim_action": (highdim_actions, oracle_highdim_calls),
        "oracle_call_union_oracle_action": (
            union_oracle_actions,
            oracle_union_calls,
        ),
    }
    indicator = {
        "incumbent": {key: float(value) for key, value in incumbent_calls.items()}
    }
    comparisons: dict[str, Any] = {}
    for candidate_name, (actions, calls) in candidates.items():
        evaluated = _evaluate(
            baselines=baselines,
            zooms=zooms,
            actions_by_method={
                "incumbent": incumbent_actions,
                "decoupled": actions,
            },
            scores_by_method={
                "incumbent": indicator["incumbent"],
                "decoupled": {key: float(value) for key, value in calls.items()},
            },
            threshold_by_method={"incumbent": 0.5, "decoupled": 0.5},
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=seed,
        )
        comparisons[candidate_name] = _rename_candidate(
            {key: value for key, value in evaluated.items() if key != "score_rows"},
            candidate_name,
        )

    return {
        "scientific_status": (
            "post-hoc opened-development stopping-versus-action decomposition; "
            "oracle rows are ceilings and cannot support a deployable claim"
        ),
        "n_sources": HIGHDIM_DECOMPOSITION_SOURCES,
        "n_decisions": HIGHDIM_DECOMPOSITION_DECISIONS,
        "lambda_cost": DECOUPLED_LAMBDA_COST,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": seed,
        "union_counts": {
            "equal_proposal_pairs": equal_pairs,
            "unequal_proposal_pairs": len(union) - equal_pairs,
            "unique_union_rows": union_rows,
        },
        "call_counts": {
            "incumbent": sum(bool(value) for value in incumbent_calls.values()),
            "highdim": sum(bool(value) for value in highdim_calls.values()),
            "oracle_highdim_action": sum(
                bool(value) for value in oracle_highdim_calls.values()
            ),
            "oracle_union": sum(bool(value) for value in oracle_union_calls.values()),
        },
        "comparisons": comparisons,
        "audits": {
            "score_rows_outcome_free": True,
            "score_coverage_exact": True,
            "population_exact": True,
            "matched_call_counts_exact": True,
            "union_cardinality_exact": True,
            "screenqa_inputs_used": False,
            "protected_role_inputs_used": False,
        },
    }
