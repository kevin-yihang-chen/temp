from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from analyze_chartqapro_pilot import (
    EXPECTED_MODEL_SHA256,
    EXPECTED_SOURCE_REPORT_SHA256,
    _evaluate,
    _output_compatibility,
    _policy_rows,
    _read_manifest,
    _read_object,
    _rescore_spec,
    _sha256,
    _validate_inputs,
    _write_json,
)
from beyond_entropy.dataset import read_jsonl


EXPECTED_MANIFEST_SHA256 = (
    "5a3ddca2e6476196aac8ad4fa7bc00033f2ac9c39d2011fe21fa070e965b97d4"
)
EXPECTED_PILOT_REPORT_SHA256 = (
    "93e6f04989fa00c247406baaad2815a486b8d145bf8fa932b83648cf5995fe99"
)
EXPECTED_REPLAY_AUDIT_SHA256 = (
    "173ff249f1fb8c25b73abdc28f32d705bd3d25737dea6d3bd58b8ce042106480"
)
EXPECTED_ROLLOUT_CODE_REVISION = "d9b35b8e735848872e5ea315cfd56cd0398512a6"
EXPECTED_STATES = 1625


def _primary_criterion(evaluation: Mapping[str, Any]) -> dict[str, bool]:
    policies = evaluation.get("policies")
    image_bootstrap = evaluation.get("primary_image_bootstrap")
    if not isinstance(policies, Mapping) or not isinstance(image_bootstrap, Mapping):
        raise ValueError("formal evaluation is missing policies or image bootstrap")
    primary = policies.get("frozen_factorized_context")
    always_random = policies.get("always_random")
    exhaustive = policies.get("exhaustive_entropy")
    if not all(
        isinstance(value, Mapping)
        for value in (primary, always_random, exhaustive)
    ):
        raise ValueError("formal evaluation is missing a required policy")
    assert isinstance(primary, Mapping)
    assert isinstance(always_random, Mapping)
    assert isinstance(exhaustive, Mapping)
    primary_bootstrap = primary.get("bootstrap")
    if not isinstance(primary_bootstrap, Mapping):
        raise ValueError("primary policy is missing question bootstrap")
    question_metrics = primary_bootstrap.get("metrics")
    image_metrics = image_bootstrap.get("metrics")
    if not isinstance(question_metrics, Mapping) or not isinstance(
        image_metrics,
        Mapping,
    ):
        raise ValueError("formal bootstrap metrics are missing")
    question_utility = question_metrics.get("mean_policy_utility")
    image_utility = image_metrics.get("mean_policy_utility")
    if not isinstance(question_utility, Mapping) or not isinstance(
        image_utility,
        Mapping,
    ):
        raise ValueError("formal utility intervals are missing")
    criterion = {
        "positive_mean_utility": float(primary["mean_policy_utility"]) > 0.0,
        "question_bootstrap_utility_lower_above_zero": (
            float(question_utility["ci_low"]) > 0.0
        ),
        "image_bootstrap_utility_lower_above_zero": (
            float(image_utility["ci_low"]) > 0.0
        ),
        "positive_released_score_gain": float(primary["accuracy_gain"]) > 0.0,
        "lower_tool_use_than_unconditional_one_crop": (
            float(primary["tool_use_rate"])
            < float(always_random["tool_use_rate"])
        ),
        "lower_tool_use_than_exhaustive_four_crop": (
            float(primary["tool_use_rate"])
            < float(exhaustive["tool_use_rate"])
        ),
    }
    criterion["passed"] = all(criterion.values())
    return criterion


def _build_markdown(report: Mapping[str, Any]) -> str:
    criterion = report["primary_confirmation_criterion"]
    compatibility = report["output_diagnostics"]
    assert isinstance(criterion, Mapping)
    assert isinstance(compatibility, Mapping)
    lines = [
        "# ChartQAPro untouched formal evaluation",
        "",
        "> Frozen 1,625-question formal target; no target-derived tuning.",
        "",
        f"- Primary confirmation passed: **{criterion['passed']}**",
        f"- Empty outputs: {compatibility['empty_outputs']}",
        "- Baseline max-token cap rate: {:.4f}".format(
            compatibility["baseline_max_token_capped_rate"]
        ),
        "- Raw constrained-format compliance: {:.4f}".format(
            compatibility["constrained_format_compliance"]
        ),
        "- Conservative canonical-parse compliance: {:.4f}".format(
            compatibility["constrained_canonical_format_compliance"]
        ),
        "",
        "## Frozen primary criterion",
        "",
    ]
    for name, passed in criterion.items():
        lines.append(f"- {name}: **{passed}**")
    lines.append("")
    for scorer_name in ("released", "paper_spec", "paper_spec_canonical"):
        evaluation = report[scorer_name]
        assert isinstance(evaluation, Mapping)
        lines.extend(
            [
                f"## {scorer_name.replace('_', ' ').title()} scorer",
                "",
                "| Policy | Score gain | Tool rate | Utility |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, result in _policy_rows(evaluation):
            lines.append(
                "| {} | {:.4f} | {:.4f} | {:.4f} |".format(
                    name,
                    result["accuracy_gain"],
                    result["tool_use_rate"],
                    result["mean_policy_utility"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen ChartQAPro untouched formal target",
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actual_hashes = {
        "manifest": _sha256(args.manifest),
        "frozen_model": _sha256(args.frozen_model),
        "source_report": _sha256(args.source_report),
        "pilot_report": _sha256(args.pilot_report),
        "replay_audit": _sha256(args.replay_audit),
    }
    expected_hashes = {
        "manifest": EXPECTED_MANIFEST_SHA256,
        "frozen_model": EXPECTED_MODEL_SHA256,
        "source_report": EXPECTED_SOURCE_REPORT_SHA256,
        "pilot_report": EXPECTED_PILOT_REPORT_SHA256,
        "replay_audit": EXPECTED_REPLAY_AUDIT_SHA256,
    }
    if actual_hashes != expected_hashes:
        raise ValueError(f"formal input hash mismatch: {actual_hashes}")
    pilot_report = _read_object(args.pilot_report)
    replay_audit = _read_object(args.replay_audit)
    pilot_acceptance = pilot_report.get("compatibility_acceptance")
    if not isinstance(pilot_acceptance, Mapping) or pilot_acceptance.get(
        "passed"
    ) is not True:
        raise ValueError("bound compatibility pilot did not pass")
    if replay_audit.get("passed") is not True:
        raise ValueError("bound prompt-isolation replay audit did not pass")

    records = read_jsonl(args.rollouts)
    manifest = _read_manifest(args.manifest)
    provenance = _read_object(args.rollouts.with_suffix(".provenance.json"))
    input_validation = _validate_inputs(
        records,
        manifest,
        provenance,
        rollouts=args.rollouts,
        expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        expected_rollout_code_revision=EXPECTED_ROLLOUT_CODE_REVISION,
        expected_states=EXPECTED_STATES,
    )
    model = _read_object(args.frozen_model)
    source_report = _read_object(args.source_report)
    source_evaluation = source_report.get("evaluation")
    if not isinstance(source_evaluation, Mapping):
        raise ValueError("source report has no evaluation object")
    source_entropy_threshold = float(source_evaluation["source_entropy_threshold"])
    strata = {state_id: str(row["stratum"]) for state_id, row in manifest.items()}

    output_diagnostics = _output_compatibility(
        records,
        manifest,
        expected_states=EXPECTED_STATES,
    )
    spec_records = _rescore_spec(records, manifest)
    spec_canonical_records = _rescore_spec(
        records,
        manifest,
        canonicalize_constrained=True,
    )
    released = _evaluate(
        records,
        model=model,
        source_entropy_threshold=source_entropy_threshold,
        strata=strata,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    paper_spec = _evaluate(
        spec_records,
        model=model,
        source_entropy_threshold=source_entropy_threshold,
        strata=strata,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    paper_spec_canonical = _evaluate(
        spec_canonical_records,
        model=model,
        source_entropy_threshold=source_entropy_threshold,
        strata=strata,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    primary_criterion = _primary_criterion(released)
    report = {
        "scientific_status": (
            "untouched ChartQAPro formal evaluation; released-code scorer primary"
        ),
        "input_hashes": actual_hashes,
        "input_validation": input_validation,
        "output_diagnostics": output_diagnostics,
        "primary_confirmation_criterion": primary_criterion,
        "released": released,
        "paper_spec": paper_spec,
        "paper_spec_canonical": paper_spec_canonical,
    }
    report_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    _write_json(report_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_build_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": _sha256(report_path),
                "markdown": str(markdown_path),
                "primary_confirmation_passed": primary_criterion["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
