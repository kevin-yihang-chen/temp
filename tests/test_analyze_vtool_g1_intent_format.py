from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_analyzer() -> ModuleType:
    path = ROOT / "scripts" / "analyze_vtool_g1_intent_format.py"
    spec = importlib.util.spec_from_file_location(
        "analyze_vtool_g1_intent_format", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_inspection_separates_format_and_argument_failures() -> None:
    analyzer = _load_analyzer()
    labels = {"x": ("North America",), "y": ()}

    canonical_bare = analyzer.inspect_output(
        'focus_on_x_values_with_draw(image_1, ["North America"], columns_bbox)',
        prompt_labels=labels,
    )
    assert canonical_bare["raw_bare_tool_intent"] is True
    assert canonical_bare["python_ast_parseable"] is True
    assert canonical_bare["single_allowed_focus_call"] is True
    assert canonical_bare["runtime_argument_schema_valid"] is True
    assert canonical_bare["axis_label_contract_valid"] is True
    assert canonical_bare["fence_only_repair_executable"] is True
    assert canonical_bare["prompt_contract_valid"] is False

    malformed_arguments = analyzer.inspect_output(
        'focus_on_x_values_with_draw("North America")', prompt_labels=labels
    )
    assert malformed_arguments["python_ast_parseable"] is True
    assert malformed_arguments["single_allowed_focus_call"] is True
    assert malformed_arguments["runtime_argument_schema_valid"] is False
    assert malformed_arguments["fence_only_repair_executable"] is False
    assert "runtime_argument_schema_invalid" in malformed_arguments["failure_reasons"]

    mixed_answer = analyzer.inspect_output(
        'focus_on_x_values_with_draw(image_1, ["North America"], columns_bbox)\n'
        "FINAL ANSWER: 2.5 TERMINATE",
        prompt_labels=labels,
    )
    assert mixed_answer["raw_bare_tool_intent"] is True
    assert mixed_answer["python_ast_parseable"] is False
    assert mixed_answer["fence_only_repair_executable"] is False

    exact_prompt_contract = analyzer.inspect_output(
        "```python\n"
        'display(focus_on_x_values_with_draw(image_1, ["North America"], '
        "columns_bbox))\n"
        "```",
        prompt_labels=labels,
    )
    assert exact_prompt_contract["raw_bare_tool_intent"] is False
    assert exact_prompt_contract["display_wrapper_present"] is True
    assert exact_prompt_contract["prompt_contract_valid"] is True


def test_job_206205_posthoc_intent_format_evidence() -> None:
    analyzer = _load_analyzer()
    run_dir = (
        ROOT
        / "artifacts"
        / "docvqa-train-factorized-v2"
        / "g1-runs"
        / "paired-signed-v1"
        / "job-206205"
    )
    runtime_vtool = Path(
        "/userhome/cs3/yihangc/Documents/runtime/"
        "vtool-action-credit-g1/recipe/vtool/vtool.py"
    )
    if not runtime_vtool.is_file():
        pytest.skip("pinned Job 206205 VTool runtime is unavailable")
    report, exit_code = analyzer.analyze_intent_format(
        rollout_dir=run_dir / "rollouts",
        official_analysis_path=run_dir / "rollout-analysis.json",
        runtime_vtool_path=runtime_vtool,
    )

    assert exit_code == 0
    assert (
        report["decision"]
        == "g1_zero_parser_valid_support_with_malformed_bare_tool_intent"
    )
    assert report["official_g1_decision"] == "paired_signed_g1_stop_rule_triggered"
    assert report["official_g1_decision_changed"] is False
    assert all(report["checks"].values())
    assert report["failures"] == []
    assert report["summary"]["rows"] == 64
    assert report["summary"]["counts"]["parser_valid_tool_call"] == 0
    assert report["summary"]["counts"]["raw_direct_final_answer"] == 48
    assert report["summary"]["counts"]["raw_bare_tool_intent"] == 16
    assert report["summary"]["counts"]["runtime_argument_schema_valid"] == 0
    assert report["summary"]["counts"]["fence_only_repair_executable"] == 0
    assert report["per_step"]["1"]["counts"]["raw_bare_tool_intent"] == 12
    assert report["per_step"]["2"]["counts"]["raw_bare_tool_intent"] == 4
    assert report["runtime_contract"]["typed_bbox_name_by_axis"] == {
        "x": "columns_bbox",
        "y": "rows_bbox",
    }
    assert (
        report["runtime_contract"]["prompt_declared_axis_bbox_variables_exposed"]
        is False
    )
    assert report["protected_split_contents_accessed"] is False
    assert report["model_weights_loaded"] is False
