from __future__ import annotations

import json
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from beyond_entropy.refocus_chart_audit import sha256_file
from beyond_entropy.refocus_typed_action import RefocusTypedAction
from beyond_entropy.refocus_typed_action_runtime import (
    execute_renderer_owned_action,
    load_refocus_runtime,
)
from scripts.run_refocus_typed_action_b0_generation import (
    _scientific_decision,
    _validate_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "refocus_typed_action_b0_generation_v1.json"


def test_b0_generation_protocol_is_zero_checkpoint_and_fully_frozen() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _validate_protocol(config)

    assert config["study_role"] == "baseline_correctness_only"
    assert config["uses_reward_target"] is False
    assert config["data"]["protected_split_contents_accessed"] is False
    assert config["sampling"]["generation_count"] == 16
    assert len(config["sampling"]["seeds"]) == 16
    assert len(set(config["sampling"]["seeds"])) == 16
    assert config["analysis"]["minimum_tool_intents"] == 2
    assert config["analysis"]["minimum_conditional_execution_rate"] == 0.8
    assert config["analysis"]["nested_metrics"] == [
        "tool_intent",
        "complete_python_fence",
        "python_syntax_valid",
        "argument_contract_valid",
        "parser_valid",
        "execution_success",
    ]
    assert config["resources"]["gpu_count"] == 1
    assert config["resources"]["gpu_type"] == "H800"
    assert config["resources"]["optimizer_steps"] == 0
    assert config["resources"]["checkpoints_written"] == 0
    assert config["resources"]["notification_email"] == ("yihangc@connect.hku.hk")
    assert config["resources"]["slurm_mail_type"] == "ALL"
    for key in (
        "dataset",
        "converter_report",
        "processor_executor_report",
    ):
        assert (ROOT / config["data"][key]).is_file()
    assert sha256_file(ROOT / config["data"]["dataset"]) == (
        config["data"]["dataset_sha256"]
    )
    assert sha256_file(ROOT / config["data"]["converter_report"]) == (
        config["data"]["converter_report_sha256"]
    )
    assert sha256_file(ROOT / config["data"]["processor_executor_report"]) == (
        config["data"]["processor_executor_report_sha256"]
    )


def test_b0_generation_decision_rule_is_frozen_before_outputs() -> None:
    def decide(intent: int, parser_valid: int, execution: int) -> str:
        return _scientific_decision(
            intent_count=intent,
            parser_valid_count=parser_valid,
            execution_count=execution,
            minimum_intents=2,
            minimum_conditional_execution_rate=0.8,
        )

    assert decide(0, 0, 0) == "typed_action_b0_insufficient_tool_intent_support"
    assert decide(2, 0, 0) == "typed_action_b0_malformed_tool_intent"
    assert decide(2, 2, 1) == "typed_action_b0_malformed_tool_intent"
    assert decide(2, 2, 2) == "typed_action_b0_format_gate_passed"


def test_b0_slurm_contract_binds_zero_checkpoint_job_and_all_mail_events() -> None:
    submit_path = ROOT / "scripts" / "submit_refocus_typed_action_b0_generation_h800.sh"
    worker_path = ROOT / "scripts" / "slurm_refocus_typed_action_b0_generation_h800.sh"
    runner_path = ROOT / "scripts" / "run_refocus_typed_action_b0_generation.py"
    subprocess.run(["bash", "-n", str(submit_path)], check=True)
    subprocess.run(["bash", "-n", str(worker_path)], check=True)
    submit = submit_path.read_text(encoding="utf-8")
    worker = worker_path.read_text(encoding="utf-8")
    runner = runner_path.read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:h800:1" in worker
    assert "#SBATCH --mail-user=yihangc@connect.hku.hk" in worker
    assert "#SBATCH --mail-type=ALL" in worker
    assert "--time=00:30:00" in worker
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN" in worker
    assert "resources.checkpoints_written == 0" in worker
    assert ".checkpoints_written == 0" in worker
    assert "wrote unexpected files or checkpoint directories" in worker
    assert "sbatch --test-only --export=NONE" in submit
    assert "converter_report_sha256" in submit
    assert "processor_report_sha256" in submit
    assert "raw_model_text_executed" in runner
    assert "execute_renderer_owned_action" in runner
    assert 'optimizer_steps": 0' in runner
    assert 'checkpoints_written": 0' in runner


def test_generation_runtime_executes_only_a_renderer_owned_action() -> None:
    runtime_path = Path(
        "/userhome/cs3/yihangc/Documents/runtime/"
        "vtool-action-credit-g1/recipe/vtool/refocus_tools.py"
    )
    if not runtime_path.is_file():
        pytest.skip("pinned VTool runtime is unavailable")
    runtime = load_refocus_runtime(runtime_path)
    metadata = {
        "source": "chartqa_v_bar",
        "x_values": ["A"],
        "y_values": [],
        "x_values_bbox": {"A": {"x1": 2, "y1": 2, "x2": 10, "y2": 10}},
        "y_values_bbox": {},
    }
    response, output, changed = execute_renderer_owned_action(
        runtime=runtime,
        image=Image.new("RGB", (16, 16), color="white"),
        metadata=metadata,
        action=RefocusTypedAction(axis="x", mode="draw", labels=("A",)),
    )
    assert response.startswith("```python\ndisplay(")
    assert isinstance(output, Image.Image)
    assert changed is True
