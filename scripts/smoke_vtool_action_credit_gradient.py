#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from beyond_entropy.vtool_action_credit import (  # noqa: E402
    ACTION_CREDIT_KEY,
    ACTION_TOKEN_COUNT_KEY,
    ANSWER_TOKEN_COUNT_KEY,
    OBSERVATION_TOKEN_COUNT_KEY,
    PAIR_VALID_KEY,
    TRAJECTORY_ID_KEY,
    CreditMode,
    inject_token_local_action_credit,
)


@dataclass
class SmokeData:
    batch: dict[str, Any]
    non_tensor_batch: dict[str, Any]
    meta_info: dict[str, Any] = field(default_factory=dict)


def build_batch(torch: Any) -> SmokeData:
    return SmokeData(
        batch={
            "advantages": torch.tensor(
                [[-0.25, -0.25, 0.0, 0.0, -0.25, -0.25, -0.25, 0.0]],
                dtype=torch.float32,
            ),
            "returns": torch.zeros(1, 8, dtype=torch.float32),
            "response_mask": torch.tensor([[1, 1, 0, 0, 1, 1, 1, 0]], dtype=torch.long),
        },
        non_tensor_batch={
            TRAJECTORY_ID_KEY: ["gradient-smoke-trajectory"],
            ACTION_TOKEN_COUNT_KEY: [2],
            OBSERVATION_TOKEN_COUNT_KEY: [2],
            ANSWER_TOKEN_COUNT_KEY: [3],
            ACTION_CREDIT_KEY: [0.95],
            PAIR_VALID_KEY: [True],
        },
    )


def gradient_for_mode(torch: Any, mode: CreditMode) -> tuple[list[float], list[float]]:
    data, _ = inject_token_local_action_credit(build_batch(torch), mode=mode)
    logits = torch.zeros(1, 8, dtype=torch.float32, requires_grad=True)
    log_probs = torch.nn.functional.logsigmoid(logits)
    loss = -(log_probs * data.batch["advantages"] * data.batch["response_mask"]).sum()
    loss.backward()
    if logits.grad is None:
        raise RuntimeError("autograd did not populate logit gradients")
    return (
        data.batch["advantages"][0].detach().cpu().tolist(),
        logits.grad[0].detach().cpu().tolist(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify signed action credit reaches only intended policy tokens."
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for this gradient smoke") from exc

    signed_advantages, signed_gradient = gradient_for_mode(torch, "signed")
    zero_advantages, zero_gradient = gradient_for_mode(torch, "zero")
    shuffled_data, shuffled_metrics = inject_token_local_action_credit(
        build_batch(torch), mode="shuffled"
    )
    shuffled_advantages = shuffled_data.batch["advantages"][0].detach().cpu().tolist()
    checks = {
        "signed_action_gradient_nonzero": all(
            abs(value) > 0.0 for value in signed_gradient[:2]
        ),
        "zero_control_action_gradient_zero": all(
            value == 0.0 for value in zero_gradient[:2]
        ),
        "observation_gradient_zero": all(
            value == 0.0 for value in signed_gradient[2:4]
        ),
        "signed_answer_gradient_nonzero": all(
            abs(value) > 0.0 for value in signed_gradient[4:7]
        ),
        "padding_gradient_zero": signed_gradient[7] == 0.0,
        "answer_advantage_control_matched": signed_advantages[4:7]
        == zero_advantages[4:7],
        "single_tool_shuffled_action_loss_skipped": shuffled_advantages[:2]
        == [0.0, 0.0],
        "single_tool_shuffled_skip_reported": shuffled_metrics[
            "action_credit/shuffled_batch_skipped"
        ]
        == 1.0,
    }
    report = {
        "schema": "vtool_action_credit_gradient_smoke_v1",
        "torch_version": torch.__version__,
        "signed_advantages": signed_advantages,
        "zero_advantages": zero_advantages,
        "single_tool_shuffled_advantages": shuffled_advantages,
        "single_tool_shuffled_metrics": shuffled_metrics,
        "signed_logit_gradients": signed_gradient,
        "zero_logit_gradients": zero_gradient,
        "checks": checks,
        "decision": (
            "token_local_action_credit_gradient_smoke_passed"
            if all(checks.values())
            else "token_local_action_credit_gradient_smoke_failed"
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
