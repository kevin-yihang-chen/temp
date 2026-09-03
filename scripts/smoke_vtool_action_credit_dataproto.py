#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import torch

from beyond_entropy.vtool_action_credit import (
    ACTION_CREDIT_KEY,
    ACTION_TOKEN_COUNT_KEY,
    ANSWER_TOKEN_COUNT_KEY,
    OBSERVATION_TOKEN_COUNT_KEY,
    PAIR_VALID_KEY,
    TRAJECTORY_ID_KEY,
    inject_token_local_action_credit,
)
from verl.protocol import DataProto


def main() -> None:
    response_mask = torch.tensor(
        [
            [1, 1, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ]
    )
    advantages = torch.tensor(
        [
            [0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0],
            [-0.2, -0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
            [-0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    data = DataProto.from_dict(
        tensors={
            "advantages": advantages,
            "returns": torch.zeros_like(advantages),
            "response_mask": response_mask,
        },
        non_tensors={
            TRAJECTORY_ID_KEY: np.asarray(
                ["tool-rescue", "direct-a", "tool-harm", "direct-b"],
                dtype=object,
            ),
            ACTION_TOKEN_COUNT_KEY: np.asarray([2, 0, 1, 0]),
            OBSERVATION_TOKEN_COUNT_KEY: np.asarray([2, 0, 1, 0]),
            ANSWER_TOKEN_COUNT_KEY: np.asarray([3, 4, 2, 2]),
            ACTION_CREDIT_KEY: np.asarray([0.95, 0.0, -1.05, 0.0]),
            PAIR_VALID_KEY: np.asarray([True, False, True, False]),
        },
    )

    data, metrics = inject_token_local_action_credit(data, mode="signed")
    chunks = data.chunk(4)
    donor_key = "vtool_action_credit_donor_trajectory_id"
    checks = {
        "all_non_tensor_fields_are_ndarrays": all(
            isinstance(value, np.ndarray) for value in data.non_tensor_batch.values()
        ),
        "four_equal_dp_chunks_created": [len(chunk) for chunk in chunks]
        == [1, 1, 1, 1],
        "signed_donors_preserved": data.non_tensor_batch[donor_key].tolist()
        == ["tool-rescue", None, "tool-harm", None],
        "action_masks_preserved": data.batch["action_mask"].tolist()
        == [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "answer_masks_preserved": data.batch["answer_mask"].tolist()
        == [
            [0, 0, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        "tool_count_is_two": metrics["action_credit/tool_trajectory_count"] == 2.0,
        "cuda_not_initialized": not torch.cuda.is_initialized(),
    }
    report = {
        "schema": "vtool_action_credit_dataproto_chunk_smoke_v1",
        "decision": "vtool_action_credit_dataproto_chunk_passed",
        "chunks": len(chunks),
        "checks": checks,
        "protected_split_contents_accessed": False,
        "model_weights_loaded": False,
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(report, sort_keys=True))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
