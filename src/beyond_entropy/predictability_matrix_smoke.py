from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .predictability_audit import AUDIT_BENCHMARKS, BinaryToolOutcome, PreActionInputs
from .predictability_matrix import BenchmarkAuditData, run_predictability_matrix
from .predictability_modeling import AuditExample
from .schema import ActionRecord, BBox


def _synthetic_role(
    *, benchmark_index: int, role_index: int, count: int
) -> tuple[list[AuditExample], list[ActionRecord]]:
    examples: list[AuditExample] = []
    records: list[ActionRecord] = []
    for index in range(count):
        value = ((index * 7 + role_index) % 10) / 9.0
        y0 = 1.0 if index % 5 == 0 else 0.0
        if y0 == 0.0:
            y_tool = 1.0 if value > 0.55 else 0.0
        else:
            y_tool = 0.0 if value < 0.15 else 1.0
        prefix = f"b{benchmark_index}-r{role_index}-i{index}"
        inputs = PreActionInputs(
            state_id=f"state-{prefix}",
            image_id=f"image-{prefix}",
            source_id=f"source-{benchmark_index}-{role_index}-{index // 2}",
            entropy_before=0.1 + value,
            max_probability=1.0 - 0.5 * value,
            top1_top2_margin=1.0 - value,
            shallow_question_features=(value, value**2, float(index % 3)),
            question_embedding=(value, 1.0 - value, value**2, 1.0),
            global_visual_embedding=(1.0 - value, value, 1.0, value**2),
            pooled_language_state=(value, value**2, 1.0),
            pooled_visual_state=(1.0 - value, value, 1.0),
            fused_multimodal_state=(value, 1.0 - value, value * (1.0 - value)),
        )
        outcome = BinaryToolOutcome(
            state_id=inputs.state_id,
            replicate_id="replicate-000",
            image_id=inputs.image_id,
            source_id=inputs.source_id,
            selected_action_id="ug-grid-00",
            y0=y0,
            y_tool=y_tool,
            tool_cost=4.0,
            tool_calls=4,
        )
        rgb_hash = f"{benchmark_index * 100000 + role_index * 1000 + index + 1:064x}"
        examples.append(AuditExample(inputs, outcome, rgb_hash))
        records.append(
            ActionRecord(
                state_id=inputs.state_id,
                image_id=inputs.image_id,
                source_id=inputs.source_id,
                question="synthetic question",
                original_image=f"{inputs.image_id}.png",
                replicate_id="replicate-000",
                generation_seed=0,
                action_id="answer-now",
                action_type="ANSWER",
                candidate_bbox=None,
                entropy_before=inputs.entropy_before,
                entropy_after=inputs.entropy_before,
                answer_before="baseline",
                answer_after="baseline",
                correct_before=y0,
                correct_after=y0,
                tool_cost=0.0,
                pre_action_features={},
                metadata={},
            )
        )
        for action_index in range(4):
            records.append(
                ActionRecord(
                    state_id=inputs.state_id,
                    image_id=inputs.image_id,
                    source_id=inputs.source_id,
                    question="synthetic question",
                    original_image=f"{inputs.image_id}.png",
                    replicate_id="replicate-000",
                    generation_seed=0,
                    action_id=f"ug-grid-0{action_index}",
                    action_type="ZOOM",
                    candidate_bbox=BBox(
                        0.25 * (action_index % 2),
                        0.25 * (action_index // 2),
                        0.5 + 0.25 * (action_index % 2),
                        0.5 + 0.25 * (action_index // 2),
                    ),
                    entropy_before=inputs.entropy_before,
                    entropy_after=0.1 + 0.1 * action_index,
                    answer_before="baseline",
                    answer_after=f"crop-{action_index}",
                    correct_before=y0,
                    correct_after=(
                        y_tool
                        if action_index == 0
                        else y0 if action_index == 1 else 0.0
                    ),
                    tool_cost=1.0,
                    pre_action_features={},
                    metadata={},
                )
            )
    return examples, records


def build_synthetic_datasets() -> dict[str, BenchmarkAuditData]:
    result = {}
    for benchmark_index, benchmark in enumerate(AUDIT_BENCHMARKS):
        train, _ = _synthetic_role(
            benchmark_index=benchmark_index, role_index=0, count=40
        )
        validation, validation_siblings = _synthetic_role(
            benchmark_index=benchmark_index, role_index=1, count=20
        )
        test, test_siblings = _synthetic_role(
            benchmark_index=benchmark_index, role_index=2, count=20
        )
        result[benchmark] = BenchmarkAuditData(
            train=train,
            validation=validation,
            test=test,
            validation_siblings=validation_siblings,
            test_siblings=test_siblings,
        )
    return result


def _atomic_write(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a non-scientific 36-cell matrix smoke"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=100)
    args = parser.parse_args(argv)
    report = run_predictability_matrix(
        build_synthetic_datasets(),
        lambda_cost=0.05,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=0.95,
        bootstrap_seed=20260903,
        call_rates=(0.0, 0.01, 0.05, 0.1, 0.5, 1.0),
        formal_claim_eligible=False,
    )
    _atomic_write(report, Path(args.output))


if __name__ == "__main__":
    main()
