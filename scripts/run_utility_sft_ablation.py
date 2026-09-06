"""Evaluate semantic-dependence controls for one frozen Utility-SFT selector."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from beyond_entropy.predictability_matrix_artifacts import (
    atomic_json_write_exclusive,
    sha256_file,
)
from beyond_entropy.utility_dataset import UtilityInputs, UtilitySample, load_utility_development
try:
    from scripts.render_utility_sft_figures import ranking_summary
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from render_utility_sft_figures import ranking_summary


BENCHMARKS = ("chartqa", "docvqa", "hrbench")
CONDITIONS = ("original", "question_shuffle", "image_shuffle", "region_ablation")


def source_derangement(samples: Sequence[UtilitySample]) -> dict[str, UtilitySample]:
    """Map every source to a deterministic donor from a different source."""

    by_source: dict[str, list[UtilitySample]] = {}
    for sample in samples:
        by_source.setdefault(sample.inputs.state.source_id, []).append(sample)
    sources = sorted(by_source)
    if len(sources) < 2:
        raise ValueError("shuffle controls require at least two sources")
    return {
        source: sorted(
            by_source[sources[(index + 1) % len(sources)]],
            key=lambda sample: sample.inputs.state.state_id,
        )[0]
        for index, source in enumerate(sources)
    }


def controlled_inputs(
    sample: UtilitySample, donor: UtilitySample, condition: str
) -> tuple[UtilityInputs, bool]:
    state = sample.inputs.state
    donor_state = donor.inputs.state
    if donor_state.source_id == state.source_id:
        raise ValueError("shuffle donor must come from a different source")
    if condition == "original":
        return sample.inputs, False
    if condition == "question_shuffle":
        return UtilityInputs(
            replace(
                state,
                question=donor_state.question,
                model_prompt=donor_state.model_prompt,
            ),
            sample.inputs.action_space,
        ), False
    if condition == "image_shuffle":
        return UtilityInputs(
            replace(
                state,
                image_id=donor_state.image_id,
                image_path=donor_state.image_path,
            ),
            sample.inputs.action_space,
        ), False
    if condition == "region_ablation":
        return sample.inputs, True
    raise ValueError(f"unknown control: {condition}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def main() -> None:
    import torch

    from beyond_entropy.utility_qwen import QwenSpatialUtility

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--selector-sha256", required=True)
    parser.add_argument("--validation-freeze", required=True)
    parser.add_argument("--validation-freeze-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("semantic controls require a GPU Slurm job")
    for path, expected, name in (
        (args.selector, args.selector_sha256, "selector"),
        (args.validation_freeze, args.validation_freeze_sha256, "validation freeze"),
    ):
        if len(expected) != 64 or sha256_file(path) != expected:
            raise ValueError(f"{name} hash mismatch")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    validation = json.loads(Path(args.validation_freeze).read_text(encoding="utf-8"))
    valid_report_contracts = {
        "utility_sft_development_pilot_v1": "three_domain_development_pilot",
        "utility_sft_development_correction_v1": "three_domain_development_correction",
    }
    if (
        config.get("method") != "utility"
        or config.get("scope") not in valid_report_contracts.values()
        or config.get("test_authorized") is not False
        or report.get("schema") not in valid_report_contracts
        or config.get("scope") != valid_report_contracts[report["schema"]]
        or report.get("method") != "utility"
        or report.get("test_accessed") is not False
        or report.get("selector_sha256") != args.selector_sha256
        or report.get("provenance", {}).get("config_sha256") != sha256_file(args.config)
        or validation.get("schema") != "utility_sft_validation_freeze_v1"
        or validation.get("test_data_present") is not False
    ):
        raise ValueError("ablation inputs violate the frozen development contract")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True,
        dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True,
        min_pixels=config["min_pixels"], max_pixels=config["max_pixels"],
    )
    model = QwenSpatialUtility(
        backbone, processor, temperature=config["temperature"],
        head_dim=config["head_dim"], min_pixels=config["min_pixels"],
        max_pixels=config["max_pixels"],
    )
    saved = torch.load(args.selector, map_location="cpu", weights_only=False)
    if saved.get("provenance") != report["provenance"]:
        raise ValueError("selector provenance differs from completed report")
    parameters = _mapping(saved.get("parameters"), "selector parameters")
    expected_parameters = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(parameters) != expected_parameters:
        raise ValueError("selector parameter coverage differs from trainable model")
    incompatible = model.load_state_dict(parameters, strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) & expected_parameters:
        raise ValueError("selector state failed strict trainable-parameter restoration")
    model.eval()

    results: dict[str, Any] = {}
    raw_predictions: dict[str, Any] = {}
    with torch.no_grad():
        for benchmark in BENCHMARKS:
            entry = validation["inventory"][benchmark]
            if sha256_file(entry["dataset"]) != entry["dataset_sha256"]:
                raise ValueError(f"{benchmark} validation dataset changed")
            samples = load_utility_development(entry["dataset"], role="validation")
            donors = source_derangement(samples)
            condition_scores: dict[str, list[list[float]]] = {
                condition: [] for condition in CONDITIONS
            }
            condition_measurements: dict[str, list[Mapping[str, int]]] = {
                condition: [] for condition in CONDITIONS
            }
            for condition in CONDITIONS:
                for sample in samples:
                    donor = donors[sample.inputs.state.source_id]
                    inputs, region_ablation = controlled_inputs(sample, donor, condition)
                    output = model(inputs, region_ablation=region_ablation)
                    condition_scores[condition].append(
                        output["predicted_gain"][0].float().cpu().tolist()
                    )
                    condition_measurements[condition].append(dict(model.last_measurement))
            true = [list(sample.gains) for sample in samples]
            summaries = {
                condition: ranking_summary(true, condition_scores[condition])
                for condition in CONDITIONS
            }
            original = summaries["original"]
            effects = {
                condition: {
                    "pairwise_accuracy_drop": (
                        None
                        if original["pairwise_ranking_accuracy"] is None
                        or summaries[condition]["pairwise_ranking_accuracy"] is None
                        else original["pairwise_ranking_accuracy"]
                        - summaries[condition]["pairwise_ranking_accuracy"]
                    ),
                    "top1_regret_increase": (
                        summaries[condition]["mean_top1_regret"]
                        - original["mean_top1_regret"]
                    ),
                }
                for condition in CONDITIONS[1:]
            }
            if any(
                measurement.get("vision_encoder_calls") != 1
                or measurement.get("candidate_crop_executions") != 0
                for rows in condition_measurements.values() for measurement in rows
            ):
                raise RuntimeError("ablation selector violated single-image inference")
            results[benchmark] = {
                "ranking": summaries,
                "ablation_minus_original": effects,
                "states": len(samples),
                "sources": len({sample.inputs.state.source_id for sample in samples}),
            }
            raw_predictions[benchmark] = {
                condition: {
                    sample.inputs.state.state_id: scores
                    for sample, scores in zip(samples, condition_scores[condition], strict=True)
                }
                for condition in CONDITIONS
            }
    payload = {
        "schema": "utility_sft_semantic_ablation_v1",
        "formal_claim_eligible": False,
        "test_accessed": False,
        "job_id": os.environ["SLURM_JOB_ID"],
        "selector": str(Path(args.selector).resolve()),
        "selector_sha256": args.selector_sha256,
        "report": str(Path(args.report).resolve()),
        "report_sha256": sha256_file(args.report),
        "validation_freeze": str(Path(args.validation_freeze).resolve()),
        "validation_freeze_sha256": args.validation_freeze_sha256,
        "shuffle": "deterministic cyclic derangement of sorted source IDs",
        "results": results,
        "predictions": raw_predictions,
    }
    atomic_json_write_exclusive(args.output, payload)
    print(json.dumps({
        "output": str(Path(args.output).resolve()), "sha256": sha256_file(args.output),
        "results": results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
