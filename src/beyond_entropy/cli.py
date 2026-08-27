from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Sequence

from .dataset import group_by_decision, read_jsonl, split_by_group, write_jsonl
from .metrics import (
    bootstrap_entropy_diagnostic,
    bootstrap_policy_evaluation,
    diagnostic_to_dict,
    entropy_diagnostic,
    evaluate_policy,
)
from .model import LinearGainModel
from .policies import (
    AnswerNowPolicy,
    EntropyReductionThresholdPolicy,
    EntropyFixedZoomPolicy,
    EntropyExpectedRandomZoomPolicy,
    EntropyRandomZoomPolicy,
    EntropySearchPolicy,
    EntropyThresholdPolicy,
    FixedCenterZoomPolicy,
    ExpectedRandomZoomPolicy,
    LearnedVOIPolicy,
    OracleVOIPolicy,
    Policy,
    RandomZoomPolicy,
    tune_entropy_thresholds,
    tune_entropy_single_crop_thresholds,
    tune_entropy_expected_random_threshold,
)
from .report import build_markdown_report
from .schema import ActionRecord
from .simulate import simulate_counterfactual_dataset


def _write_json(value: object, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evaluate(
    records: Sequence[ActionRecord],
    *,
    lambda_cost: float,
    model: LinearGainModel | None,
    seed: int,
    entropy_threshold: float | None = None,
    entropy_reduction_threshold: float | None = None,
    entropy_random_threshold: float | None = None,
    entropy_fixed_threshold: float | None = None,
    entropy_expected_random_threshold: float | None = None,
    bootstrap_resamples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    if lambda_cost < 0.0:
        raise ValueError("lambda_cost must be non-negative")
    policies: list[Policy] = [
        AnswerNowPolicy(),
        RandomZoomPolicy(seed=seed),
        FixedCenterZoomPolicy(),
        ExpectedRandomZoomPolicy(),
        EntropySearchPolicy(),
    ]
    if entropy_threshold is not None:
        policies.append(EntropyThresholdPolicy(entropy_threshold))
    if entropy_reduction_threshold is not None:
        policies.append(EntropyReductionThresholdPolicy(entropy_reduction_threshold))
    if entropy_random_threshold is not None:
        policies.append(EntropyRandomZoomPolicy(entropy_random_threshold, seed=seed))
    if entropy_fixed_threshold is not None:
        policies.append(EntropyFixedZoomPolicy(entropy_fixed_threshold))
    if entropy_expected_random_threshold is not None:
        policies.append(
            EntropyExpectedRandomZoomPolicy(entropy_expected_random_threshold)
        )
    if model is not None:
        policies.append(LearnedVOIPolicy(model, lambda_cost=lambda_cost))
    policies.append(OracleVOIPolicy(lambda_cost))
    if bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples must be non-negative")
    policy_results: list[dict[str, object]] = []
    for policy_index, policy in enumerate(policies):
        result: dict[str, object] = dict(
            evaluate_policy(records, policy, lambda_cost=lambda_cost)
        )
        if bootstrap_resamples:
            result["bootstrap"] = bootstrap_policy_evaluation(
                records,
                policy,
                lambda_cost=lambda_cost,
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed + policy_index,
            )
        policy_results.append(result)
    return {
        "lambda_cost": lambda_cost,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "baseline_thresholds": {
            "entropy": entropy_threshold,
            "entropy_reduction": entropy_reduction_threshold,
            "entropy_random_zoom": entropy_random_threshold,
            "entropy_fixed_zoom": entropy_fixed_threshold,
            "entropy_uniform_random_expectation": entropy_expected_random_threshold,
        },
        "entropy_diagnostic": diagnostic_to_dict(entropy_diagnostic(records)),
        "policy_results": policy_results,
    }


def command_simulate(args: argparse.Namespace) -> None:
    records = simulate_counterfactual_dataset(
        n_states=args.n_states,
        num_candidates=args.num_candidates,
        questions_per_image=args.questions_per_image,
        seed=args.seed,
    )
    write_jsonl(records, args.output)
    print(json.dumps({"output": str(args.output), "records": len(records)}, indent=2))


def command_diagnose(args: argparse.Namespace) -> None:
    result = diagnostic_to_dict(entropy_diagnostic(read_jsonl(args.data)))
    print(json.dumps(result, indent=2, sort_keys=True))


def command_train(args: argparse.Namespace) -> None:
    records = read_jsonl(args.data)
    model = LinearGainModel.fit(records, alpha=args.alpha)
    model.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "features": list(model.encoder.names),
                "target": "delta_success",
            },
            indent=2,
        )
    )


def command_evaluate(args: argparse.Namespace) -> None:
    records = read_jsonl(args.data)
    model = LinearGainModel.load(args.model) if args.model else None
    report = _evaluate(
        records,
        lambda_cost=args.lambda_cost,
        model=model,
        seed=args.seed,
        entropy_threshold=args.entropy_threshold,
        entropy_reduction_threshold=args.entropy_reduction_threshold,
    )
    if args.output:
        _write_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


def command_demo(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = simulate_counterfactual_dataset(
        n_states=args.n_states,
        num_candidates=args.num_candidates,
        questions_per_image=args.questions_per_image,
        seed=args.seed,
    )
    train, test = split_by_group(
        records,
        group=args.split_group,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    model_path = output_dir / "gain_model.json"
    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    write_jsonl(train, train_path)
    write_jsonl(test, test_path)
    model = LinearGainModel.fit(train, alpha=args.alpha)
    model.save(model_path)
    entropy_threshold, reduction_threshold = tune_entropy_thresholds(
        train,
        lambda_cost=args.lambda_cost,
    )
    entropy_random_threshold, entropy_fixed_threshold = (
        tune_entropy_single_crop_thresholds(
            train,
            lambda_cost=args.lambda_cost,
            seed=args.seed,
        )
    )
    entropy_expected_random_threshold = tune_entropy_expected_random_threshold(
        train,
        lambda_cost=args.lambda_cost,
    )
    report = _evaluate(
        test,
        lambda_cost=args.lambda_cost,
        model=model,
        seed=args.seed,
        entropy_threshold=entropy_threshold,
        entropy_reduction_threshold=reduction_threshold,
        entropy_random_threshold=entropy_random_threshold,
        entropy_fixed_threshold=entropy_fixed_threshold,
        entropy_expected_random_threshold=entropy_expected_random_threshold,
    )
    report["run"] = {
        "synthetic": True,
        "seed": args.seed,
        "n_states": args.n_states,
        "num_candidates": args.num_candidates,
        "questions_per_image": args.questions_per_image,
        "train_fraction": args.train_fraction,
        "split_group": args.split_group,
        "train_decisions": len(group_by_decision(train)),
        "test_decisions": len(group_by_decision(test)),
    }
    _write_json(report, report_path)
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "train_data": str(train_path),
                "test_data": str(test_path),
                "model": str(model_path),
                "report": str(report_path),
                "markdown_report": str(markdown_path),
            },
            indent=2,
        )
    )


def command_fit_baseline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.data)
    train, test = split_by_group(
        records,
        group=args.split_group,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    model_path = output_dir / "gain_model.json"
    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    write_jsonl(train, train_path)
    write_jsonl(test, test_path)
    model = LinearGainModel.fit(train, alpha=args.alpha)
    model.save(model_path)
    entropy_threshold, reduction_threshold = tune_entropy_thresholds(
        train,
        lambda_cost=args.lambda_cost,
    )
    entropy_random_threshold, entropy_fixed_threshold = (
        tune_entropy_single_crop_thresholds(
            train,
            lambda_cost=args.lambda_cost,
            seed=args.seed,
        )
    )
    entropy_expected_random_threshold = tune_entropy_expected_random_threshold(
        train,
        lambda_cost=args.lambda_cost,
    )
    report = _evaluate(
        test,
        lambda_cost=args.lambda_cost,
        model=model,
        seed=args.seed,
        entropy_threshold=entropy_threshold,
        entropy_reduction_threshold=reduction_threshold,
        entropy_random_threshold=entropy_random_threshold,
        entropy_fixed_threshold=entropy_fixed_threshold,
        entropy_expected_random_threshold=entropy_expected_random_threshold,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    report["run"] = {
        "synthetic": False,
        "source_data": str(args.data.resolve()),
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "split_group": args.split_group,
        "train_decisions": len(group_by_decision(train)),
        "test_decisions": len(group_by_decision(test)),
        "train_records": len(train),
        "test_records": len(test),
        "model_target": "delta_success",
        "model_features": list(model.encoder.names),
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    _write_json(report, report_path)
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "train_data": str(train_path),
                "test_data": str(test_path),
                "model": str(model_path),
                "report": str(report_path),
                "markdown_report": str(markdown_path),
            },
            indent=2,
        )
    )


def command_collect_qwen(args: argparse.Namespace) -> None:
    from .benchmarks import load_manifest, scorer_by_name
    from .crops import UGGridProposer
    from .qwen_backend import Qwen25VLBackend
    from .rollout import CachedVisualBackend, collect_sibling_rollouts

    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if (
        args.expected_manifest_sha256
        and manifest_sha256 != args.expected_manifest_sha256
    ):
        raise ValueError(
            "manifest SHA-256 mismatch: "
            f"expected {args.expected_manifest_sha256}, got {manifest_sha256}"
        )
    examples = load_manifest(args.manifest, limit=args.limit)
    proposer = UGGridProposer(
        candidate_count=args.candidate_count,
        visual_crop_ratio=args.visual_crop_ratio,
        visual_cost=args.visual_cost,
    )
    scorer = scorer_by_name(args.scorer)
    records: list[ActionRecord] = []
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(
                f"output already exists: {args.output}; pass --resume to continue"
            )
        if args.output.stat().st_size:
            records = read_jsonl(args.output)
            default_system_prompt = "You are a helpful assistant."
            for record in records:
                backend_names = ["baseline_backend"]
                if record.action_type == "ZOOM":
                    backend_names.append("action_backend")
                for backend_name in backend_names:
                    backend_metadata = record.metadata.get(backend_name)
                    if not isinstance(backend_metadata, dict):
                        raise ValueError(
                            f"checkpoint record is missing {backend_name} metadata"
                        )
                    expected_backend_values = {
                        "model_revision": args.model_revision,
                        "max_new_tokens": args.max_new_tokens,
                        "min_pixels": args.min_pixels,
                        "max_pixels": args.max_pixels,
                    }
                    for name, expected in expected_backend_values.items():
                        if backend_metadata.get(name) != expected:
                            raise ValueError(
                                f"checkpoint {backend_name} {name} mismatch: "
                                f"expected {expected!r}, got {backend_metadata.get(name)!r}"
                            )
                    recorded_prompt = backend_metadata.get(
                        "system_prompt",
                        default_system_prompt,
                    )
                    if recorded_prompt != args.system_prompt:
                        raise ValueError(
                            f"checkpoint {backend_name} system_prompt mismatch"
                        )
    initial_record_count = len(records)
    expected_per_state = (args.candidate_count + 1) * len(args.generation_seeds)
    checkpoint_counts: dict[str, int] = {}
    for record in records:
        checkpoint_counts[record.state_id] = checkpoint_counts.get(record.state_id, 0) + 1
    for state_id, count in checkpoint_counts.items():
        if count != expected_per_state:
            raise ValueError(
                f"checkpoint state {state_id!r} has {count} records; "
                f"expected {expected_per_state}"
            )
    manifest_state_ids = {example.state.state_id for example in examples}
    unexpected_states = set(checkpoint_counts) - manifest_state_ids
    if unexpected_states:
        raise ValueError(
            f"checkpoint contains states outside the manifest: {sorted(unexpected_states)}"
        )
    pending = [
        example for example in examples if example.state.state_id not in checkpoint_counts
    ]
    if pending:
        backend = CachedVisualBackend(
            Qwen25VLBackend(
                args.model,
                revision=args.model_revision,
                device_map=args.device_map,
                dtype=args.dtype,
                attention_implementation=args.attention_implementation,
                max_new_tokens=args.max_new_tokens,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                local_files_only=not args.allow_download,
                system_prompt=args.system_prompt,
            )
        )
        for position, example in enumerate(pending, start=1):
            records.extend(
                collect_sibling_rollouts(
                    [example],
                    proposals=proposer,
                    backend=backend,
                    scorer=scorer,
                    generation_seeds=args.generation_seeds,
                )
            )
            write_jsonl(records, args.output)
            print(
                json.dumps(
                    {
                        "checkpoint": str(args.output),
                        "completed_this_run": position,
                        "pending_this_run": len(pending) - position,
                        "total_completed": len(checkpoint_counts) + position,
                        "total_examples": len(examples),
                    }
                ),
                flush=True,
            )
    diagnostic_path = args.output.with_suffix(".diagnostic.json")
    diagnostic = {
        "point_estimate": diagnostic_to_dict(entropy_diagnostic(records)),
        "bootstrap": bootstrap_entropy_diagnostic(
            records,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        ),
    }
    _write_json(diagnostic, diagnostic_path)

    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    provenance_path = args.output.with_suffix(".provenance.json")
    provenance = {
        "scientific_status": "diagnostic; not a benchmark claim",
        "code_revision": os.environ.get("BE_CODE_REVISION"),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "output": str(args.output.resolve()),
        "model": args.model,
        "model_revision": args.model_revision,
        "ug_framework_revision": args.ug_revision,
        "scorer": args.scorer,
        "examples": len(examples),
        "completed_examples": len(examples),
        "resumed_from_records": initial_record_count,
        "candidate_count": args.candidate_count,
        "visual_crop_ratio": args.visual_crop_ratio,
        "visual_cost": args.visual_cost,
        "generation_seeds": list(args.generation_seeds),
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "max_new_tokens": args.max_new_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "device_map": args.device_map,
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "system_prompt": args.system_prompt,
        "local_files_only": not args.allow_download,
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "qwen-vl-utils": package_version("qwen-vl-utils"),
            "Pillow": package_version("Pillow"),
        },
    }
    _write_json(provenance, provenance_path)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "diagnostic": str(diagnostic_path),
                "provenance": str(provenance_path),
                "examples": len(examples),
                "records": len(records),
            },
            indent=2,
        )
    )


def command_extract_qwen_features(args: argparse.Namespace) -> None:
    from .qwen_semantic import extract_qwen_semantic_dataset

    rollouts_sha256 = hashlib.sha256(args.rollouts.read_bytes()).hexdigest()
    if (
        args.expected_rollouts_sha256
        and rollouts_sha256 != args.expected_rollouts_sha256
    ):
        raise ValueError(
            "rollout SHA-256 mismatch: "
            f"expected {args.expected_rollouts_sha256}, got {rollouts_sha256}"
        )
    result = extract_qwen_semantic_dataset(
        rollouts_path=args.rollouts,
        output_path=args.output,
        model_name_or_path=args.model,
        revision=args.model_revision,
        device_map=args.device_map,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        local_files_only=not args.allow_download,
        question_feature_mode=args.question_feature_mode,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decisions": len(result["decisions"]),
                "source_rollouts_sha256": rollouts_sha256,
                "model_revision": args.model_revision,
            },
            indent=2,
        )
    )


def command_fit_semantic(args: argparse.Namespace) -> None:
    from .semantic_training import fit_semantic_gain_experiment

    report = fit_semantic_gain_experiment(
        feature_path=args.features,
        rollouts_path=args.rollouts,
        output_dir=args.output_dir,
        split_group=args.split_group,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        lambdas=args.lambda_costs,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rank_weight=args.rank_weight,
        nonzero_weight=args.nonzero_weight,
        transition_weight=args.transition_weight,
        similarity_cv_folds=args.similarity_cv_folds,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "best_epoch": report["run"]["best_epoch"],
                "epochs_ran": report["run"]["epochs_ran"],
                "test_decisions": report["run"]["test_decisions"],
            },
            indent=2,
        )
    )


def command_initialize_semantic_checkpoint(args: argparse.Namespace) -> None:
    from .qwen_semantic import initialize_semantic_feature_checkpoint

    result = initialize_semantic_feature_checkpoint(
        source_feature_path=args.source_features,
        target_rollouts_path=args.target_rollouts,
        output_path=args.output,
    )
    initialization = result["metadata"]["checkpoint_initialization"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "initialized_decisions": initialization["initialized_decisions"],
                "target_decisions": initialization["target_decisions"],
                "source_features_sha256": initialization["source_features_sha256"],
                "target_rollouts_sha256": result["metadata"][
                    "source_rollouts_sha256"
                ],
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beyond-entropy",
        description="Counterfactual visual value-of-information research utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="generate paired synthetic rollouts")
    simulate.add_argument("--output", type=Path, required=True)
    simulate.add_argument("--n-states", type=int, default=600)
    simulate.add_argument("--num-candidates", type=int, default=4)
    simulate.add_argument("--questions-per-image", type=int, default=2)
    simulate.add_argument("--seed", type=int, default=7)
    simulate.set_defaults(func=command_simulate)

    diagnose = subparsers.add_parser("diagnose", help="measure entropy/success mismatch")
    diagnose.add_argument("--data", type=Path, required=True)
    diagnose.set_defaults(func=command_diagnose)

    train = subparsers.add_parser("train", help="fit the pre-action success-gain model")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--alpha", type=float, default=1.0)
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", help="compare stopping and crop policies")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--model", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--lambda-cost", type=float, default=0.05)
    evaluate.add_argument("--entropy-threshold", type=float)
    evaluate.add_argument("--entropy-reduction-threshold", type=float)
    evaluate.add_argument("--seed", type=int, default=7)
    evaluate.set_defaults(func=command_evaluate)

    demo = subparsers.add_parser("demo", help="run the complete synthetic MVP")
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts/demo-v2"))
    demo.add_argument("--n-states", type=int, default=600)
    demo.add_argument("--num-candidates", type=int, default=4)
    demo.add_argument("--questions-per-image", type=int, default=2)
    demo.add_argument("--train-fraction", type=float, default=0.7)
    demo.add_argument(
        "--split-group",
        choices=("source_id", "image_id", "state_id"),
        default="image_id",
    )
    demo.add_argument("--lambda-cost", type=float, default=0.05)
    demo.add_argument("--alpha", type=float, default=1.0)
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(func=command_demo)

    fit_baseline = subparsers.add_parser(
        "fit-baseline",
        help="fit and evaluate a leakage-safe scalar gain baseline",
    )
    fit_baseline.add_argument("--data", type=Path, required=True)
    fit_baseline.add_argument("--output-dir", type=Path, required=True)
    fit_baseline.add_argument("--train-fraction", type=float, default=0.7)
    fit_baseline.add_argument(
        "--split-group",
        choices=("source_id", "image_id", "state_id"),
        default="image_id",
    )
    fit_baseline.add_argument("--lambda-cost", type=float, default=0.05)
    fit_baseline.add_argument("--alpha", type=float, default=1.0)
    fit_baseline.add_argument("--seed", type=int, default=17)
    fit_baseline.add_argument("--bootstrap-resamples", type=int, default=0)
    fit_baseline.add_argument("--bootstrap-seed", type=int, default=0)
    fit_baseline.set_defaults(func=command_fit_baseline)

    collect_qwen = subparsers.add_parser(
        "collect-qwen",
        help="collect frozen Qwen2.5-VL sibling rollouts from a JSONL manifest",
    )
    collect_qwen.add_argument("--manifest", type=Path, required=True)
    collect_qwen.add_argument("--expected-manifest-sha256")
    collect_qwen.add_argument("--output", type=Path, required=True)
    collect_qwen.add_argument(
        "--resume",
        action="store_true",
        help="resume from a checkpoint containing only complete states",
    )
    collect_qwen.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    collect_qwen.add_argument("--model-revision", default="main")
    collect_qwen.add_argument(
        "--ug-revision",
        default="13050ee49865e4330519108f42d1ccfccff1aee1",
    )
    collect_qwen.add_argument("--scorer", choices=("vstar", "chartqa"), required=True)
    collect_qwen.add_argument("--candidate-count", type=int, default=4)
    collect_qwen.add_argument("--visual-crop-ratio", type=float, default=2.0)
    collect_qwen.add_argument("--visual-cost", type=float, default=1.0)
    collect_qwen.add_argument("--generation-seeds", type=int, nargs="+", default=[0])
    collect_qwen.add_argument("--bootstrap-resamples", type=int, default=2000)
    collect_qwen.add_argument("--bootstrap-seed", type=int, default=0)
    collect_qwen.add_argument("--limit", type=int)
    collect_qwen.add_argument("--max-new-tokens", type=int, default=16)
    collect_qwen.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    collect_qwen.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    collect_qwen.add_argument("--device-map", default="cuda:0")
    collect_qwen.add_argument("--dtype", default="bfloat16")
    collect_qwen.add_argument("--attention-implementation", default="sdpa")
    collect_qwen.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
    )
    collect_qwen.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face network access (offline cache is the default)",
    )
    collect_qwen.set_defaults(func=command_collect_qwen)

    extract_qwen_features = subparsers.add_parser(
        "extract-qwen-features",
        help="extract one-pass frozen Qwen semantic ROI features",
    )
    extract_qwen_features.add_argument("--rollouts", type=Path, required=True)
    extract_qwen_features.add_argument("--expected-rollouts-sha256")
    extract_qwen_features.add_argument("--output", type=Path, required=True)
    extract_qwen_features.add_argument(
        "--resume",
        action="store_true",
        help="resume a feature checkpoint containing complete decisions",
    )
    extract_qwen_features.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    extract_qwen_features.add_argument("--model-revision", default="main")
    extract_qwen_features.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    extract_qwen_features.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    extract_qwen_features.add_argument("--device-map", default="cuda:0")
    extract_qwen_features.add_argument("--dtype", default="bfloat16")
    extract_qwen_features.add_argument("--attention-implementation", default="sdpa")
    extract_qwen_features.add_argument(
        "--question-feature-mode",
        choices=("input_mean", "contextual_text_mean"),
        default="input_mean",
    )
    extract_qwen_features.add_argument("--allow-download", action="store_true")
    extract_qwen_features.set_defaults(func=command_extract_qwen_features)

    fit_semantic = subparsers.add_parser(
        "fit-semantic",
        help="fit and evaluate the frozen-Qwen semantic ROI gain head",
    )
    fit_semantic.add_argument("--features", type=Path, required=True)
    fit_semantic.add_argument("--rollouts", type=Path, required=True)
    fit_semantic.add_argument("--output-dir", type=Path, required=True)
    fit_semantic.add_argument("--train-fraction", type=float, default=0.7)
    fit_semantic.add_argument("--validation-fraction", type=float, default=0.2)
    fit_semantic.add_argument(
        "--split-group",
        choices=("source_id", "image_id", "state_id"),
        default="image_id",
    )
    fit_semantic.add_argument(
        "--lambda-costs",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05, 0.1, 0.2],
    )
    fit_semantic.add_argument("--hidden-dim", type=int, default=64)
    fit_semantic.add_argument("--dropout", type=float, default=0.2)
    fit_semantic.add_argument("--learning-rate", type=float, default=1e-3)
    fit_semantic.add_argument("--weight-decay", type=float, default=1e-3)
    fit_semantic.add_argument("--rank-weight", type=float, default=1.0)
    fit_semantic.add_argument("--nonzero-weight", type=float, default=8.0)
    fit_semantic.add_argument("--transition-weight", type=float, default=8.0)
    fit_semantic.add_argument("--similarity-cv-folds", type=int, default=5)
    fit_semantic.add_argument("--bootstrap-resamples", type=int, default=2000)
    fit_semantic.add_argument("--bootstrap-seed", type=int, default=0)
    fit_semantic.add_argument("--max-epochs", type=int, default=500)
    fit_semantic.add_argument("--patience", type=int, default=50)
    fit_semantic.add_argument("--seed", type=int, default=17)
    fit_semantic.add_argument("--device", default="cuda")
    fit_semantic.set_defaults(func=command_fit_semantic)

    initialize_semantic = subparsers.add_parser(
        "initialize-semantic-checkpoint",
        help="validate and rebase partial semantic features onto larger rollouts",
    )
    initialize_semantic.add_argument("--source-features", type=Path, required=True)
    initialize_semantic.add_argument("--target-rollouts", type=Path, required=True)
    initialize_semantic.add_argument("--output", type=Path, required=True)
    initialize_semantic.set_defaults(func=command_initialize_semantic_checkpoint)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
