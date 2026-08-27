from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .dataset import read_jsonl, split_by_state, write_jsonl
from .metrics import diagnostic_to_dict, entropy_diagnostic, evaluate_policy
from .model import LinearValueModel
from .policies import (
    AnswerNowPolicy,
    EntropySearchPolicy,
    FixedCenterZoomPolicy,
    LearnedVOIPolicy,
    OracleVOIPolicy,
    RandomZoomPolicy,
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
    model: LinearValueModel | None,
    seed: int,
) -> dict[str, object]:
    policies = [
        AnswerNowPolicy(),
        RandomZoomPolicy(seed=seed),
        FixedCenterZoomPolicy(),
        EntropySearchPolicy(),
    ]
    if model is not None:
        policies.append(LearnedVOIPolicy(model))
    policies.append(OracleVOIPolicy(lambda_cost))
    return {
        "lambda_cost": lambda_cost,
        "entropy_diagnostic": diagnostic_to_dict(entropy_diagnostic(records)),
        "policy_results": [
            evaluate_policy(records, policy, lambda_cost=lambda_cost) for policy in policies
        ],
    }


def command_simulate(args: argparse.Namespace) -> None:
    records = simulate_counterfactual_dataset(
        n_states=args.n_states,
        num_candidates=args.num_candidates,
        seed=args.seed,
    )
    write_jsonl(records, args.output)
    print(json.dumps({"output": str(args.output), "records": len(records)}, indent=2))


def command_diagnose(args: argparse.Namespace) -> None:
    result = diagnostic_to_dict(entropy_diagnostic(read_jsonl(args.data)))
    print(json.dumps(result, indent=2, sort_keys=True))


def command_train(args: argparse.Namespace) -> None:
    records = read_jsonl(args.data)
    model = LinearValueModel.fit(
        records,
        lambda_cost=args.lambda_cost,
        alpha=args.alpha,
    )
    model.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "features": list(model.encoder.names),
                "lambda_cost": model.lambda_cost,
            },
            indent=2,
        )
    )


def command_evaluate(args: argparse.Namespace) -> None:
    records = read_jsonl(args.data)
    model = LinearValueModel.load(args.model) if args.model else None
    lambda_cost = model.lambda_cost if model is not None and args.lambda_cost is None else args.lambda_cost
    if lambda_cost is None:
        lambda_cost = 0.05
    report = _evaluate(records, lambda_cost=lambda_cost, model=model, seed=args.seed)
    if args.output:
        _write_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


def command_demo(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = simulate_counterfactual_dataset(
        n_states=args.n_states,
        num_candidates=args.num_candidates,
        seed=args.seed,
    )
    train, test = split_by_state(records, train_fraction=args.train_fraction, seed=args.seed)
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    model_path = output_dir / "value_model.json"
    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    write_jsonl(train, train_path)
    write_jsonl(test, test_path)
    model = LinearValueModel.fit(train, lambda_cost=args.lambda_cost, alpha=args.alpha)
    model.save(model_path)
    report = _evaluate(test, lambda_cost=args.lambda_cost, model=model, seed=args.seed)
    report["run"] = {
        "synthetic": True,
        "seed": args.seed,
        "n_states": args.n_states,
        "num_candidates": args.num_candidates,
        "train_fraction": args.train_fraction,
        "train_states": len({record.state_id for record in train}),
        "test_states": len({record.state_id for record in test}),
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
    simulate.add_argument("--seed", type=int, default=7)
    simulate.set_defaults(func=command_simulate)

    diagnose = subparsers.add_parser("diagnose", help="measure entropy/success mismatch")
    diagnose.add_argument("--data", type=Path, required=True)
    diagnose.set_defaults(func=command_diagnose)

    train = subparsers.add_parser("train", help="fit the pre-action VOI value model")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--lambda-cost", type=float, default=0.05)
    train.add_argument("--alpha", type=float, default=1.0)
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", help="compare stopping and crop policies")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--model", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--lambda-cost", type=float)
    evaluate.add_argument("--seed", type=int, default=7)
    evaluate.set_defaults(func=command_evaluate)

    demo = subparsers.add_parser("demo", help="run the complete synthetic MVP")
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--n-states", type=int, default=600)
    demo.add_argument("--num-candidates", type=int, default=4)
    demo.add_argument("--train-fraction", type=float, default=0.7)
    demo.add_argument("--lambda-cost", type=float, default=0.05)
    demo.add_argument("--alpha", type=float, default=1.0)
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(func=command_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
