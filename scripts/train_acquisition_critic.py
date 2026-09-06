#!/usr/bin/env python3
"""Train linear and small-MLP risk/gain critics on frozen sequential features."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import torch

from beyond_entropy.acquisition_critic import (
    AcquisitionCritic,
    examples_from_feature_dataset,
)
from beyond_entropy.sequential_metrics import binary_auroc, risk_metrics


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_development(path: Path, role: str):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    examples = examples_from_feature_dataset(payload)
    metadata = payload["metadata"]
    if metadata.get("dataset_role") != role or role not in {"train", "validation"}:
        raise ValueError("critic trainer accepts only correctly labelled development roles")
    return payload, examples


def matrix(examples, level):
    first = item_vector = examples[0].inputs.feature_vector(level)
    result = torch.empty((len(examples), len(first)), dtype=torch.float32)
    result[0] = torch.tensor(first, dtype=torch.float32)
    for index, item in enumerate(examples[1:], start=1):
        item_vector = item.inputs.feature_vector(level)
        if len(item_vector) != result.shape[1]:
            raise ValueError("critic feature dimensions changed between rows")
        result[index] = torch.tensor(item_vector, dtype=torch.float32)
    return result


def regression_metrics(predictions, targets):
    p = [float(x) for x in predictions]
    y = [float(x) for x in targets]
    pm, ym = sum(p) / len(p), sum(y) / len(y)
    numerator = sum((a - pm) * (b - ym) for a, b in zip(p, y))
    denominator = (
        sum((a - pm) ** 2 for a in p) * sum((b - ym) ** 2 for b in y)
    ) ** 0.5
    return {
        "mse": sum((a - b) ** 2 for a, b in zip(p, y)) / len(y),
        "mae": sum(abs(a - b) for a, b in zip(p, y)) / len(y),
        "pearson": None if denominator == 0 else numerator / denominator,
        "useful_auroc": binary_auroc(p, [int(value > 0) for value in y]),
        "sign_accuracy": sum((a > 0) == (b > 0) for a, b in zip(p, y)) / len(y),
    }


def train_one(*, train_x, train_y, val_x, val_y, architecture, target, config, seed, device):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_x = ((train_x - mean) / scale).to(device)
    val_x_device = ((val_x - mean) / scale).to(device)
    train_y = train_y.to(device)
    model = AcquisitionCritic(
        train_x.shape[1],
        architecture=architecture,
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best = None
    stale = 0
    for epoch in range(int(config["epochs"])):
        order = torch.randperm(len(train_x), generator=generator)
        model.train()
        losses = []
        for start in range(0, len(order), int(config["batch_size"])):
            indices = order[start : start + int(config["batch_size"])].to(device)
            prediction = model(train_x[indices])
            if target == "risk":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    prediction, train_y[indices]
                )
            else:
                loss = torch.nn.functional.mse_loss(prediction, train_y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("critic loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_raw = model(val_x_device)
            if target == "risk":
                val_prediction = torch.sigmoid(val_raw)
                selection_loss = torch.mean((val_prediction - val_y.to(device)) ** 2)
            else:
                val_prediction = val_raw
                selection_loss = torch.mean((val_prediction - val_y.to(device)) ** 2)
        score = float(selection_loss)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best is None or score < best["selection_loss"] - 1e-10:
            best = {
                "epoch": epoch + 1,
                "selection_loss": score,
                "state_dict": state,
                "train_loss": sum(losses) / len(losses),
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(config["early_stopping_patience"]):
            break
    assert best is not None
    model.load_state_dict(best["state_dict"])
    model.eval()
    with torch.no_grad():
        raw = model(val_x_device).cpu()
        prediction = torch.sigmoid(raw) if target == "risk" else raw
    if target == "risk":
        metrics = risk_metrics(prediction.tolist(), (1.0 - val_y).tolist())
    else:
        metrics = regression_metrics(prediction.tolist(), val_y.tolist())
    return {
        "target": target,
        "architecture": architecture,
        "feature_level": config["feature_level"],
        "seed": seed,
        "input_dim": int(train_x.shape[1]),
        "hidden_dim": int(config["hidden_dim"]),
        "mean": mean,
        "scale": scale,
        "state_dict": best["state_dict"],
        "best_epoch": best["epoch"],
        "selection_loss": best["selection_loss"],
        "metrics": metrics,
        "predictions": prediction,
    }


def serializable(result):
    return {
        key: value
        for key, value in result.items()
        if key not in {"mean", "scale", "state_dict", "predictions"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--validation-features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if (
        config.get("schema") != "sequential_critic_config_v1"
        or config.get("test_authorized") is not False
        or config.get("feature_level") not in {"semantic", "state_semantic"}
        or config.get("architectures") != ["linear", "mlp"]
    ):
        raise ValueError("invalid or overly broad sequential critic config")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    train_path = Path(args.train_features).resolve()
    validation_path = Path(args.validation_features).resolve()
    train_payload, train_examples = load_development(train_path, "train")
    validation_payload, validation_examples = load_development(validation_path, "validation")
    if train_payload["metadata"]["benchmark"] != validation_payload["metadata"]["benchmark"]:
        raise ValueError("train/validation benchmark mismatch")
    train_sources = {item.inputs.source_id for item in train_examples}
    validation_sources = {item.inputs.source_id for item in validation_examples}
    train_rgb = {row["image_rgb_sha256"] for row in train_payload["rows"]}
    validation_rgb = {row["image_rgb_sha256"] for row in validation_payload["rows"]}
    if train_sources & validation_sources or train_rgb & validation_rgb:
        raise ValueError("source/RGB leakage between train and validation")

    tick = time.monotonic()
    level = config["feature_level"]
    train_x = matrix(train_examples, level)
    validation_x = matrix(validation_examples, level)
    train_targets = {
        "risk": torch.tensor([item.remaining_risk for item in train_examples]),
        "gain": torch.tensor([item.gain for item in train_examples]),
    }
    validation_targets = {
        "risk": torch.tensor([item.remaining_risk for item in validation_examples]),
        "gain": torch.tensor([item.gain for item in validation_examples]),
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = []
    for target in ("risk", "gain"):
        for architecture in config["architectures"]:
            for seed in config["seeds"]:
                result = train_one(
                    train_x=train_x,
                    train_y=train_targets[target],
                    val_x=validation_x,
                    val_y=validation_targets[target],
                    architecture=architecture,
                    target=target,
                    config=config,
                    seed=int(seed),
                    device=device,
                )
                results.append(result)
                print(json.dumps(serializable(result), allow_nan=False), flush=True)
    selected = {}
    for target in ("risk", "gain"):
        candidates = [item for item in results if item["target"] == target]
        selected[target] = min(
            candidates,
            key=lambda item: (item["selection_loss"], item["architecture"], item["seed"]),
        )
    checkpoint = {
        "schema": "sequential_acquisition_critics_v1",
        "config": config,
        "benchmark": train_payload["metadata"]["benchmark"],
        "feature_level": level,
        "selected": {
            target: {
                key: value
                for key, value in selected[target].items()
                if key != "predictions"
            }
            for target in ("risk", "gain")
        },
    }
    checkpoint_path = output / "critics.pt"
    torch.save(checkpoint, checkpoint_path)
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "schema": "sequential_acquisition_critic_report_v1",
        "scientific_status": "development_only_test_unopened",
        "test_accessed": False,
        "benchmark": train_payload["metadata"]["benchmark"],
        "train_rows": len(train_examples),
        "validation_rows": len(validation_examples),
        "train_sources": len(train_sources),
        "validation_sources": len(validation_sources),
        "source_overlap": 0,
        "rgb_overlap": 0,
        "feature_level": level,
        "results": [serializable(item) for item in results],
        "selected": {target: serializable(selected[target]) for target in selected},
        "config": config,
        "config_sha256": sha256_file(Path(args.config)),
        "train_features_sha256": sha256_file(train_path),
        "validation_features_sha256": sha256_file(validation_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "code_revision": revision,
        "elapsed_seconds": time.monotonic() - tick,
        "device": str(device),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (output / "report.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"report": str(output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
