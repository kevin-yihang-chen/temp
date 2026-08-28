#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

from beyond_entropy.benchmarks import extract_answer_letter
from beyond_entropy.cross_benchmark import (
    UG_REFERENCE_COMMIT,
    docvqa_anls_match,
    docvqa_target,
    evalai_normalize_answer,
    textvqa_soft_match,
    textvqa_target,
)
from beyond_entropy.rollout import GroundTruth


REFERENCE_FILES = {
    "docvqa_metric": Path("lmms-eval/lmms_eval/api/metrics.py"),
    "textvqa_processor": Path(
        "lmms-eval/lmms_eval/tasks/_task_utils/vqa_eval_metric.py"
    ),
    "textvqa_task": Path("lmms-eval/lmms_eval/tasks/textvqa/utils.py"),
    "hrbench_task": Path("lmms-eval/lmms_eval/tasks/hrbench/utils.py"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_definitions(
    path: Path,
    names: set[str],
    namespace: dict[str, Any],
) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in names:
                node.decorator_list = []
                selected.append(node)
    found = {node.name for node in selected}  # type: ignore[attr-defined]
    if found != names:
        raise RuntimeError(f"missing definitions in {path}: {sorted(names - found)}")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _assert_close(local: float, reference: float, *, label: str) -> None:
    if abs(local - reference) > 1e-12:
        raise AssertionError(
            f"{label} mismatch: local={local:.17g}, reference={reference:.17g}"
        )


def audit(reference_root: Path) -> dict[str, Any]:
    root = reference_root.resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UG_REFERENCE_COMMIT:
        raise RuntimeError(
            f"reference commit mismatch: expected {UG_REFERENCE_COMMIT}, got {commit}"
        )
    paths = {name: root / relative for name, relative in REFERENCE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing reference files: " + ", ".join(missing))

    processor_namespace = _load_definitions(
        paths["textvqa_processor"],
        {"EvalAIAnswerProcessor"},
        {"re": re},
    )
    processor = processor_namespace["EvalAIAnswerProcessor"]()
    normalization_cases = {
        "",
        "The TWO, cats!",
        "cant",
        "1,000",
        "A red-blue sign",
        "value 1.5.",
        "someone's answer",
        "yes/no",
        "(three)",
        "line\nbreak",
    }
    tokens = ("The", "cant", "TWO", "1,000", "red-blue", "value 1.5")
    separators = (" ", ", ", " / ", "\n")
    normalization_cases.update(
        left + separator + right
        for left in tokens
        for separator in separators
        for right in tokens
    )
    for case in sorted(normalization_cases):
        normalized_local = evalai_normalize_answer(case)
        normalized_reference = processor(case)
        if normalized_local != normalized_reference:
            raise AssertionError(
                f"TextVQA normalization mismatch for {case!r}: "
                f"local={normalized_local!r}, reference={normalized_reference!r}"
            )

    text_task_namespace = _load_definitions(
        paths["textvqa_task"],
        {"textvqa_process_results"},
        {
            "EvalAIAnswerProcessor": processor_namespace["EvalAIAnswerProcessor"],
            "statistics": statistics,
        },
    )
    reference_textvqa = text_task_namespace["textvqa_process_results"]
    textvqa_cases = 0
    for matches in range(11):
        text_references = ["two"] * matches + ["other"] * (10 - matches)
        text_local = textvqa_soft_match(
            "2",
            GroundTruth(textvqa_target(text_references)),
        )
        text_reference = float(
            reference_textvqa(
                {"answers": list(text_references), "question_id": 0},
                ["2"],
            )["exact_match"]
        )
        _assert_close(text_local, text_reference, label=f"TextVQA matches={matches}")
        textvqa_cases += 1

    metric_namespace = _load_definitions(
        paths["docvqa_metric"],
        {"levenshtein_distance", "anls"},
        {},
    )
    reference_anls = metric_namespace["anls"]
    docvqa_inputs = (
        (("abcd",), "abcd"),
        (("abcd",), "abce"),
        (("abcd",), "abxx"),
        (("abcd",), "xxxx"),
        (("a   b",), "acb"),
        (("north", "south"), "south"),
        (("",), ""),
        (("invoice 42",), "INVOICE   42"),
    )
    for doc_references, prediction in docvqa_inputs:
        doc_local = docvqa_anls_match(
            prediction,
            GroundTruth(docvqa_target(doc_references)),
        )
        doc_reference = float(reference_anls(doc_references, [prediction])["anls"])
        _assert_close(
            doc_local,
            doc_reference,
            label=f"DocVQA {doc_references!r}/{prediction!r}",
        )

    hrbench_namespace = _load_definitions(
        paths["hrbench_task"],
        {"extract_answer_letter"},
        {"re": re},
    )
    reference_extract = hrbench_namespace["extract_answer_letter"]
    hrbench_responses = (
        "A",
        "(B)",
        "C.",
        "Option D",
        "The answer is A.",
        "I choose B because it is visible.",
        "No valid option",
        "A or B",
        "",
    )
    for response in hrbench_responses:
        extracted_local = extract_answer_letter(response)
        extracted_reference = reference_extract(response)
        if extracted_local != extracted_reference:
            raise AssertionError(
                f"HRBench extraction mismatch for {response!r}: "
                f"local={extracted_local!r}, reference={extracted_reference!r}"
            )

    return {
        "passed": True,
        "reference_commit": commit,
        "reference_files": {
            str(REFERENCE_FILES[name]): _sha256(path)
            for name, path in sorted(paths.items())
        },
        "checks": {
            "docvqa_anls_cases": len(docvqa_inputs),
            "hrbench_extraction_cases": len(hrbench_responses),
            "textvqa_normalization_cases": len(normalization_cases),
            "textvqa_soft_accuracy_cases": textvqa_cases,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Differentially audit cross-benchmark scorers against pinned UG code"
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args.reference_root)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
