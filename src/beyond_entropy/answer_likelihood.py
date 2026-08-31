from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .dataset import group_by_decision, read_jsonl
from .rollout import AgentState, VisualObservation
from .schema import ActionRecord


SCHEMA = "visual_action_answer_nll_v1"
TARGET_RULE = "normalized_mode_then_shortest_then_first_index_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_answer(value: object) -> str:
    return " ".join(str(value).strip().split())


def accepted_answers(target: object) -> tuple[str, ...]:
    """Return non-empty accepted answers without silently stringifying a mapping."""

    raw: object
    if isinstance(target, Mapping):
        if "answers" in target:
            raw = target["answers"]
        elif "answer" in target:
            raw = [target["answer"]]
        else:
            raise ValueError("target mapping must contain answers or answer")
    else:
        raw = [target]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("target answers must be a sequence")
    answers = tuple(_normalize_answer(value) for value in raw)
    if not answers or any(not answer for answer in answers):
        raise ValueError("target answers must be non-empty strings")
    return answers


def canonical_target_answer(target: object) -> tuple[str, int, int]:
    """Choose one deterministic answer span for teacher-forced likelihood.

    Repeated normalized answers vote for the target used by TextVQA-style
    manifests.  Ties prefer the shortest span, then the first manifest order.
    The raw answer never needs to be written to the score artifact.
    """

    answers = accepted_answers(target)
    folded = [answer.casefold() for answer in answers]
    counts = Counter(folded)
    best_key = min(
        counts,
        key=lambda key: (
            -counts[key],
            len(key.split()),
            len(key),
            folded.index(key),
        ),
    )
    index = folded.index(best_key)
    return answers[index], index, counts[best_key]


@dataclass(frozen=True)
class ManifestTarget:
    state_id: str
    image_id: str
    source_id: str
    question: str
    model_prompt: str
    answer: str
    answer_index: int
    answer_votes: int
    answer_count: int

    @property
    def answer_sha256(self) -> str:
        return hashlib.sha256(self.answer.encode("utf-8")).hexdigest()


def load_manifest_targets(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, ManifestTarget]:
    manifest_path = Path(path)
    if expected_sha256 is not None and sha256_file(manifest_path) != expected_sha256:
        raise ValueError("answer-likelihood manifest SHA-256 mismatch")
    targets: dict[str, ManifestTarget] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                state_id = str(value["state_id"])
                image_id = str(value["image_id"])
                source_id = str(value["source_id"])
                question = str(value["question"])
                model_prompt = str(value.get("model_prompt", question))
                answers = accepted_answers(value["target"])
                answer, answer_index, answer_votes = canonical_target_answer(value["target"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid answer-likelihood manifest row at {manifest_path}:"
                    f"{line_number}: {exc}"
                ) from exc
            if not all((state_id, image_id, source_id, question, model_prompt)):
                raise ValueError("answer-likelihood manifest identifiers must be non-empty")
            if state_id in targets:
                raise ValueError(f"duplicate manifest state_id: {state_id}")
            targets[state_id] = ManifestTarget(
                state_id=state_id,
                image_id=image_id,
                source_id=source_id,
                question=question,
                model_prompt=model_prompt,
                answer=answer,
                answer_index=answer_index,
                answer_votes=answer_votes,
                answer_count=len(answers),
            )
    if not targets:
        raise ValueError("answer-likelihood manifest is empty")
    return targets


@dataclass(frozen=True)
class AnswerLikelihoodRequest:
    state: AgentState
    observations: tuple[VisualObservation, ...]
    target_answer: str


@dataclass(frozen=True)
class AnswerLikelihoodScore:
    mean_nll: float
    sum_nll: float
    token_count: int

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            raise ValueError("answer-likelihood token_count must be positive")
        if not math.isfinite(self.mean_nll) or not math.isfinite(self.sum_nll):
            raise ValueError("answer-likelihood scores must be finite")
        if self.mean_nll < 0.0 or self.sum_nll < 0.0:
            raise ValueError("answer-likelihood scores must be non-negative")


class Qwen25VLAnswerLikelihood:
    """Teacher-forced target-answer NLL under a frozen Qwen visual context."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str,
        device_map: str = "cuda:0",
        dtype: str = "bfloat16",
        attention_implementation: str = "sdpa",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 768 * 28 * 28,
        local_files_only: bool = True,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        from .qwen_backend import Qwen25VLBackend

        self.backend = Qwen25VLBackend(
            model_name_or_path,
            revision=revision,
            device_map=device_map,
            dtype=dtype,
            attention_implementation=attention_implementation,
            max_new_tokens=1,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            local_files_only=local_files_only,
            system_prompt=system_prompt,
        )

    def score(self, request: AnswerLikelihoodRequest) -> AnswerLikelihoodScore:
        try:
            import torch  # type: ignore[import-not-found]
            import torch.nn.functional as functional  # type: ignore[import-not-found]
            from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Qwen answer likelihood requires the qwen extra") from exc

        if not request.observations or request.observations[0].kind != "ORIGINAL":
            raise ValueError("answer likelihood requires ORIGINAL as the first observation")
        messages = self.backend._messages(request.state, request.observations)
        prompt_text = self.backend.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt_text + request.target_answer
        tokenizer = self.backend.processor.tokenizer
        prompt_token_ids = tokenizer(
            prompt_text, add_special_tokens=False
        ).input_ids
        full_token_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        if full_token_ids[: len(prompt_token_ids)] != prompt_token_ids:
            raise ValueError("target answer changes the tokenized prompt prefix")
        answer_token_count = len(full_token_ids) - len(prompt_token_ids)
        if answer_token_count <= 0:
            raise ValueError("target answer produced no tokens")

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.backend.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        target_device = next(self.backend.model.parameters()).device
        inputs = inputs.to(target_device)
        with torch.inference_mode():
            outputs = self.backend.model(
                **inputs,
                use_cache=False,
                logits_to_keep=answer_token_count + 1,
            )
        logits = outputs.logits[0]
        if logits.shape[0] != answer_token_count + 1:
            raise RuntimeError("Qwen did not honor answer-span logits_to_keep")
        targets = inputs.input_ids[0, -answer_token_count:]
        losses = functional.cross_entropy(
            logits[:-1].to(torch.float32),
            targets,
            reduction="none",
        )
        sum_nll = float(losses.sum().item())
        return AnswerLikelihoodScore(
            mean_nll=sum_nll / answer_token_count,
            sum_nll=sum_nll,
            token_count=answer_token_count,
        )


def request_for_record(
    record: ActionRecord, target: ManifestTarget
) -> AnswerLikelihoodRequest:
    if record.state_id != target.state_id:
        raise ValueError("rollout state does not match manifest target")
    if (
        record.image_id != target.image_id
        or record.source_id != target.source_id
        or record.question != target.question
    ):
        raise ValueError("rollout content does not match manifest target")
    state = AgentState(
        state_id=record.state_id,
        image_id=record.image_id,
        source_id=record.source_id,
        image_path=record.original_image,
        question=record.question,
        model_prompt=target.model_prompt,
    )
    observations = [
        VisualObservation(
            kind="ORIGINAL",
            image_path=record.original_image,
            action_id="original",
            bbox=None,
        )
    ]
    if record.action_type == "ZOOM":
        observations.append(
            VisualObservation(
                kind="ZOOM",
                image_path=record.original_image,
                action_id=record.action_id,
                bbox=record.candidate_bbox,
            )
        )
    return AnswerLikelihoodRequest(
        state=state,
        observations=tuple(observations),
        target_answer=target.answer,
    )


def _ordered_decisions(records: Sequence[ActionRecord]) -> list[list[ActionRecord]]:
    grouped = group_by_decision(records)
    decisions: list[list[ActionRecord]] = []
    for key in sorted(grouped):
        siblings = sorted(
            grouped[key], key=lambda row: (row.action_type != "ANSWER", row.action_id)
        )
        decisions.append(siblings)
    return decisions


def _atomic_write_jsonl(rows: Sequence[Mapping[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"answer-likelihood staging file exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid answer-likelihood checkpoint at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError("answer-likelihood rows must be JSON objects")
            rows.append(value)
    return rows


def score_rollout_answer_likelihood(
    *,
    manifest: str | Path,
    rollouts: str | Path,
    output: str | Path,
    score_request: Callable[[AnswerLikelihoodRequest], AnswerLikelihoodScore],
    expected_manifest_sha256: str | None = None,
    expected_rollouts_sha256: str | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    checkpoint_interval: int = 32,
    resume: bool = False,
    model: str,
    model_revision: str,
    code_revision: str,
    scientific_status: str,
) -> dict[str, Any]:
    """Score complete sibling decisions with restart-safe atomic checkpoints."""

    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid answer-likelihood shard configuration")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    manifest_path = Path(manifest).resolve()
    rollout_path = Path(rollouts).resolve()
    destination = Path(output).resolve()
    manifest_sha256 = sha256_file(manifest_path)
    rollouts_sha256 = sha256_file(rollout_path)
    if expected_manifest_sha256 and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("answer-likelihood manifest SHA-256 mismatch")
    if expected_rollouts_sha256 and rollouts_sha256 != expected_rollouts_sha256:
        raise ValueError("answer-likelihood rollout SHA-256 mismatch")

    targets = load_manifest_targets(
        manifest_path, expected_sha256=expected_manifest_sha256
    )
    records = read_jsonl(rollout_path)
    decisions = _ordered_decisions(records)
    selected = [
        siblings
        for position, siblings in enumerate(decisions)
        if position % shard_count == shard_index
    ]
    if not selected:
        raise ValueError("answer-likelihood shard contains no decisions")
    if {siblings[0].state_id for siblings in decisions} != set(targets):
        raise ValueError("manifest and rollout state coverage differ")

    config = {
        "schema": SCHEMA,
        "target_rule": TARGET_RULE,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "rollouts": str(rollout_path),
        "rollouts_sha256": rollouts_sha256,
        "model": model,
        "model_revision": model_revision,
        "code_revision": code_revision,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "scientific_status": scientific_status,
    }
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected_keys = [
        (row.state_id, row.replicate_id, row.action_id)
        for siblings in selected
        for row in siblings
    ]
    sibling_count = len(selected[0])
    if any(len(siblings) != sibling_count for siblings in selected):
        raise ValueError("answer-likelihood decisions have inconsistent sibling counts")
    rows: list[dict[str, Any]] = []
    if destination.exists():
        if not resume:
            raise FileExistsError(f"answer-likelihood output exists: {destination}")
        rows = _read_rows(destination)
        if len(rows) % sibling_count:
            raise ValueError("answer-likelihood checkpoint ends within a decision")
        observed_keys = [
            (str(row["state_id"]), str(row["replicate_id"]), str(row["action_id"]))
            for row in rows
        ]
        if observed_keys != expected_keys[: len(observed_keys)]:
            raise ValueError("answer-likelihood checkpoint is not an exact prefix")
        if any(row.get("config_sha256") != config_sha256 for row in rows):
            raise ValueError("answer-likelihood checkpoint configuration mismatch")

    completed_decisions = len(rows) // sibling_count
    for offset, siblings in enumerate(selected[completed_decisions:], start=1):
        target = targets[siblings[0].state_id]
        for record in siblings:
            score = score_request(request_for_record(record, target))
            rows.append(
                {
                    "schema": SCHEMA,
                    "config_sha256": config_sha256,
                    "state_id": record.state_id,
                    "replicate_id": record.replicate_id,
                    "source_id": record.source_id,
                    "image_id": record.image_id,
                    "action_id": record.action_id,
                    "action_type": record.action_type,
                    "target_answer_sha256": target.answer_sha256,
                    "target_answer_index": target.answer_index,
                    "target_answer_votes": target.answer_votes,
                    "target_answer_count": target.answer_count,
                    "answer_mean_nll": score.mean_nll,
                    "answer_sum_nll": score.sum_nll,
                    "answer_token_count": score.token_count,
                    "entropy_before": record.entropy_before,
                    "entropy_after": record.entropy_after,
                    "correct_before": record.correct_before,
                    "correct_after": record.correct_after,
                    "tool_cost": record.tool_cost,
                }
            )
        if offset % checkpoint_interval == 0 or offset == len(selected) - completed_decisions:
            _atomic_write_jsonl(rows, destination)
            print(
                json.dumps(
                    {
                        "checkpoint": str(destination),
                        "completed_decisions": completed_decisions + offset,
                        "total_decisions": len(selected),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    output_sha256 = sha256_file(destination)
    provenance = {
        **config,
        "config_sha256": config_sha256,
        "decisions": len(selected),
        "records": len(rows),
        "sources": len({siblings[0].source_id for siblings in selected}),
        "checkpoint_interval": checkpoint_interval,
        "resumed_from_decisions": completed_decisions,
        "output": str(destination),
        "output_sha256": output_sha256,
        "raw_targets_written": False,
    }
    provenance_path = destination.with_suffix(".provenance.json")
    temporary = provenance_path.with_name(provenance_path.name + ".tmp")
    temporary.write_text(
        json.dumps(provenance, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(provenance_path)
    return provenance
