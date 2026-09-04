from __future__ import annotations

import math
import os
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .predictability_audit import (
    AUDIT_BENCHMARKS,
    PREDICTOR_LEVELS,
    TARGET_FAMILIES,
    AuditVerdict,
    BenchmarkVerdictEvidence,
    classify_completed_audit,
)


FINAL_AUDIT_FILENAME = "PREDICTABILITY_AUDIT.md"


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _benchmark_mapping(report: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _mapping(report.get(key), name=key)
    if set(value) != set(AUDIT_BENCHMARKS):
        raise ValueError(f"{key} must contain exactly the three frozen benchmarks")
    return value


def _validate_bootstrap(value: Any, *, name: str) -> None:
    interval = _mapping(value, name=name)
    if (
        interval.get("resamples") != 20_000
        or _number(interval.get("confidence_level"), name=f"{name}.confidence") != 0.95
        or interval.get("resampling_unit") != "source_id"
    ):
        raise ValueError(f"{name} does not use the frozen 20,000-source bootstrap")
    for field in ("point", "lower", "upper"):
        _number(interval.get(field), name=f"{name}.{field}")


def validate_completed_formal_report(report: Mapping[str, Any]) -> None:
    """Fail closed unless this is the complete, one-shot, formal test report."""

    if report.get("schema") != "predictability_matrix_report_v3":
        raise ValueError("final audit requires predictability matrix report v3")
    if report.get("formal_claim_eligible") is not True:
        raise ValueError("final audit requires formal_claim_eligible=true")
    if report.get("frozen_before_test") is not True:
        raise ValueError("final audit requires a matrix frozen before test")
    matrix = _mapping(report.get("matrix"), name="matrix")
    if (
        matrix.get("complete") is not True
        or matrix.get("expected_cells") != 36
        or matrix.get("completed_cells") != 36
        or matrix.get("missing") != []
    ):
        raise ValueError("final audit requires the complete registered 36-cell matrix")
    if tuple(report.get("seeds", ())) != (17, 29, 47):
        raise ValueError("final audit requires the three frozen seeds")
    for key in (
        "split_audits",
        "strong_baselines",
        "oracle_headroom",
        "primary_deployable",
        "post_action_probe",
        "representation_diagnostic",
    ):
        _benchmark_mapping(report, key)
    for benchmark, value in _benchmark_mapping(report, "split_audits").items():
        audit = _mapping(value, name=f"split_audits.{benchmark}")
        test_role = _mapping(
            audit.get("test_role_validation"),
            name=f"split_audits.{benchmark}.test_role_validation",
        )
        if audit.get("passed") is not True or test_role.get("passed") is not True:
            raise ValueError(f"{benchmark} source/RGB split audit did not pass")
    cells = report.get("cells")
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise ValueError("formal report cells must be a sequence")
    cell_keys: list[tuple[str, str, str]] = []
    for raw_cell in cells:
        cell = _mapping(raw_cell, name="cell")
        cell_key = (
            str(cell.get("benchmark")),
            str(cell.get("predictor_level")),
            str(cell.get("target")),
        )
        cell_keys.append(cell_key)
        seeds = cell.get("seeds")
        if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
            raise ValueError("formal cell seeds must be a sequence")
        if tuple(_mapping(seed, name="seed").get("seed") for seed in seeds) != (
            17,
            29,
            47,
        ):
            raise ValueError("formal cell seed inventory differs")
        for seed in seeds:
            seed_report = _mapping(seed, name="seed")
            _validate_bootstrap(
                seed_report.get("paired_vs_strongest_baseline"),
                name="cell paired bootstrap",
            )
    expected_cell_keys = {
        (benchmark, level, target)
        for benchmark in AUDIT_BENCHMARKS
        for level in PREDICTOR_LEVELS
        for target in TARGET_FAMILIES
    }
    if len(cell_keys) != 36 or set(cell_keys) != expected_cell_keys:
        raise ValueError("formal report cell records differ from the 36-cell matrix")
    for benchmark in AUDIT_BENCHMARKS:
        primary = _mapping(
            _benchmark_mapping(report, "primary_deployable")[benchmark],
            name=f"{benchmark}.primary",
        )
        post = _mapping(
            _mapping(
                _benchmark_mapping(report, "post_action_probe")[benchmark],
                name=f"{benchmark}.post",
            ).get("ensemble"),
            name=f"{benchmark}.post.ensemble",
        )
        representation = _mapping(
            _benchmark_mapping(report, "representation_diagnostic")[benchmark],
            name=f"{benchmark}.representation",
        )
        oracle = _mapping(
            _mapping(
                _benchmark_mapping(report, "oracle_headroom")[benchmark],
                name=f"{benchmark}.headroom",
            ).get("privileged_binary_oracle"),
            name=f"{benchmark}.oracle",
        )
        for name, interval in (
            ("primary test", primary.get("paired_vs_strongest_baseline")),
            ("post-action test", post.get("paired_vs_answer_now")),
            (
                "L3 validation",
                representation.get("validation_paired_vs_strongest_baseline"),
            ),
            (
                "L3 test",
                representation.get("test_paired_vs_strongest_baseline"),
            ),
            ("oracle test", oracle.get("paired_vs_answer_now")),
        ):
            _validate_bootstrap(interval, name=f"{benchmark}.{name}")
    access = _mapping(report.get("one_shot_test_access"), name="one_shot_test_access")
    required_access = {
        "ledger",
        "ledger_sha256",
        "frozen_model_sha256",
        "frozen_report_sha256",
        "protocol_sha256",
        "allocation_report_sha256",
        "code_revision",
        "test_artifacts",
    }
    if not required_access <= set(access):
        raise ValueError("one-shot test access evidence is incomplete")
    for name in (
        "ledger_sha256",
        "frozen_model_sha256",
        "frozen_report_sha256",
        "protocol_sha256",
        "allocation_report_sha256",
    ):
        value = access.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"one-shot test access {name} is invalid")
    revision = access.get("code_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("one-shot test access code revision is invalid")


def build_verdict_evidence(
    report: Mapping[str, Any],
) -> tuple[BenchmarkVerdictEvidence, ...]:
    validate_completed_formal_report(report)
    headrooms = _benchmark_mapping(report, "oracle_headroom")
    primaries = _benchmark_mapping(report, "primary_deployable")
    post_actions = _benchmark_mapping(report, "post_action_probe")
    representations = _benchmark_mapping(report, "representation_diagnostic")
    evidence: list[BenchmarkVerdictEvidence] = []
    for benchmark in AUDIT_BENCHMARKS:
        headroom = _mapping(headrooms[benchmark], name=f"{benchmark}.headroom")
        oracle = _mapping(
            headroom.get("privileged_binary_oracle"),
            name=f"{benchmark}.privileged_binary_oracle",
        )
        primary = _mapping(primaries[benchmark], name=f"{benchmark}.primary")
        primary_paired = _mapping(
            primary.get("paired_vs_strongest_baseline"),
            name=f"{benchmark}.primary.paired",
        )
        operating = _mapping(
            primary.get("operating_point_vs_strongest_baseline"),
            name=f"{benchmark}.primary.operating_point",
        )
        post = _mapping(post_actions[benchmark], name=f"{benchmark}.post_action")
        post_ensemble = _mapping(
            post.get("ensemble"), name=f"{benchmark}.post_action.ensemble"
        )
        post_paired = _mapping(
            post_ensemble.get("paired_vs_answer_now"),
            name=f"{benchmark}.post_action.paired",
        )
        representation = _mapping(
            representations[benchmark], name=f"{benchmark}.representation"
        )
        validation_paired = _mapping(
            representation.get("validation_paired_vs_strongest_baseline"),
            name=f"{benchmark}.representation.validation",
        )
        test_paired = _mapping(
            representation.get("test_paired_vs_strongest_baseline"),
            name=f"{benchmark}.representation.test",
        )
        evidence.append(
            BenchmarkVerdictEvidence(
                benchmark=benchmark,
                oracle_utility=_number(
                    oracle.get("utility"), name=f"{benchmark}.oracle.utility"
                ),
                primary_deployable_beats_strongest_baseline_lower_ci=_number(
                    primary_paired.get("lower"),
                    name=f"{benchmark}.primary.lower",
                ),
                maximum_lower_ci_across_all_deployable_policies=_number(
                    primary.get(
                        "maximum_lower_ci_across_all_deployable_cells_and_primary"
                    ),
                    name=f"{benchmark}.all_deployable.maximum_lower",
                ),
                deployable_accuracy_cost_pareto=_boolean(
                    operating.get("accuracy_cost_pareto"),
                    name=f"{benchmark}.primary.pareto",
                ),
                deployable_rescue_precision_higher=_boolean(
                    operating.get("rescue_precision_higher"),
                    name=f"{benchmark}.primary.rescue",
                ),
                deployable_harm_rate_not_higher=_boolean(
                    operating.get("harm_rate_not_higher"),
                    name=f"{benchmark}.primary.harm",
                ),
                post_action_probe_utility_lower_ci=_number(
                    post_paired.get("lower"), name=f"{benchmark}.post.lower"
                ),
                l3_in_domain_improvement_lower_ci=_number(
                    validation_paired.get("lower"),
                    name=f"{benchmark}.l3.validation.lower",
                ),
                l3_image_or_cross_domain_improvement_upper_ci=_number(
                    test_paired.get("upper"), name=f"{benchmark}.l3.test.upper"
                ),
            )
        )
    return tuple(evidence)


def classify_formal_report(report: Mapping[str, Any]) -> AuditVerdict:
    return classify_completed_audit(build_verdict_evidence(report))


def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    return f"{_number(value, name='rendered metric'):.6f}"


def _mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows]
    finite = [float(value) for value in values if isinstance(value, (int, float))]
    return None if not finite else mean(finite)


def _recommendation(verdict: AuditVerdict) -> str:
    return {
        AuditVerdict.GO: (
            "Proceed only to a selective router, then post-training and RL credit "
            "assignment around the same fixed visual tool."
        ),
        AuditVerdict.PIVOT: (
            "Stop optimizing static pre-action gates. Study a cheap probe that "
            "acquires partial visual evidence, updates uncertainty, and then decides "
            "whether to continue."
        ),
        AuditVerdict.REPRESENTATION: (
            "Stop single-domain router tuning. Study only domain-general "
            "tool-utility representations and cross-domain transfer."
        ),
        AuditVerdict.STOP: (
            "Stop router, post-training, and uncertainty work for this pairing; "
            "replace the visual tool or the benchmark/task pairing."
        ),
    }[verdict]


def render_predictability_audit(
    report: Mapping[str, Any], *, report_sha256: str
) -> str:
    """Render the sole terminal artifact, refusing inconclusive evidence."""

    if len(report_sha256) != 64:
        raise ValueError("formal report SHA-256 is invalid")
    evidence = build_verdict_evidence(report)
    verdict = classify_completed_audit(evidence)
    baselines = _benchmark_mapping(report, "strong_baselines")
    headrooms = _benchmark_mapping(report, "oracle_headroom")
    primaries = _benchmark_mapping(report, "primary_deployable")
    post_actions = _benchmark_mapping(report, "post_action_probe")
    representations = _benchmark_mapping(report, "representation_diagnostic")
    evidence_map = {item.benchmark: item for item in evidence}
    lines = [
        "# Pre-action visual-tool utility predictability audit",
        "",
        f"**Frozen verdict: {verdict.value}.**",
        "",
        "> This verdict uses the complete preregistered 3 benchmark x 4 predictor "
        "levels x 3 target formulations matrix, three fixed seeds, and one "
        "source/RGB-disjoint held-out test transaction.",
        "",
        "## Answer to the research question",
        "",
        _recommendation(verdict),
        "",
        "## Oracle headroom",
        "",
        "| Benchmark | Always-call utility | Binary-oracle utility | Oracle 95% CI | Rescue rate | Harm rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for benchmark in AUDIT_BENCHMARKS:
        item = _mapping(headrooms[benchmark], name=f"{benchmark}.headroom")
        oracle = _mapping(item["privileged_binary_oracle"], name="oracle")
        interval = _mapping(oracle["paired_vs_answer_now"], name="oracle interval")
        targets = _mapping(item["raw_targets"], name="raw targets")
        lines.append(
            f"| {benchmark} | {_fmt(_mapping(item['always_call'], name='always')['utility'])} "
            f"| {_fmt(oracle['utility'])} | [{_fmt(interval['lower'])}, "
            f"{_fmt(interval['upper'])}] | {_fmt(targets['rescue_rate'])} | "
            f"{_fmt(targets['harm_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Primary deployable policy and strongest baseline",
            "",
            "| Benchmark | Strongest baseline | Delta utility 95% CI | Accuracy (candidate/base) | Cost (candidate/base) | Pareto | Rescue precision higher | Harm no higher |",
            "|---|---|---:|---:|---:|:---:|:---:|:---:|",
        ]
    )
    for benchmark in AUDIT_BENCHMARKS:
        item = _mapping(primaries[benchmark], name=f"{benchmark}.primary")
        paired = _mapping(item["paired_vs_strongest_baseline"], name="paired")
        op = _mapping(item["operating_point_vs_strongest_baseline"], name="op")
        lines.append(
            f"| {benchmark} | `{item['strongest_baseline']}` | "
            f"{_fmt(paired['point'])} [{_fmt(paired['lower'])}, {_fmt(paired['upper'])}] | "
            f"{_fmt(op['candidate_accuracy'])}/{_fmt(op['baseline_accuracy'])} | "
            f"{_fmt(op['candidate_cost'])}/{_fmt(op['baseline_cost'])} | "
            f"{'yes' if op['accuracy_cost_pareto'] else 'no'} | "
            f"{'yes' if op['rescue_precision_higher'] else 'no'} | "
            f"{'yes' if op['harm_rate_not_higher'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "Undefined rescue/harm rates for a zero-call policy are compared as "
            "0.0, exactly as frozen before test.",
            "",
            "## Rescue and harm at the selected operating point",
            "",
            "| Benchmark | Candidate rescue precision | Baseline rescue precision | Candidate harm/call | Baseline harm/call | Gain/call |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for benchmark in AUDIT_BENCHMARKS:
        item = _mapping(primaries[benchmark], name=f"{benchmark}.primary")
        op = _mapping(item["operating_point_vs_strongest_baseline"], name="op")
        policy = _mapping(item["test_policy"], name="policy")
        lines.append(
            f"| {benchmark} | {_fmt(op['candidate_rescue_precision'])} | "
            f"{_fmt(op['baseline_rescue_precision'])} | "
            f"{_fmt(op['candidate_harm_rate_per_call'])} | "
            f"{_fmt(op['baseline_harm_rate_per_call'])} | "
            f"{_fmt(policy['marginal_gain_per_call'])} |"
        )

    lines.extend(
        [
            "",
            "## Predictor ladder",
            "",
            "Each row is the mean across the three fixed seeds; model variant and "
            "threshold selection used validation only.",
            "",
            "| Benchmark | Level | Target | Utility | AUROC | AUPRC | Brier | Calibration | Rescue AUPRC | Harm AUPRC |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    cells = report.get("cells")
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise ValueError("formal report cells must be a sequence")
    cell_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for raw_cell in cells:
        cell = _mapping(raw_cell, name="cell")
        key = (
            str(cell.get("benchmark")),
            str(cell.get("predictor_level")),
            str(cell.get("target")),
        )
        cell_index[key] = cell
    expected_keys = {
        (benchmark, level, target)
        for benchmark in AUDIT_BENCHMARKS
        for level in PREDICTOR_LEVELS
        for target in TARGET_FAMILIES
    }
    if set(cell_index) != expected_keys:
        raise ValueError("formal report cell inventory differs from the 36-cell matrix")
    for benchmark in AUDIT_BENCHMARKS:
        for level in PREDICTOR_LEVELS:
            for target in TARGET_FAMILIES:
                cell = cell_index[(benchmark, level, target)]
                seeds = cell.get("seeds")
                if not isinstance(seeds, Sequence) or len(seeds) != 3:
                    raise ValueError("formal cell requires three seed reports")
                policies = [
                    _mapping(_mapping(seed, name="seed")["test_policy"], name="policy")
                    for seed in seeds
                ]
                predictions = [
                    _mapping(
                        _mapping(seed, name="seed")["test_prediction"],
                        name="prediction",
                    )
                    for seed in seeds
                ]
                lines.append(
                    f"| {benchmark} | `{level}` | `{target}` | "
                    f"{_fmt(_mean_metric(policies, 'incremental_utility'))} | "
                    f"{_fmt(_mean_metric(predictions, 'auroc'))} | "
                    f"{_fmt(_mean_metric(predictions, 'auprc'))} | "
                    f"{_fmt(_mean_metric(predictions, 'brier'))} | "
                    f"{_fmt(_mean_metric(predictions, 'calibration_error'))} | "
                    f"{_fmt(_mean_metric(predictions, 'rescue_auprc'))} | "
                    f"{_fmt(_mean_metric(predictions, 'harm_auprc'))} |"
                )

    lines.extend(
        [
            "",
            "### Validation-selected seed score curves",
            "",
            "These are test-time means of the three independently frozen seed "
            "curves. They expose accuracy versus call rate, accuracy versus visual "
            "cost, and utility versus call rate without selecting a test point.",
            "",
            "| Benchmark | Requested call rate | Realized call rate | Accuracy | Visual cost | Utility |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for benchmark in AUDIT_BENCHMARKS:
        primary = _mapping(primaries[benchmark], name="primary")
        selected = primary.get("selected_cell_keys")
        if not isinstance(selected, Sequence) or len(selected) != 3:
            raise ValueError("primary deployable requires three selected cell keys")
        seed_curves: list[Sequence[Any]] = []
        for raw_key in selected:
            selected_key = _mapping(raw_key, name="selected cell key")
            seed = selected_key.get("seed")
            cell = cell_index[
                (
                    benchmark,
                    str(selected_key.get("level")),
                    str(selected_key.get("target")),
                )
            ]
            seed_reports = cell.get("seeds")
            if not isinstance(seed_reports, Sequence):
                raise ValueError("selected cell seed reports are missing")
            matches = [
                _mapping(item, name="selected seed")
                for item in seed_reports
                if _mapping(item, name="selected seed").get("seed") == seed
            ]
            if len(matches) != 1:
                raise ValueError("selected cell seed report is not unique")
            curve = matches[0].get("test_curve")
            if not isinstance(curve, Sequence) or isinstance(curve, (str, bytes)):
                raise ValueError("selected seed test curve is missing")
            seed_curves.append(curve)
        if not seed_curves or any(
            len(curve) != len(seed_curves[0]) for curve in seed_curves
        ):
            raise ValueError("selected seed test curves are not aligned")
        for index in range(len(seed_curves[0])):
            rows = [
                _mapping(curve[index], name="selected curve point")
                for curve in seed_curves
            ]
            requested = _number(
                rows[0].get("requested_call_rate"), name="requested call rate"
            )
            if any(
                _number(row.get("requested_call_rate"), name="requested call rate")
                != requested
                for row in rows[1:]
            ):
                raise ValueError("selected seed requested call rates differ")
            lines.append(
                f"| {benchmark} | {_fmt(requested)} | "
                f"{_fmt(_mean_metric(rows, 'call_rate'))} | "
                f"{_fmt(_mean_metric(rows, 'accuracy'))} | "
                f"{_fmt(_mean_metric(rows, 'cost'))} | "
                f"{_fmt(_mean_metric(rows, 'incremental_utility'))} |"
            )

    lines.extend(
        [
            "",
            "## Accuracy-cost frontier",
            "",
            "The JSON report contains every per-seed curve point. The table below "
            "shows the actual frozen majority-vote operating point against its "
            "validation-selected strongest baseline.",
            "",
            "| Benchmark | Policy | Accuracy | Visual cost | Tool-call rate | Utility |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for benchmark in AUDIT_BENCHMARKS:
        primary = _mapping(primaries[benchmark], name="primary")
        baseline_report = _mapping(baselines[benchmark], name="baseline")
        strongest_name = str(primary["strongest_baseline"])
        baseline_test = _mapping(
            _mapping(baseline_report["test"], name="baseline test")[strongest_name],
            name="strongest metrics",
        )
        for name, metrics in (
            (
                "primary deployable",
                _mapping(primary["test_policy"], name="primary test"),
            ),
            (strongest_name, baseline_test),
        ):
            lines.append(
                f"| {benchmark} | `{name}` | {_fmt(metrics['accuracy'])} | "
                f"{_fmt(metrics['cost'])} | {_fmt(metrics['call_rate'])} | "
                f"{_fmt(metrics['incremental_utility'])} |"
            )

    lines.extend(
        [
            "",
            "## Diagnostic upper bound and representation check",
            "",
            "| Benchmark | Post-action vs answer-now 95% CI | L3 validation vs baseline 95% CI | L3 test vs baseline 95% CI | Max lower CI over all deployables |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for benchmark in AUDIT_BENCHMARKS:
        post = _mapping(
            _mapping(post_actions[benchmark], name="post")["ensemble"],
            name="post ensemble",
        )
        post_paired = _mapping(post["paired_vs_answer_now"], name="post paired")
        representation = _mapping(representations[benchmark], name="representation")
        validation = _mapping(
            representation["validation_paired_vs_strongest_baseline"],
            name="l3 validation",
        )
        test = _mapping(
            representation["test_paired_vs_strongest_baseline"], name="l3 test"
        )
        evidence_item = evidence_map[benchmark]
        lines.append(
            f"| {benchmark} | [{_fmt(post_paired['lower'])}, {_fmt(post_paired['upper'])}] | "
            f"[{_fmt(validation['lower'])}, {_fmt(validation['upper'])}] | "
            f"[{_fmt(test['lower'])}, {_fmt(test['upper'])}] | "
            f"{_fmt(evidence_item.maximum_lower_ci_across_all_deployable_policies)} |"
        )

    access = _mapping(report["one_shot_test_access"], name="one-shot access")
    lines.extend(
        [
            "",
            "## Bootstrap and integrity",
            "",
            "All primary confidence intervals are paired whole-source bootstrap "
            "intervals with 20,000 resamples. The complete machine-readable report "
            "retains the deterministic seed schedule and all individual curves.",
            "",
            f"- Formal test report SHA-256: `{report_sha256}`",
            f"- Frozen matrix SHA-256: `{access['frozen_model_sha256']}`",
            f"- Frozen inventory SHA-256: `{access['frozen_report_sha256']}`",
            f"- Test access ledger SHA-256: `{access['ledger_sha256']}`",
            f"- Protocol SHA-256: `{access['protocol_sha256']}`",
            f"- Untouched-test allocation report SHA-256: `{access['allocation_report_sha256']}`",
            f"- Clean code revision: `{access['code_revision']}`",
            "",
            "The test transaction is consumed. This result must not be used to "
            "select a replacement predictor, threshold, feature, seed, or verdict rule.",
            "",
        ]
    )
    return "\n".join(lines)


def write_predictability_audit(path: str | Path, markdown: str) -> None:
    destination = Path(path).resolve()
    if destination.name != FINAL_AUDIT_FILENAME:
        raise ValueError(f"final audit filename must be {FINAL_AUDIT_FILENAME}")
    if destination.exists():
        raise FileExistsError("refusing to overwrite final predictability audit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError("final audit staging file already exists")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(markdown)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
