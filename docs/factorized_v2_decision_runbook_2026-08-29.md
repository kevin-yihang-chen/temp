# Factorized-v2 calibration and formal decision runbook

Status: frozen operational runbook written before the fresh calibration rollout
or features completed. It does not modify the preregistered statistical rule.

## Immutable inputs and jobs

- live experiment revision: `d85c8d57db2b0c663f760e1fc43a0a9920297422`;
- pre-result formal implementation content commit:
  `38d709849a480a6a0bd6fadb4fd5b08bf308e163`;
- calibration rollout: Slurm `191716`;
- label-free feature extraction: Slurm `191717`, `afterok:191716`;
- fixed-sequence calibration: Slurm `191792`, `afterok:191717`;
- candidate SHA-256:
  `9a6c9d032ebdbc271b7d3c829fbb3d6ff167cac01b54ce75adc8da86e3063342`;
- allocation SHA-256:
  `bc0ecb4b6f49a5b0e92b90b4c30620f72246722370d59c8078753d5846f5e9b6`;
- protocol SHA-256:
  `babf01d4090263d1cfcb28c42f86f7b13ae9de4bb6bab0ca10d6e4707f02e2ca`.

The main worktree must remain clean at `d85c8d5` until job `191792` reaches a
terminal state. The formal branch must remain clean and must not be merged
before calibration output exists. Documentation-only descendants may record
this runbook, but no formal implementation file may change after `38d7098`.

## Calibration completion audit

After `191792` reports `COMPLETED` with exit code zero:

1. Confirm `191716`, `191717`, and `191792` terminal states and exit codes from
   Slurm accounting, not only log files.
2. Confirm the rollout bank contains exactly `4747 * 5 = 23735` records, one
   answer and four zoom siblings per decision, across exactly 3,000 sources.
3. Confirm the rollout audit passed and is bound to the frozen manifest, model
   revision, scientific-status string, and revision `d85c8d5`.
4. Confirm semantic features cover all 4,747 decisions and store no outcomes.
5. Compute and record SHA-256 for rollouts, rollout audit, label-free features,
   `calibration.json`, and `model.json` before interpreting any metric.
6. Confirm `calibration.json` and `model.json` are the only files created in
   `artifacts/textvqa-train-factorized-v2/fixed-sequence-calibrated/`.
7. Only then read `selection_status`, `selected_threshold`, tested threshold
   count, stopping threshold, and the per-threshold risk table.

Do not rerun calibration with a different order, risk bound, p-value cutoff,
utility floor, call-rate floor, source weighting, or threshold grid.

## Branch A: calibration does not select a policy

This branch includes `no_non_degenerate_safe_threshold`, answer-now selection,
or any failed input/provenance invariant.

1. Mark the factorized-v2 branch as a negative independent calibration.
2. Do not materialize `data/textvqa-train-factorized-v2/formal-test/`.
3. Do not inspect targets or collect rollouts for the 5,953 formal identities.
4. Retain the calibration report, calibrated answer-now model, job logs, hashes,
   and earlier positive OOF result together.
5. Render a negative result note explaining whether failure arose from risk,
   insufficient call rate, or utility below `0.001`.
6. The pre-result formal implementation may be retained on its branch for
   reproducibility, but it must not be used to open formal data.
7. Any new method becomes a new branch with new development evidence and a new
   independent calibration/formal allocation. The current formal identities
   remain untouched; they are not a development set.

## Branch B: calibration selects a non-degenerate safe threshold

This branch requires the exact status
`selected_non_degenerate_safe_threshold`, a threshold from the frozen sequence,
at least 1% source call rate, at least `0.001` source utility, and both fixed
risk tests accepted before the first stopping failure.

1. Fast-forward main to the clean pre-result formal branch tip; verify `38d7098`
   is its ancestor and no formal implementation file changed after that commit.
   Do not cherry-pick with edits or squash, so pre-calibration history remains.
2. Verify the merged tree is clean and rerun all tests and targeted type checks.
3. Run `scripts/freeze_factorized_v2_formal_policy.sh`. This must reject any
   implementation commit timestamp later than `calibration.json`.
4. Hash and validate `policy-freeze.json`; confirm it pins the candidate,
   selected model/threshold, calibration report and inputs, protocol, collector,
   feature contract, evaluator, matched-budget baselines, and Slurm jobs.
5. Submit `scripts/submit_textvqa_factorized_v2_formal_export.sh` with state-change
   email enabled. Do not inspect target content after export.
6. After the export job succeeds, verify exactly 5,953 source/RGB identities,
   zero overlap with calibration and all parent roles, and bind manifest,
   provenance, audit, model, and policy-freeze hashes.
7. Submit `scripts/submit_textvqa_factorized_v2_formal.sh`. Its rollout, feature,
   and evaluator jobs must remain a single `afterok` chain with state-change
   email enabled.
8. Do not alter the threshold, model, call budget, evaluator, bootstrap, scorer,
   or baseline definitions while this chain runs.

## One-shot formal decision

The formal branch passes only if all preregistered conditions are true:

- source-balanced utility is positive;
- the two-sided 97.5% whole-source bootstrap lower endpoint is strictly positive;
- question-weighted utility is positive;
- source-balanced call rate is at least 1%;
- evaluated threshold equals the fixed-sequence calibration choice;
- all frozen hashes and identity audits match.

The report must include raw gain, call rate, cost, gain per call, harm, negative
call mass, unnecessary calls, correct stopping, crop rescue, oracle regret,
matched-budget random/entropy gates, fixed/random crops, and UG-style exhaustive
entropy search charged for all candidate calls.

If any pass condition fails, retain the one-shot result as negative and do not
select another threshold, model, feature, or baseline on the opened formal bank.

## Implementation failures

Before formal export, a purely mechanical implementation bug may be fixed only
if it does not use calibration outcomes to change the scientific rule. The fix,
tests, reason, and new hash must be committed and included in a new policy
freeze before formal materialization.

After any formal outcome has been collected, no change to scientific logic is
allowed. A recoverable execution interruption may resume the identical hashed
collector or feature extractor. A semantic mismatch, corrupted identity bank,
changed evaluator, or unrecoverable provenance failure invalidates the run and
must not be repaired by tuning on formal outcomes.
