# Research execution plan

## Gate 0 — pipeline validity (implemented)

- Validate sibling grouping and pre/post-action field separation.
- Compute SCGR and correlation between entropy reduction and success change.
- Compare answer-now, random, fixed-center, entropy-search, learned-VOI, and
  oracle policies.
- Verify adaptive stopping and account for candidate-evaluation cost.

Exit criterion: deterministic tests pass and the synthetic control demonstrates
that the implementation can recover a usefulness signal that differs from
post-action entropy. This is a software criterion, not a scientific claim.

## Gate 1 — real frozen-VLM diagnostic

- Model: Qwen2.5-VL-3B-Instruct.
- Benchmarks: begin with V*Bench and ChartQA.
- Four crops per state, then ablate nine.
- Use paired deterministic decoding where possible; otherwise run multiple
  sibling samples and estimate expected success.
- Bootstrap confidence intervals by state, never by action row.

Exit criterion: SCGR is non-trivial with confidence intervals, and entropy-based
selection underperforms an oracle task-utility selector on a held-out split.

## Gate 2 — pre-action value learning

- Freeze the VLM and train a lightweight value head first.
- Split by source example/image to prevent sibling leakage.
- Compare regression and pairwise ranking.
- Evaluate accuracy-cost frontier, stopping, calibration, and oracle regret.

Exit criterion: learned VOI beats random/fixed policies and improves the
accuracy-cost frontier over entropy search without observing post-action fields.

## Gate 3 — VTool-R1 post-training

- Integrate only after Gate 2 succeeds.
- Compare outcome-only reward against visual-action-localized counterfactual
  advantage under matched rollout and GPU budgets.
- Keep token-type credit assignment as an ablation, not an assumed improvement.
