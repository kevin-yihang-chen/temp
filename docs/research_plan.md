# Research execution plan

## Gate 0 — pipeline validity (implemented in v0.2)

- Validate sibling grouping and pre/post-action field separation.
- Compute SCGR and correlation between entropy reduction and success change.
- Compare answer-now, random, fixed-center, entropy-search, learned-VOI, and
  oracle policies.
- Verify adaptive stopping and account for candidate-evaluation cost.
- Isolate ground truth from proposals by type and split by image/source.
- Predict success gain independently of deployment cost preference.

Exit criterion: deterministic tests pass and the synthetic control demonstrates
that the implementation can recover a usefulness signal that differs from
post-action entropy. This is a software criterion, not a scientific claim.

## Gate 1 — real frozen-VLM diagnostic

- Model: Qwen2.5-VL-3B-Instruct.
- Benchmarks: begin with V*Bench and ChartQA.
- Four crops per state, then ablate nine.
- Use paired deterministic decoding where possible; otherwise run multiple
  sibling samples and estimate expected success.
- Retain the original image when adding each zoom observation.
- Bootstrap confidence intervals by state, never by action row.

Exit criterion: SCGR is non-trivial with confidence intervals, and entropy-based
selection underperforms an oracle task-utility selector on a held-out split.

Implementation checkpoint (2026-08-28): the frozen Qwen backend, UG grid
proposer, offline benchmark exporter, task scorers, atomic resume checkpoints,
provenance sidecars, and state-cluster bootstrap are implemented. On a balanced
1,000-state ChartQA slice with a concise-answer prompt, answer-now accuracy is
0.826, exhaustive four-crop entropy search reaches 0.846 but has utility -0.180
at `lambda=0.05`, and oracle VOI reaches 0.876 with 0.05 calls/state and utility
0.0475. The oracle gain CI is [0.037, 0.064]. The full 2,500-state ChartQA test
replication gives answer-now 0.8128, entropy-search gain 0.0192
[0.0104, 0.0284] with utility -0.1808, and oracle gain 0.0504
[0.0420, 0.0592] with utility 0.0479. This satisfies the diagnostic headroom
criterion but is not a final benchmark claim. On a matched 200-state
candidate-count ablation,
moving from four to nine crops raises oracle gain from 0.045 to 0.060 but lowers
exhaustive entropy-search utility from -0.190 to -0.435. Candidate coverage
matters, but exhaustive grid expansion is not a viable policy at the registered
cost.

## Gate 2 — pre-action value learning

- Freeze the VLM and train the semantic ROI gain head first.
- Encode the image once, ROI-pool candidate regions, and fuse question, global
  image, region, bbox, and baseline-state signals.
- Split by source example/image to prevent sibling leakage.
- Compare regression and pairwise ranking.
- Predict `Delta success`; subtract `lambda * cost` only in the policy.
- Evaluate accuracy-cost frontier, stopping, calibration, and oracle regret.

Exit criterion: learned VOI beats random/fixed policies and improves the
accuracy-cost frontier over entropy search without observing post-action fields.

Implementation checkpoint (2026-08-28): one-pass Qwen spatial-grid extraction,
ROI pooling, image-grouped outer/inner splits, grouped OOF threshold calibration,
semantic MLP and low-capacity similarity ridge baselines are implemented. On the
1,000-state ChartQA diagnostic, direct sparse gain regression collapses and no
deployable learned policy has positive held-out utility. A structured head that
learns dense success-before and success-after probabilities is now the primary
next objective. Gate 2 remains open; do not proceed to Gate 3 based on the current
results.

The full-test scalar diagnostic uses 1,741 image-grouped outer-train and 759
test decisions. The best simple point estimate is entropy-gated fixed crop at
utility 0.00323, but its 95% CI [-0.00843, 0.01482] crosses zero; learned scalar
VOI has utility -0.00165 [-0.01061, 0.00744]. The full semantic experiment also
fails: structured success-difference utility is -0.00264
[-0.00732, 0.00152], validation-thresholded semantic gain is -0.00112, and
semantic similarity ridge is -0.00033. The completed five-split robustness
summary gives structured-head mean
utility -0.00035 (1/5 positive) and thresholded semantic-gain mean 0.00091 (2/5
positive). A compact two-stage rescuability gate is more promising at mean
utility 0.00201 with 4/5 positive splits, but every per-split state-bootstrap
interval crosses zero. This does not satisfy the Gate 2 exit criterion.

A subsequent factorized error-times-rescuability gate is positive in nested
image-grouped OOF evaluation at utility 0.00662, with state interval
`[0.00142, 0.01198]` and image interval `[0.00125, 0.01228]`. Its frozen
source-only transfer to 1,918 image-disjoint ChartQA validation states has
utility 0.00342, but the registered state interval
`[-0.00003, 0.00719]` and image interval `[-0.00013, 0.00725]` narrowly cross
zero, so the primary confirmation fails. A pre-registered same-gate fixed-crop
secondary is positive under both intervals, but post-hoc paired comparisons do
not establish that fixed or learned action selection beats a random crop.

Gate 2 is now split into two explicit questions. The frozen 4,500-image
high-power replication confirms stopping: utility is 0.00363 with state
interval `[0.00070, 0.00669]`, image interval `[0.00070, 0.00671]`, accuracy
gain 0.00683, and 6.4% tool use. The human stratum is positive while the
augmented stratum interval crosses zero. Spatial action selection remains
unconfirmed. Gate 3 may open only for bounded when-to-call integration and
credit-assignment ablations; action-localized RL remains a separate gated
hypothesis rather than a solved crop-ranking result.

A separately frozen chart-layout proposer also fails image-disjoint
confirmation. Its 200-state development advantage over matched UG crops is
0.01625, but the 2,137-state confirmation difference shrinks to 0.00175 with
state interval `[-0.00152, 0.00503]` and image interval
`[-0.00142, 0.00495]`. The pre-registered go/no-go therefore stops the planned
4,500-image treatment. This reinforces the separation between stopping and
action proposal: no current spatial selector has confirmed improvement.

## Gate 3 — VTool-R1 post-training

- Integrate the independently confirmed stopping signal only in a bounded
  when-to-call scaffold first.
- Compare outcome-only reward against visual-action-localized counterfactual
  advantage under matched rollout and GPU budgets.
- Keep token-type credit assignment as an ablation, not an assumed improvement.
- Do not claim or optimize a where-to-look advantage until a spatial selector
  beats matched random/fixed baselines on a new untouched target.
