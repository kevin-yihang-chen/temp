# Review remediation status

This document maps the external code review to version 0.2 changes. It separates
resolved implementation issues from work that requires a real VLM environment.

| Review item | v0.2 status | Implementation |
|---|---|---|
| Proposal can see target | Resolved | `ProposalFunction` accepts only `AgentState`; `GroundTruth` is scorer-only |
| No image-question-region model | Real adapter and diagnostic implemented | One frozen Qwen image pass, raster-restored spatial tokens, ROI pooling, question/global/region fusion |
| Entropy-search utility cost | Resolved | Policy utility subtracts `decision.visual_cost`, including every executed candidate |
| ZOOM replaces rather than adds evidence | Resolved at interface level | Backend receives `[ORIGINAL, ZOOM]`; VTool adapter still pending |
| Binary realized VOI labels | Supported explicitly | `replicate_id` and paired `generation_seed`; model targets observed success gain |
| Lambda embedded in target | Resolved | Gain model predicts `delta_success`; policy applies lambda at runtime |
| State-only split | Resolved | `split_by_group` defaults to `image_id` and supports `source_id` |
| Entropy baseline cannot stop | Resolved | Pre-action entropy threshold and post-action reduction-threshold baselines |
| SCGR too narrow | Resolved | Strict/non-beneficial SCGR, confidence precision, Top-1 mismatch and regret |
| Non-cost-aware tool metrics | Resolved | Policy utility, cost-aware unnecessary calls and marginal gain per call |
| Backend lacks batch/cache/seed | Resolved at interface level | Batch protocol, serial fallback, in-memory request cache and paired seeds |

## Remaining scientific work

The real adapter and frozen rollouts now exist, so the blocker is scientific
rather than infrastructural. The 1,000-state ChartQA diagnostic contains only 50
states with a helpful crop. Direct `{-1, 0, +1}` gain regression reaches its best
validation loss at epoch 1 and its affine calibration slope is zero. Scalar,
semantic MLP, and semantic-similarity policies all have non-positive held-out
utility, while oracle VOI retains about five accuracy points of headroom.
The full 2,500-state semantic experiment increases label support and prevents
epoch-1 collapse, but structured success-difference utility remains negative at
-0.00264 with a confidence interval crossing zero. More data alone has not
closed the scientific gap.

The next work is therefore:

1. Use the now-complete 2,500-state ChartQA frozen protocol (320 positive and
   186 negative transitions) to increase transition-label support and quantify
   split variance.
2. Compare direct sparse gain regression with a structured objective that learns
   dense success-before and success-after probabilities, with transition
   weighting and within-state ranking.
3. Report human and augmented strata separately; the opportunity rate differs
   substantially between them.
4. Treat proposal coverage separately from policy quality: a matched nine-crop
   ablation increases oracle headroom but more than doubles exhaustive-search
   cost and worsens utility.
5. Add rescue-versus-harm calibration and repeated image-grouped split analysis.
6. Continue the compact state-level rescuability gate, which is positive on 4/5
   split point estimates but has no per-split utility interval excluding zero;
   treat it as a direction, not a solved result.
7. Enter VTool-R1/GRPO only if a deployable pre-action policy has positive
   held-out utility and improves the accuracy-cost frontier.

Update after independent validation: the factorized error-times-rescuability
stopping gate transfers with utility 0.00342 at 5.68% tool use, but its primary
state and image intervals miss the zero boundary by approximately `2.6e-5` and
`1.3e-4`, respectively. This is a failed primary near miss, not confirmation.
The same gate with fixed crop 0 has positive state and image intervals as a
pre-registered secondary, although direct paired contrasts do not show that the
fixed action is better than uniform random. A 4,500-state, 4,500-image balanced
replication is frozen with unchanged model and criterion. Until it completes,
the correct status is: stopping promising but not independently confirmed;
action selection unresolved; VTool-R1 integration gated.

The chart-layout proposal follow-up is now closed by its pre-registered no-go:
the image-disjoint treatment-minus-UG point estimate is +0.00175, but both state
and image intervals cross zero. The conditional 4,500-image chart-layout run is
not launched. Future action-selection work needs a new hypothesis and untouched
target rather than additional tuning on these ChartQA outcomes.

The default helpful-assistant ChartQA prompt is retained only as a protocol
negative control because verbose generations hit the token cap and polluted
entropy. No synthetic result or prompt-polluted result should support the method
claim.
