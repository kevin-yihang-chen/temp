# Review remediation status

This document maps the external code review to version 0.2 changes. It separates
resolved implementation issues from work that requires a real VLM environment.

| Review item | v0.2 status | Implementation |
|---|---|---|
| Proposal can see target | Resolved | `ProposalFunction` accepts only `AgentState`; `GroundTruth` is scorer-only |
| No image-question-region model | Architecture implemented; real adapter pending | `SemanticGainHead` and ROI pooling consume frozen semantic features |
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

The semantic head is a genuine trainable module, but no Qwen spatial tokens or
question embeddings have been extracted in this environment. The following are
therefore intentionally still pending:

1. Pin UG/lmms-eval and Qwen revisions in a dedicated CUDA environment.
2. Implement the adapter that returns spatial visual tokens, question embeddings,
   baseline state signals, answers, and entropy.
3. Generate image-grouped sibling rollouts on a small V*Bench subset.
4. Train the semantic gain head and compare it with scalar ridge, entropy search,
   tuned entropy stopping, and oracle selection.
5. Enter VTool-R1/GRPO only if the held-out accuracy-cost frontier improves.

No synthetic result should be used as evidence that the semantic head works on a
real VLM.
