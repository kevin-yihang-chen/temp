# Related-work positioning update — 2026-08-28

## Why the positioning must change

The visual-tool literature moved quickly after the original project note. A
credible CVPR/ICCV/ECCV submission can no longer present learned cropping,
uncertainty-guided visual search, tool-use RL, or reducible agent uncertainty
alone as the contribution. The paper must isolate a narrower technical object:

> **the pre-action, cost-sensitive value of a visual acquisition relative to
> answering now, supervised by paired sibling counterfactual outcomes and used
> for both stopping and local visual-action credit.**

## Closest work and remaining gap

| Work | What it already establishes | Gap retained by this project |
|---|---|---|
| [UG / Training-free Uncertainty Guidance](https://arxiv.org/abs/2510.00705) | Scores acquired visual candidates using response uncertainty for search, frame sampling, and grounding. | Requires candidate observations and uses uncertainty rather than paired task-success improvement; it does not learn a cost-sensitive answer-now decision. |
| [VTool-R1](https://arxiv.org/abs/2505.19255) | Trains interleaved visual tool use with final outcome reward and no process supervision. | Sparse trajectory reward does not identify whether a particular visual action helped, harmed, or was unnecessary. |
| [CropVLM](https://arxiv.org/abs/2511.19820) | Trains an external crop model with RL and objective rewards such as correct-answer log likelihood; transfers the cropper across target VLMs. | Strong overlap with learned where-to-look. The remaining distinction must be the paired improvement over answer-now, explicit no-call action and cost, not merely rewarding an informative crop. |
| [MED: What Does Vision Tool-Use RL Really Learn?](https://arxiv.org/abs/2602.01334) | Decomposes tool-induced performance into gain and harm and reports that current tool-use RL mainly reduces harm while making limited progress on correcting intrinsic failures. | MED is a checkpoint-level measurement and diagnosis framework. This project must predict those gain/harm outcomes before acting and turn them into training credit. |
| [PriVE-Bench / PriVE-Tools](https://arxiv.org/abs/2607.16311) | Evaluates whether added boxes, crops, zoom panels, and contours overcome visual-prior failures; extra evidence is not universally beneficial. | Controlled evaluation rather than a learned cost-aware acquisition policy. Its findings strengthen the need for selective action value. |
| [Uncertainty Quantification in LLM Agents](https://arxiv.org/abs/2602.05073) | Provides a general agent-UQ formulation and a reducible-uncertainty perspective. | The phrase and general concept are no longer a novelty claim. The contribution must be the visual counterfactual estimator, data construction, empirical transfer, and credit mechanism. |
| [AdaptVision](https://arxiv.org/abs/2512.03794) | Learns when and where to acquire high-resolution crop tokens with a decoupled, token-efficiency-aware RL objective. | Uses a deliberately low-resolution initial view and end-to-end RL rewards; it does not expose calibrated per-candidate answer-now-relative value in task units under arbitrary deployment cost. |
| [Learning to Focus and Precise Cropping](https://arxiv.org/abs/2603.27494) | Shows that crop calls can be formalistic, then forces crop reliance through an information gap and grounding loss. | It does not construct exhaustive same-state sibling outcomes or predict rescue versus induced harm before acquisition. |
| [Do Multimodal Agents Really Benefit from Tool Use?](https://arxiv.org/abs/2606.02357) | Finds that current tool agents often do not expand solved sets and separates format-only from result-only effects. | Evaluates released end-to-end agents; this project adds action-level counterfactual acquisition utility and prospective fixed-policy tests, but only if transfer succeeds. |

Other visual-tool RL systems—including DeepEyes, Chain-of-Focus and
resource-constrained zoom-tool GRPO—make “VLMs can learn to call visual tools”
an established premise rather than a paper contribution.

## Claims that remain defensible

Subject to successful experiments, the following claims remain distinct:

1. **Answer-now-relative target.** Label each acquisition with its paired
   success change relative to the same state's no-tool sibling, not its absolute
   post-action correctness or confidence.
2. **Pre-action stopping.** Predict whether any acquisition has positive net
   value before paying for candidate observations.
3. **Cost-aware utility.** Evaluate and train on success gain minus visual cost,
   with accuracy–cost frontiers rather than accuracy alone.
4. **Localized visual credit.** Assign sibling-derived advantage to visual
   action tokens while retaining trajectory outcome advantage for reasoning and
   answer tokens.
5. **Harm avoidance as a first-class outcome.** Model helpful, neutral, and
   harmful acquisitions rather than treating every answer-preserving crop as a
   positive training example.

## Claims to avoid

- “First uncertainty-guided visual search.”
- “First VLM that learns to crop or zoom.”
- “First reducible uncertainty for agents.”
- "First counterfactual analysis of visual-tool gain and harm."
- "First evidence that multimodal tool calls are often unnecessary or
  perfunctory."
- A where-to-look claim based only on the current UG-grid or chart-layout
  proposer; the project's own image-disjoint confirmation did not show a
  spatial-selection advantage.

## Minimum experimental separation from the nearest work

A top-tier experiment should compare, under matched model and visual-compute
budgets:

- entropy-based acquired-candidate search (UG-style);
- absolute post-crop correct-answer reward or likelihood (CropVLM-style);
- trajectory outcome-only tool RL (VTool-R1/DeepEyes-style);
- answer-now-relative counterfactual value without cost;
- the full answer-now-relative, cost-sensitive action value; and
- oracle sibling value.

The key endpoints are not just final accuracy. Report tool-call rate,
call-induced harm, correction of answer-now failures, marginal gain per call,
utility over a cost frontier, calibration, and transfer to held-out datasets and
model families.
