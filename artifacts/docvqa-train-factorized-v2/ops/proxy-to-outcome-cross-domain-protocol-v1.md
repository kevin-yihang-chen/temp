# DocVQA proxy-to-outcome cross-domain replication protocol v1

Status: frozen on 2026-08-31 after completion of the opened-bank ScreenQA
proxy-to-outcome audit and before any teacher-forced target-answer likelihood
was computed for the DocVQA ranker-development sibling bank.

This is a retrospective replication on an already opened development role. The
DocVQA risk-calibration and formal outcomes already exist from the earlier
factorized-v2 branch, but they are forbidden inputs to this audit. This result
cannot be described as independent DocVQA validation and cannot revise the
failed DocVQA one-shot formal decision.

## Motivation and fixed question

The ScreenQA audit found that target-answer loss gap was more aligned with
realized signed task gain than entropy reduction: action-level Spearman
`0.2035` versus `0.0698`, answer-loss top-one task gain `0.01730` versus
`-0.00179`, and a positive source-bootstrap utility lower endpoint at sparse
descriptive call rates. Those observations were made before this replication
protocol and motivate, but do not determine, the DocVQA result.

The fixed question is whether the same proxy hierarchy and sparse-selection
signal replicate across the document-image domain under the identical target
model, answer-likelihood definition, crop family, cost, and outcome estimands.

Answer-loss or counterfactual crop supervision is not novel by itself. A
positive replication may motivate privileged-supervision distillation into a
pre-action signed-value and harm model; it is not itself a new deployable
method.

## Frozen opened population

- Manifest:
  `data/docvqa-train-factorized-v2/ranker-training/manifest.jsonl`.
- Manifest SHA-256:
  `871ea5b924badba8e0f23477fbd40d2e77085a5f69726d0c48e674e72d64a25d`.
- Sibling rollouts:
  `artifacts/docvqa-train-factorized-v2/ranker-training/qwen3b-c4-seed0/rollouts.jsonl`.
- Rollout SHA-256:
  `9109d5c896e59959937c2171e897dd81341237ff51fc5745fb4c56325ddd8fc3`.
- Rollout provenance SHA-256:
  `f672eb6c12a093825541bf84ae25b3321adf1b32a3756e86401f8a324faf3699`.
- Rollout audit SHA-256:
  `c9d196230a6381913f18f1b026bb723e59de999dbdffe6a96b9e1a66a0816c0e`.
- Exactly `13,580` decisions, `67,900` records, `3,500` whole-source
  groups, one ANSWER sibling, and four frozen UG-grid ZOOM siblings per
  decision.
- No risk-calibration, formal-test, reserve-comparator, ScreenQA protected-role,
  validation, or test record may be read by this audit.

## Frozen answer-likelihood measurement

- Model: `Qwen/Qwen2.5-VL-3B-Instruct` revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Use the exact stored model prompt; ORIGINAL alone for ANSWER; ORIGINAL plus
  the stored additive resized crop for ZOOM; system prompt
  `You are a helpful assistant.`; bfloat16; SDPA; min/max pixels
  `200704/602112`.
- Target rule: `normalized_mode_then_shortest_then_first_index_v1`. Normalize
  whitespace, select the most frequent case-insensitive accepted answer, then
  break ties by fewer whitespace tokens, fewer characters, and manifest order.
- Score only the selected answer tokens with mean teacher-forced negative log
  likelihood. Do not score the prompt or an end-of-turn token.
- Never write raw target text. Write only the target SHA-256, selection
  index/votes/count, NLL sum/mean, and token count.
- Use four deterministic state-aligned shards with atomic complete-decision
  checkpoints and exact-prefix resume.
- Use four RTX 4090 GPUs for the full score bank. This matches the accelerator
  class used for the sibling-bank pipeline and avoids the hardware-dependent
  loss ranking observed in the prior H800/4090 engineering audit. A four-hour
  wall limit is allowed; no mixed-hardware shard is allowed.

## Frozen endpoints

For every ZOOM action define:

- task gain: `correct_after - correct_before`;
- entropy proxy: `entropy_before - entropy_after`;
- answer-loss proxy: `NLL_answer-now - NLL_zoom`;
- utility: `task gain - 0.05 * tool_cost`.

Report without candidate selection:

1. Pearson and Spearman correlation of each proxy with signed task gain;
2. top-one crop task gain, utility, helpful-state rescue, induced harm, and
   oracle regret for answer loss, entropy, exact uniform random, and oracle;
3. utility, gain per call, rescue, harm, and unnecessary calls over the fixed
   call-rate grid `[0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0]`;
4. 2,000 iid whole-source percentile bootstrap resamples with seed `20260901`
   and two-sided 95% intervals;
5. action-level loss/task disagreement counts and identifier-only examples.

All grid thresholds are descriptive on opened development data and are invalid
for deployment, ScreenQA calibration, or any formal evaluation.

## Preregistered replication decision

Declare **replicated alignment** only if all conditions hold:

1. the 95% lower endpoint of answer-loss Spearman correlation is greater than
   zero;
2. the 95% lower endpoint of answer-loss top-one mean task gain is greater than
   zero;
3. the answer-loss top-one task-gain point estimate exceeds both entropy and
   exact-uniform-random point estimates;
4. at least one fixed call-rate point in `[0.005, 0.01, 0.02, 0.05, 0.10]` has
   a strictly positive 95% lower endpoint for answer-loss mean policy utility;
5. the answer-loss top-one induced-harm point estimate is lower than both
   entropy and exact-uniform-random induced-harm point estimates.

Declare **partial alignment** if conditions 1 and 2 hold but any of conditions
3 through 5 fail. Otherwise declare **non-replication**.

Only replicated alignment authorizes writing a new, separately preregistered
DocVQA-development to ScreenQA-untouched surrogate protocol. Partial alignment
or non-replication closes the method-transfer branch and supports only broader
proxy-failure and harm diagnostics. No outcome here may reopen or retune the
old DocVQA or ScreenQA candidate branches.
