# Visual-acquisition proxy-to-outcome audit protocol v1

Status: frozen on 2026-08-31 after the sole ScreenQA semantic candidate failed,
and before any teacher-forced target-answer likelihood was computed for the
ScreenQA sibling bank.  This protocol is a retrospective diagnostic on opened
ranker-training data, not a renewed candidate-development branch.

## Motivation and literature boundary

GapSight (arXiv:2608.21762, released 2026-08-22) mines candidate-crop labels from
the target model's answer-NLL or option-margin improvement and trains a one-shot
free-form router.  The Illusion of Visual Tool-Use (arXiv:2608.06270) uses
counterfactual returned observations to isolate step-level Visual Evidence Gain.
These works invalidate a broad claim that model-specific loss differences or
counterfactual visual evidence are new by themselves.

The remaining question here is narrower: when every fixed same-state action is
executed, how faithfully do common proxy improvements rank the signed final task
effect, induced harm, and cost-adjusted utility?  The answer can inform a
measurement/risk paper, but it cannot erase the failed ScreenQA candidate.

Primary sources:

- https://arxiv.org/abs/2608.21762
- https://arxiv.org/abs/2608.06270
- https://arxiv.org/abs/2602.01334

## Frozen first audit population

- ScreenQA opened ranker-training manifest:
  `artifacts/screenqa-train-factorized-v1/ranker-manifest-v1/manifest.jsonl`.
- Manifest SHA-256:
  `a2b6941e2a073b24571d2ccb50960f7c1cd70cb0ce53dc8339c7ec44a47f67ec`.
- Opened sibling rollouts:
  `artifacts/screenqa-train-factorized-v1/ranker-rollouts-v1/merged/rollouts.jsonl`.
- Rollout SHA-256:
  `0437d2a499adccb1b4e19eb0160583789cee00edf244718ecae9e290108bb8c9`.
- Exactly 14,511 decisions, 72,555 records, 1,510 source groups, one ANSWER and
  four frozen UG-grid ZOOM siblings per decision.
- The ScreenQA calibration, formal, reserve, untouched, validation, and test
  roles remain sealed and are forbidden inputs.

## Frozen answer-likelihood measurement

- Model: `Qwen/Qwen2.5-VL-3B-Instruct` revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Input contract: the exact stored model prompt; ORIGINAL alone for ANSWER;
  ORIGINAL plus the stored additive resized crop for ZOOM; system prompt
  `You are a helpful assistant.`; bfloat16; SDPA; min/max pixels
  `200704/602112`.
- Target span rule:
  `normalized_mode_then_shortest_then_first_index_v1`.  Normalize whitespace,
  select the most frequent case-insensitive accepted answer, then break ties by
  fewer whitespace tokens, fewer characters, and original manifest order.
- Score: mean teacher-forced negative log-likelihood over only the selected
  answer tokens.  Do not score the prompt or an end-of-turn token.
- Raw target text must not be written to score artifacts; write its SHA-256,
  selection index/votes/count, NLL sum/mean, and token count.
- Four deterministic state-aligned shards are allowed.  Atomic checkpoints may
  contain only complete five-sibling decisions and must support exact-prefix
  resume.

## Frozen diagnostic endpoints

For every ZOOM action define:

- task gain: `correct_after - correct_before`;
- entropy proxy: `entropy_before - entropy_after`;
- answer-loss proxy: `NLL_answer-now - NLL_zoom`;
- utility: `task gain - 0.05 * tool_cost`.

Report, without candidate selection:

1. Pearson and rank correlation of each proxy with signed task gain;
2. top-one crop task gain, rescue within helpful states, induced harm, and
   regret for answer loss, entropy, random, and oracle selection;
3. proxy-selected utility, rescue, harm, unnecessary calls, and gain per call
   over a fixed descriptive call-rate grid `[0.005, 0.01, 0.02, 0.05, 0.10,
   0.25, 0.50, 1.0]`;
4. whole-source bootstrap intervals with 2,000 resamples and seed `20260831`;
5. disagreement cases where loss improves but task correctness falls, and where
   task correctness improves without a positive loss gap.

The first full run is descriptive and development-only.  Its thresholds are not
valid for ScreenQA deployment or formal evaluation.

## Decision boundary

- Never reopen ScreenQA ranker development, calibration, or formal from this
  audit.
- If answer-loss is strongly aligned, the result motivates a newly
  preregistered method trained on a different development population and tested
  on a genuinely untouched population.
- If answer-loss is weak or harmful, reposition around proxy failure and
  prospective harm control, then require breadth across additional already
  opened development banks and at least one new untouched confirmation before a
  main-conference claim.
- No result from this audit may be described as independent ScreenQA validation.
