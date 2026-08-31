# DocVQA reserve ToolGate-style comparator result

Status: completed on 2026-09-01 under the frozen, outcome-sealed reserve
protocol. This result is a preregistered null/negative ablation. It cannot
change the earlier DocVQA formal decision or authorize threshold revision on
the opened reserve population.

## Bound population and execution

- 688 source groups, 2,585 decisions, and 12,925 ANSWER/ZOOM records from the
  immutable allocation suffix `[9506, 10194)`.
- Qwen2.5-VL-3B-Instruct, deterministic decoding, one ANSWER and four frozen
  UG-grid crops per decision.
- Four NVIDIA H800 GPUs for the completed rollout/feature pipeline.
- 20,000 paired whole-source percentile bootstrap resamples, seed `20260829`,
  two-sided 95% intervals.
- No reserve outcome entered proposal, feature extraction, model fitting,
  threshold selection, matched-budget selection, or policy-score files.

## Registered primary comparison

The shared crop proposer was the frozen factorized action-value top crop.
Policy A gated it with signed net action value; Policy B used the frozen
ToolGate-style conservative binary execute proxy. Their source-balanced
results were:

| Quantity | Policy A | Policy B, frozen |
|---|---:|---:|
| Call rate | 0.0165105 | 0.0134478 |
| Utility | 0.00284534 | 0.00246274 |
| Utility 95% interval | [-0.00015237, 0.00668478] | [-0.00034658, 0.00624375] |
| Task gain | 0.00367087 | 0.00313513 |
| Gain per call | 0.222336 | 0.233134 |
| Helpful-call precision | 0.438743 | 0.385547 |
| Induced harm | 0.00074118 | 0.00068061 |

The registered Policy-A-minus-Policy-B utility difference is
`0.0003826055`, with paired 95% interval
`[-0.0012353520, 0.0019681249]`. Its point estimate is positive, but the lower
endpoint is not. The registered conclusion is therefore
`supports_policy_a_over_policy_b=false`.

The outcome-blind, test-feature-only matched Policy-B threshold is secondary.
It called on 50 decisions and obtained source-balanced utility `0.00221686`;
it does not replace the 43-call frozen Policy-B primary.

## Mechanistic diagnosis

The two frozen gates disagree on only 0.781% of source-balanced decisions.
At least one helpful crop exists in 7.93% of questions, but the shared proposal
misses a helpful crop in 49.27% of those helpful states. Consequently this
comparison has little leverage to distinguish gate supervision: both gates
usually see the same unhelpful proposed action and usually abstain.

This result does not show that signed action-value supervision is worse than a
binary execute proxy. It shows that the registered reserve population is
insufficient to establish superiority, and that action proposal is now the
larger observed bottleneck than the marginal difference between these two
sparse gates.

## Reproducibility

- freeze SHA-256:
  `eebc6ee4c3b1affcfc97b1c953094722001838166c7c90c4bcccc88821c7315e`;
- rollout SHA-256:
  `60ac67d8154a7941f271b4034fdd8681027cd8b702074556c009f11d6aee3925`;
- outcome-free policy-score SHA-256:
  `0cbd5ed1cbeef52316abcff422e849365ca944c3b206c0692756525947069f93`;
- score-report SHA-256:
  `d68405c95dc75fbb62750ff8110abd74f05c3f62397dfbd4eb8145de20194e5e`;
- evaluation-report SHA-256:
  `c3c5103db85246c85da3fd6740194726fb7871279256acd2942369e6c7812315`.

An independent evaluator replay with the same hashes, seed, and 20,000
resamples produced an evaluation report byte-identical to the registered
artifact. A separate exact-schema audit reproduced 42 Policy-A calls, 43
frozen Policy-B calls, and 50 secondary matched calls across all 2,585 rows.

## Frozen next direction

Do not retune either gate, threshold, feature family, or crop proposer on this
opened reserve bank. The next candidate must be developed entirely on
non-ScreenQA opened populations and target proposal quality directly, using:

1. signed realized crop gain as the primary action target;
2. target-answer loss gap only as training-time auxiliary supervision;
3. an explicit induced-harm head or constraint;
4. pre-action state, image, question, and candidate geometry only at inference;
5. source-disjoint OOF fitting and a fixed call-budget/risk sequence; and
6. prospective ScreenQA risk calibration followed by one-shot formal testing,
   while its allocated calibration/formal outcomes remain sealed.

The completed Qwen2.5-VL-7B H800 mechanism replication can support the proxy
motivation, but it is not a substitute for this independent policy test.

No GitHub push is authorized by this result.
