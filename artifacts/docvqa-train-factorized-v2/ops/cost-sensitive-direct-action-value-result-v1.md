# Cost-sensitive direct action-value result v1

Status: completed on 2026-09-01 under the frozen opened-DocVQA development
protocol. The mechanical decision is
**`cost_sensitive_direct_action_value_not_advanced`**. This is the first
candidate in the current branch whose source-balanced utility point estimate
exceeds the incumbent, but it does not satisfy the registered margin, interval,
or harm rule. ScreenQA and every protected role remain sealed.

## Bound execution

- Slurm job `199940`, one NVIDIA H800, 12 CPUs, 96 GiB, completed in
  `00:00:59`, exit `0:0`, zero restarts, with all-state email enabled.
- Code revision:
  `1a79ce9ffa9db8476d342e66236c60a12cadea02`.
- 3,500 sources, 13,580 decisions, 54,320 action rows, 1,442 positive and
  52,878 negative net-utility labels.
- Five whole-source folds, 20,000 whole-source bootstrap resamples, seed
  `20260914`.
- Report SHA-256:
  `941181e02e48352ea4f10ca20f9b10b3ed85afa790ba7d25b2138c0be984c464`.
- Score-report SHA-256:
  `5ae3623879a96c46ac5ffce40b4acc0b84b53e30c58dc2fe2d3f8006fbedf83d`.
- Outcome-free score rows SHA-256:
  `9512d000ca3c2567fd36711f20eca10619acd390622ccf1f05cc9930145dcaec`.
- Model SHA-256:
  `762005d1124ea68ca993f5a58e56317bdec0e026f24751504b8d5e295c3e6bb1`.
- Completion SHA-256:
  `f77068b07c70d2f32fc06ab6d9a4ca6a13df3dc7df35be83147d0dfacb533dfc`.

Every source has equal training mass; within-source mass is proportional to
absolute net utility. All five heads converge in seven iterations. Population,
utility counts, semantic alignment, feature dimension, fold exclusion,
source/global weight mass, matched-call, incumbent-reproduction, finite-score,
and no-leakage audits pass.

## Registered result

Both methods execute exactly 225 pooled calls. Source-balanced metrics are:

| Metric | Incumbent | Cost-sensitive direct value |
|---|---:|---:|
| Utility | 0.00317324 | 0.00330033 |
| Raw gain | 0.00397661 | 0.00413167 |
| Gain per call | 0.247496 | 0.248495 |
| Helpful-call precision | 0.394407 | 0.487057 |
| Induced harm | 0.000341563 | 0.000591115 |
| Negative-value call mass | 0.00977110 | 0.00880239 |
| Helpful-state proposal recovery | 0.516410 | 0.701761 |

The candidate-minus-incumbent utility difference is `+0.000127093`, with 95%
interval `[-0.000757203, 0.000864439]`. Gain per call and helpful-call
precision pass, and negative-value call mass falls. Advancement still fails
because:

1. `+0.000127093` is below the required `+0.00025` margin;
2. the paired lower endpoint is below `-0.0005`; and
3. induced harm is above the incumbent despite lower total negative-call mass.

## Fixed-result call/action decomposition

A deterministic 20,000-source-bootstrap decomposition with seed `20260915`
changed no fitted score, action, or threshold. It is descriptive only.

- The two 225-call sets share 146 decisions; each has 79 exclusive decisions.
- Candidate call set plus incumbent actions yields utility `0.00319727`,
  difference `+0.0000240317`, interval
  `[-0.000846839, 0.000738688]`, and induced harm `0.000591220`.
- Incumbent call set plus candidate actions yields utility `0.00310318`,
  difference `-0.0000700529`, interval
  `[-0.000191323, 0.0000105820]`, and induced harm `0.000351087`.
- Candidate call set plus an oracle choosing between the candidate and
  incumbent action yields utility `0.00334605`, only `0.0000457143` above the
  realized candidate, while retaining induced harm `0.000591115`.
- Incumbent call set plus the same two-action oracle improves over the incumbent
  by only `0.0000105820`.

The candidate action is already close to the two-action oracle on its selected
states. The excess harm follows the candidate call set even when the incumbent
action is substituted. The next change must therefore be a conservative
stopping/risk composition, not another crop proposer or action-value target.

## Frozen next direction

Do not tune C, utility weights, cost, feature mode, call budget, or action model
on this result. The next candidate may compose the two already serialized,
outcome-free OOF stopping scores without fitting:

1. convert incumbent and cost-sensitive scores to deterministic global
   empirical percentile ranks independently;
2. define consensus confidence as the minimum of the two percentile ranks;
3. retain the cost-sensitive selected action;
4. match exactly 225 calls with complete ties preserved; and
5. retain the same utility, interval, precision, and harm advancement rule.

This canonical minimum-rank rule represents agreement-based risk control and
has no tunable mixing weight. It must be frozen before evaluation. If it fails,
do not try alternative means, weights, or rank aggregations on this opened
population; reposition the paper around the mechanism and failure evidence or
move method development to a fresh population.

No GitHub push is authorized by this result.
