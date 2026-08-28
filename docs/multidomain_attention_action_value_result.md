# Multidomain attention action-value development result

Date: 2026-08-28

## Scientific status

This is a development-only source-grouped OOF comparison. It jointly fits the
DocVQA and TextVQA development banks and does not read either formal bank. The
experiment tests whether question-conditioned regional evidence has a
directionally consistent cost-aware signal across domains; it is not a formal
confirmation and does not replace either frozen formal policy.

The pooled model uses one shared factorized error/rescue/harm estimator, a
shared no-call margin, and domain-balanced model selection. Source groups are
held out during all OOF decisions. Utility is task-score gain minus
`0.05 * tool calls`.

## Results

Intervals use 5,000 source-cluster bootstrap resamples over the pooled 400
development sources.

| Feature mode | Domain | Decisions | Gain | Tool rate | Utility |
|---|---|---:|---:|---:|---:|
| semantic-context | DocVQA | 824 | +0.00218 | 1.33% | +0.00151 |
| semantic-context | TextVQA | 318 | +0.00503 | 5.66% | +0.00220 |
| semantic-context | pooled | 1,142 | +0.00297 | 2.54% | +0.00170 |
| hybrid-context-semantic | DocVQA | 824 | +0.00218 | 1.09% | +0.00163 |
| hybrid-context-semantic | TextVQA | 318 | +0.00189 | 3.14% | +0.00031 |
| hybrid-context-semantic | pooled | 1,142 | +0.00210 | 1.66% | +0.00127 |

The semantic-context pooled utility has a 95% CI of
`[-0.000680, +0.004658]`; the hybrid interval is
`[-0.000654, +0.003704]`. Thus both domains have positive point estimates,
and the raw-gain intervals are positive, but neither cost-adjusted pooled
interval excludes zero. The semantic-context model is the stronger of the two:
it has higher pooled, domain-balanced, worst-domain, and TextVQA utility.

## Interpretation

This removes the earlier concern that the learned stopping rule works only
when fitted separately on DocVQA: one shared policy makes selective calls with
positive point-estimate utility on both development domains. However, its
evidence is still too sparse for a significance claim: only 29 of 1,142 OOF
states call the tool, and 79.3% of those calls are unnecessary under the
current binary helpfulness diagnostic. The main remaining problem is precise
stopping, not the existence of useful regional actions.

The correct use of this result is architecture selection and power planning
for a future untouched cross-domain split. It must not be used to reinterpret
the already inspected TextVQA formal result or to tune the frozen DocVQA
secondary analysis.

## Reproducibility

- training code revision: `52d14a3718883deb916346acdf9f1b188983d8bf`
- DocVQA development rollouts SHA-256:
  `4d3d3a33f644d1f5122aabecd47a8168d2dce2db5014692b508ba76ae4ddbe52`
- TextVQA development rollouts SHA-256:
  `a94c72b1977e86436c6187248f64826a34b791151c52a7c7b73ca89f92b97ddb`
- semantic-context report / model SHA-256:
  `bc7456862d565d66fdc3c081a45463497b900e70e946f35a13aacd3c897236d4` /
  `806d875a9e16bdf6ee2ae3b0b5e8b3afb7f13399d02e7276162367d4cb976173`
- semantic-context regularizer / margin: `alpha=100.0` /
  `0.13243717658829612`
- hybrid report / model SHA-256:
  `5938017e91ea3c17b85fa0ef66fde34343b087187a17ffef599685d45652a0fa` /
  `21cd5ffeef353e6c0d07c688e80f251ce4cd8ecf40483fe989267f0d978c133c`
- hybrid regularizer / margin: `alpha=100.0` /
  `0.1408784562504658`

