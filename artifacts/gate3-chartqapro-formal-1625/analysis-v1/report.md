# ChartQAPro untouched formal evaluation

> Frozen 1,625-question formal target; no target-derived tuning.

- Primary confirmation passed: **False**
- Empty outputs: 0
- Baseline max-token cap rate: 0.0086
- Raw constrained-format compliance: 0.8686
- Conservative canonical-parse compliance: 0.9984

## Frozen primary criterion

- positive_mean_utility: **False**
- question_bootstrap_utility_lower_above_zero: **False**
- image_bootstrap_utility_lower_above_zero: **False**
- positive_released_score_gain: **True**
- lower_tool_use_than_unconditional_one_crop: **True**
- lower_tool_use_than_exhaustive_four_crop: **True**
- passed: **False**

## Released scorer

| Policy | Score gain | Tool rate | Utility |
|---|---:|---:|---:|
| frozen_factorized_context | 0.0024 | 0.0929 | -0.0023 |
| frozen_source_entropy | 0.0049 | 0.3575 | -0.0130 |
| always_random | 0.0165 | 1.0000 | -0.0335 |
| exhaustive_entropy | 0.0259 | 1.0000 | -0.1741 |
| oracle | 0.0846 | 0.0966 | 0.0798 |

## Paper Spec scorer

| Policy | Score gain | Tool rate | Utility |
|---|---:|---:|---:|
| frozen_factorized_context | 0.0024 | 0.0929 | -0.0023 |
| frozen_source_entropy | 0.0049 | 0.3575 | -0.0130 |
| always_random | 0.0161 | 1.0000 | -0.0339 |
| exhaustive_entropy | 0.0259 | 1.0000 | -0.1741 |
| oracle | 0.0846 | 0.0966 | 0.0798 |

## Paper Spec Canonical scorer

| Policy | Score gain | Tool rate | Utility |
|---|---:|---:|---:|
| frozen_factorized_context | 0.0024 | 0.0929 | -0.0023 |
| frozen_source_entropy | 0.0049 | 0.3575 | -0.0130 |
| always_random | 0.0096 | 1.0000 | -0.0404 |
| exhaustive_entropy | 0.0216 | 1.0000 | -0.1784 |
| oracle | 0.0778 | 0.0898 | 0.0733 |
