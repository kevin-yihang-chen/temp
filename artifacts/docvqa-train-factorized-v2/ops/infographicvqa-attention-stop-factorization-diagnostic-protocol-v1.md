# InfographicVQA attention stop-factorization diagnostic protocol v1

Date: 2026-09-02 (Asia/Hong_Kong)

Status: descriptive protocol written after the raw-attention train result was
opened. It cannot revise that negative decision or authorize calibration,
validation, or test.

## Question

The raw-attention action is paired-superior to every registered deployable
where comparator at the 5% and 10% entropy call budgets, but its end-to-end
utility remains negative. Quantify how much utility is recoverable by improving
only the stopping rule while holding the raw-attention action fixed.

## Bound inputs

```text
9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e  rollouts
009cdbfa1761f9b53b791a50f70b7e98bdf275eec8743d8bcaf078a52ded8ce8  raw-attention features
6eb313cf1bf4e5f61a8decc0c6ef70605009826c1bf4f815deb7f8111ec7bf40  feature completion
27ba5df9d45f9837f685d64589e32740238de6ff0ce46ce54ce6a1ac21a1d471  feature audit
5c8bced0fdad0a4f7c3ad0dca8bf8cf31d40be4c9d2318c6b42ea72d065366ee  raw-attention evaluation
ea38fb7adb024a1c96a6ec160d921687affb3ac0222aecba3f5d422728a4cbf5  raw-attention evaluation completion
17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6  bootstrap indices
5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0  bootstrap source order
```

Require 23,946 decisions, 2,204 whole-source groups, 4,406 images, feature
revision `2020b423f7daa6e8b9a942a02308137136bba548`, and model revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`.

## Diagnostic policies

Hold `argmax(question_region_attention)` fixed as the action for every state.
At the registered call budgets 0.5%, 1%, 2%, 5%, and 10%, report:

1. the original answer-entropy stopping set;
2. an outcome-free set ranked by maximum region-attention score;
3. an outcome-free set ranked by the top-two attention margin; and
4. a privileged ceiling that selects at most the budget's states with largest
   positive realized net value for the fixed action.

Also report the unrestricted positive-net fixed-action ceiling and the full
task-action positive-net ceiling. Reuse the frozen 20,000 paired whole-source
bootstrap samples. Report source- and question-balanced utility, gain, harm,
call mass, positive-net precision/recall, and paired utility differences from
entropy stopping.

## Interpretation boundary

Realized-utility policies are non-deployable ceilings. Max-score and margin
policies are post-hoc diagnostics on already opened train outcomes. No policy,
rate, or combination may be selected for a formal claim from this run.

The diagnostic may justify one separately frozen source-OOF stopping model
whose action selector is fixed before its folds are fit. Any such model must
use outcome-free inference features, whole-source exclusion, multiplicity-aware
evaluation, and a later independent calibration protocol.

All submitted execution states must email `yihangc@connect.hku.hk`. No GitHub
push is authorized.
