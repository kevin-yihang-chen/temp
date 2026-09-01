# InfographicVQA relative-where source-OOF protocol

Status: frozen after the job-203078 oracle factorization and design audit, and
before fitting any model in this protocol or computing any new ranker endpoint.
This is an official-train-only confirmatory branch. Validation and test remain
sealed.

## Bound inputs

```text
9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e  merged rollouts
884de4ebc1ba83226393871cdfff819bc7553a61eee6ffa11697db03332ac646  answer-NLL teacher rows
d0508726a50b4c1e54778392d08329b242a680fc13292cac1ebec8b42a175300  label-free semantic features
0916a6b5a32e15c4f5b3bf920e1ecd4f304aeb97ae186e0e3e846391e2304203  image manifest
7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6a232b0837a  outer source folds
8b4977776841794185eedd8cbbfdb8e23ec2c44cc0be9931cc2e672b232f344c  inner source folds, audit-only and unused for fitting
c8338b1960ca223c892c3f992b0ccc3027d543113f67c82a734eccd7e3699c4b  frozen DECAR OOF predictions
ab4a179c2141de60c9a1d173e34a7166d075935e5283569439e3d93424344a62  frozen entropy/OOF-where hybrid evaluation
6ef0869b453e1a70ad5f479e8a9604aa04ac0419a1dc92d1ce353c54f66f3025  oracle-where factorization evaluation
17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6  formal bootstrap indices
5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0  formal bootstrap source order
c520dc3fd9a5d25c0b7e626a88a55e629e56c8a6d1a473feb53d94bad4689cf0  oracle-where result record
c35f685b5b83af00cf2132cdcf8083e46566e3696bd82ec0a8ad4fe0adc0550b  relative-where design audit
```

Every hash must be checked before fit. The population is fixed at 23,946
decisions, 4,406 images, and 2,204 sources with one ANSWER-NOW and four ordered
UG crops per decision. No validation/test path, target answer text, OCR label,
answer type, post-crop field, outcome, NLL target, source ID, or fold ID is an
inference feature.

## Fixed source-OOF procedure

Use the existing five source-disjoint outer folds. For each variant and outer
fold, fit once on the other four folds and predict every decision in the held-
out sources. No inner selection, checkpoint selection, early stopping,
hyperparameter search, retry on scientific performance, or refit on an outer
test source is allowed. The output is exactly one source-held-out row per
decision. Twenty fits are expected: five folds times four variants.

The prediction row may contain identifiers, outer fold, four finite predicted
scores/probabilities, selected action, and top-one/top-two margin. It must not
contain teacher gaps, targets, task deltas, correctness, answers, post-action
entropy, utility, or any other outcome-derived field.

## Shared feature standardization and optimizer

Use the already-materialized original-image-only features: 3,584-dimensional
question, global-image, and four ROI vectors plus the 16 registered
candidate/baseline scalars. Fit all per-dimension means and scales only on the
outer-training sources with source-balanced decision weights. Constant scales
become one.

Project question, global, and ROI vectors separately from 3,584 to 64
dimensions. All networks are float32, deterministic, full-batch AdamW for 200
fixed epochs, learning rate 0.001, weight decay 0.0001, no dropout or
normalization layers. The MLP is `input -> 128 -> 32 -> 1` with GELU after the
first two layers. Use seed `20260923 + 100*outer_fold + variant_index`.

## Primary and fixed ablations

The sole primary is `relative_teacher_entropy`.

For a candidate, let `q`, `g`, and `r` be its projected question, global, and
ROI vectors; let `r_bar` be the mean of the four projected ROI vectors; let `s`
be its standardized 16-vector and `s_bar` the four-candidate scalar mean. The
544-dimensional relative fusion is

```text
[q, g, r, r-r_bar, q*r, q*(r-r_bar), g*r, g*(r-r_bar), s, s-s_bar].
```

The fixed variants are:

1. `relative_teacher_entropy` (primary): relative fusion, privileged
   answer-NLL-gap ranking target, entropy-tail weighting;
2. `absolute_teacher_entropy`: 464-dimensional absolute fusion
   `[q,g,r,q*r,q*g,g*r,r-g,s]`, same target and weighting;
3. `relative_teacher_uniform`: primary architecture and target without the
   entropy-tail multiplier;
4. `relative_task_entropy`: primary architecture and weighting with realized
   task delta as its training-only ranking target.

No variant can replace the primary after seeing outer endpoints.

## Fixed relational ranking loss

For each outer-training decision, form a target distribution uniform over the
candidate or candidates with the exact maximum target. Let target range be
`max(target)-min(target)`. Compute its source-balanced mean over positive-range
training decisions, `range_scale`, and multiply the decision weight by
`range/(range+range_scale)`; zero-range task rows receive zero weight.

For entropy-weighted variants, compute the training-fold empirical percentile
of answer-now entropy using average ranks for exact ties, mapping the minimum
to zero and maximum to one. Multiply by `1 + 4*percentile`. The uniform variant
uses multiplier one. Normalize final decision weights to sum one.

The loss is weighted listwise cross entropy to the maximum-target distribution
plus 0.5 times weighted pairwise logistic loss. Pairwise terms use every
unequal target pair and are normalized within a decision by absolute target
difference. This loss is fixed; no temperature, label smoothing, focal
parameter, or outcome-dependent threshold is allowed.

## Frozen evaluation

After prediction and audit hashes are final, reuse the exact entropy-selected
identities, rates, actual calls (120, 240, 479, 1,198, 2,395), cost 0.05, and
formal `int32 [20000,2204]` paired whole-source bootstrap. Evaluate:

- the four new OOF variants;
- frozen `entropy_when_decar_where` and
  `entropy_when_task_value_where`;
- one-crop entropy-random and fixed-grid-00;
- ANSWER-NOW;
- privileged raw teacher-NLL argmax and task-oracle argmax, clearly labeled
  non-deployable ceilings.

Report question- and source-balanced ANLS gain, utility, call/crop rate,
helpful precision, induced harm, negative-utility calls, action regret, teacher
agreement, all 95% intervals, and paired source-utility differences. For every
bootstrap draw also report primary oracle-gap closure

```text
(U_primary - U_old_DECAR) / (U_task_oracle - U_old_DECAR)
```

when its denominator is strictly positive.

## Confirmatory train gate

An operating point supports the primary only if all conditions hold:

1. at least 100 calls and 50 called sources;
2. primary source-utility 95% lower endpoint is strictly positive;
3. paired primary-minus-old-DECAR and primary-minus-old-task-value utility 95%
   lower endpoints are both strictly positive;
4. primary point utility is strictly above entropy-random, fixed-grid-00, and
   all three new ablations;
5. primary induced harm and negative-utility-call mass are each no greater
   than both one-crop entropy-random and fixed-grid-00;
6. primary point oracle-gap closure is at least 0.50;
7. every hash, fold, source exclusion, coverage, leakage, identity, action,
   cost, bootstrap, arithmetic, and seal audit passes.

If multiple points qualify, select higher primary source utility, then lower
induced harm, then lower rate. Emit `relative_where_train_supported` only if a
point qualifies; otherwise emit `relative_where_train_not_supported`. A
positive train decision authorizes writing a separate frozen validation
protocol but does not itself read validation or test. An ablation cannot open
validation if the primary fails.

Write fit artifacts atomically under
`artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1/relative-where-oof-v1`
and evaluation artifacts under its `evaluation-v1` child. Every submitted task
must email `yihangc@connect.hku.hk` on all Slurm state changes. No GitHub push
is authorized.
