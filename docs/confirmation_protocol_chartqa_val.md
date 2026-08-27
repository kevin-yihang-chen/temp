# Independent ChartQA validation confirmation protocol

## Status and freeze point

This protocol is frozen before any Qwen rollout is collected on the target
validation states. The development data are the 2,500 ChartQA test states. The
confirmation data are the public `HuggingFaceM4/ChartQA` validation split at
revision `b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5`.

Two validation questions whose image hash also occurred in the development test
set are excluded before rollout collection. The frozen target therefore contains
1,918 states (958 human, 960 augmented), 1,054 unique images, and zero image or
state overlap with development. Its manifest SHA-256 is
`d3178218853b10447228963e839716f0eac768b51bdc0f5b4a83268d3819b58b`.

## Frozen rollout protocol

- Qwen2.5-VL-3B-Instruct revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- One answer-now action and four UG-grid crop siblings per state.
- Original image retained alongside each crop.
- Deterministic generation seed 0 and `max_new_tokens=16`.
- Concise final-answer-only system prompt used by the development protocol.
- ChartQA task scorer and state-cluster bootstrap.

## Frozen primary policy

The primary learned policy is the context-only factorized gate introduced in
code revision `e7626605ff491624a49694ea7f5a28f43760a129`:

1. estimate whether the baseline answer is wrong;
2. among development states whose baseline is wrong, estimate whether any crop
   can rescue the answer;
3. multiply the two probabilities;
4. apply the regularization and absolute call threshold selected only on an
   image-grouped development split;
5. if the gate fires, execute one uniform random crop. Evaluation uses the exact
   mean of the four frozen sibling outcomes to remove arbitrary action-seed
   variance.

The primary cost coefficient remains `lambda=0.05`. Target labels may be used
only after the policy, scaler, regularization, and threshold have been frozen on
development data.

The serialized source-only deployment model was frozen before confirmation
rollout completion. Its SHA-256 is
`5d5c0f781a7141726e786d6ad87b861a6395c489bcf9ad8567a2e9ca825c3330`;
it uses error `C=0.001`, conditional-rescue `C=0.1`, seed 17, and absolute
threshold `0.45069723964195885`. The paired source report SHA-256 is
`1f05ddeef52fa9abced549479cdb8fa386578d12600fb874a964a12a4d927462`.

## Primary criterion

The confirmation succeeds only if the transferred factorized policy has:

- positive mean utility at `lambda=0.05`;
- a 95% state-bootstrap utility interval whose lower endpoint is above zero;
- positive accuracy gain; and
- lower tool use than unconditional one-crop and exhaustive entropy policies.

The source-tuned entropy gate, unconditional uniform random crop, exhaustive
four-crop entropy search, answer-now, and oracle VOI are reported under the same
target rollout table. A positive point estimate with an interval crossing zero
does not pass.

An exploratory secondary policy was frozen before target rollout completion and
without reading target outcomes. It retains the identical primary state gate but
uses a source-only question-type-by-quadrant logistic ranker to choose one crop.
It cannot replace or change the primary criterion. The secondary action-model
SHA-256 is
`5989974482785b31868473e7a925708d15f6f1fbac3095906ded7a88def53bbd`,
and its source-only fit report SHA-256 is
`c0510901bc351ea9bac799497775ff53f7bb42b23ad574b555a7995c9922f35c`.
The implementation was frozen at code revision
`92de08347c4cb0c1e066e54365297f19df24a115`.

## Secondary analyses

- Human and augmented validation strata are reported separately.
- Helpful, harmful, and transition counts are reported.
- A cost frontier may be reported, but it cannot replace the registered
  `lambda=0.05` primary result.
- Unlabeled target-quantile calibration is exploratory and is not the primary
  confirmation policy.
- No semantic feature, action ranker, threshold, prompt, or candidate-set change
  is permitted after target outcomes are inspected.
