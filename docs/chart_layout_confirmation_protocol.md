# Chart-layout candidate confirmation protocol

## Status and separation

The chart-layout proposer was designed after inspecting a 200-state ChartQA
development slice. The confirmation target is frozen before any chart-layout
rollout is collected on it. Every image occurring in the 200-state development
manifest is excluded, including additional questions that share those images.

The target contains 2,137 states (1,046 human and 1,091 augmented) from 1,320
unique images. It has zero image overlap with the 189 development images. The
target manifest SHA-256 is
`d7c96df369259c8c3645bf64c27c220936636c92e359171a50e420344c5ff0bd`.
The complete 2,500-state source manifest SHA-256 is
`3c485aa5c09cc9491f866ba5737a78c2b79c3539c6de2663c964b2cff90d814a`,
and the excluded development manifest SHA-256 is
`f7e1616e3378f6c781ef166ebf78c8650cfcdd0d9d5f0d653a2f1ad4d573db17`.

## Frozen candidate protocols

Both sides use Qwen2.5-VL-3B-Instruct revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`, deterministic seed 0, the concise
answer-only prompt, four crops, additive original-plus-crop observations, and
the ChartQA scorer.

- Baseline: the existing four spatially balanced UG grid crops.
- Treatment: four square chart-layout crops at left-top, left-middle,
  center-middle, and right-middle.

The candidate count and per-crop visual cost remain matched. The treatment was
frozen at code revision `71b4a64099a8d0bb2323acf1e8f87852a825bd63`.

## Primary criterion

The primary estimand is treatment-minus-baseline accuracy gain (equivalently
utility difference at equal one-crop cost) for the exact uniform-random
one-crop policy. Confirmation succeeds only if:

- matched answer-now outputs are identical;
- the paired point estimate is positive;
- the 95% state-bootstrap interval has a lower endpoint above zero; and
- an image-cluster paired bootstrap sensitivity interval also has a lower
  endpoint above zero.

Fixed-center, exhaustive entropy search, oracle VOI, helpful/harmful state
rates, and transition counts are secondary. The 200-state development result
cannot be pooled into the confirmation estimate.
