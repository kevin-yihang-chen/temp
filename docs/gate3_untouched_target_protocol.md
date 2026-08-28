# Gate 3 untouched-target protocol

## Status

This protocol is registered before any project-model rollout or per-example
answer inspection on the target benchmark. Public dataset metadata, the paper,
the official evaluation code, and aggregate results reported by the benchmark
authors were inspected only to choose the target and define the scorer.

The purpose is deliberately narrow: test whether the already-frozen
**when-to-call** gate transfers to a harder, independently sourced chart QA
benchmark. It does not test a learned crop selector and it cannot establish a
where-to-look claim.

## Primary target and frozen revisions

The primary target is ChartQAPro:

- paper: 1,948 questions over 1,341 charts from 157 sources, including
  infographics and dashboards;
- dataset: `ahmed-masry/ChartQAPro` revision
  `e27c2874825874d6767d2bbc538ed4f0dc2c64c2`;
- official code: `vis-nlp/ChartQAPro` revision
  `4b422c658270aff1d3105fd0fb39b1dd5de9f08c`; and
- primary scorer: byte-semantic parity with the released official evaluator and
  the pinned VLMEvalKit adapter; and
- scorer sensitivity: the paper-specified exact-match rule for Fact Checking
  and Multi Choice.

The sensitivity is registered because both released implementations compute an
`always_use_exact_match` category flag but fail to pass it to the scoring
helper. Consequently their actual behavior applies ANLS unless a `Year` flag
forces exact matching. Released-code parity remains primary for benchmark
comparability; the paper-specified correction is always reported alongside it
and cannot replace the primary after outcomes are inspected.

The fixed Parquet also contains two structural `Year` anomalies: source index
1326 has two conversational answer turns and four flags, while index 1358 has
four answer turns and two flags. The exporter preserves these source values and
records the anomaly. Both frozen scorers follow released behavior and use only
the final flag for a Conversational row; no target answer is modified or
discarded by this compatibility rule.

The same pre-outcome structural audit finds empty final answer strings at
source indices 1358 and 1359. These two rows have no defined target under the
released scorer (only an empty model response could match), so they are
excluded before the image-group split. They are not rewritten as
`unanswerable`, and the audit records their indices and empty positions. No
other empty question, answer, or year field is permitted.

An independent scorer self-check finds 12 additional non-empty targets for
which at least one frozen scorer assigns the gold answer itself a score below
1.0. These include one non-finite numeric-like scalar and bracketed answers
affected by the released list parser. Before any model rollout, the exclusion
rule is therefore generalized: a target is eligible only if both the released
primary scorer and paper-spec sensitivity scorer assign its own gold answer
exactly 1.0. The exporter records source index, type, shape, answer length,
answer SHA-256, and both self-scores without rewriting the answer. The split is
computed only after this deterministic label-validity filter.

ChartQAPro is preferred over the public VTool `Refocus_Chart` test because the
latter has exact decoded-RGB-plus-question overlap with the project's ChartQA
development data. It is preferred as the primary over ChartMuseum because its
official scorer is deterministic and does not require an LLM judge.

ChartMuseum is reserved as a secondary robustness target at dataset revision
`462d46deb187d8a40c5a9de4e69e14f1df982e58` and code revision
`c3feaea144fecae71508add5570222dfc83ede6b`. Its result cannot replace the
ChartQAPro primary because the official evaluation uses an LLM judge.

## Pre-outcome identity audit

Before running the VLM, decode every target image to normalized RGB and compute
the existing image and normalized-question hashes. Audit against every image in
the ChartQA development, validation-confirmation, and train-replication
manifests, plus the pinned VTool test audit source.

- Exclude an entire target image group if its RGB hash occurs in any prior
  project split, even when the target question differs.
- Report exact image, question, and joint-key overlaps, duplicates, invalid
  images, and exclusions in a provenance artifact.
- Index, filename, source URL, and fuzzy text are not acceptable identity
  evidence.
- Abort rather than silently repair a malformed or ambiguous identity join.

This audit may read images and question strings but must not read project-model
predictions, correctness, or per-action target outcomes.

## Frozen pilot/formal split

After overlap exclusions, group all remaining rows by normalized RGB hash.
Rank groups by
`sha256("chartqapro-gate3-pilot-v1\0" + rgb_sha256)` and assign the first 200
image groups to the compatibility pilot. Every other image group is the formal
target. This makes the split deterministic, question-count agnostic, and
strictly image-disjoint.

The pilot may be used only to:

- verify image decoding, conversation serialization, and official scoring;
- make the final-answer format compatible with MCQ, conversational,
  hypothetical, and unanswerable questions; and
- measure runtime and memory for Slurm sizing.

It may not be used to refit the stopping model, scaler, regularization,
threshold, cost, crop geometry, or primary criterion. Any prompt or scorer
compatibility change must be frozen in a new provenance record before the first
formal-target rollout. Pilot outcomes are never pooled with the formal result.

## Frozen model, actions, and gate

- Qwen2.5-VL-3B-Instruct revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Deterministic seed 0, SDPA, and a prompt frozen after compatibility testing.
- One answer-now action plus the existing four UG-grid crops.
- Original image retained alongside every crop observation.
- Official ChartQAPro answer scorer, mapped to a per-question score in `[0, 1]`.
- Cost coefficient `lambda=0.05` and 5,000 paired bootstrap resamples.

The primary policy is the byte-frozen factorized context gate:

- model SHA-256
  `5d5c0f781a7141726e786d6ad87b861a6395c489bcf9ad8567a2e9ca825c3330`;
- absolute call threshold `0.45069723964195885`; and
- uniform one-crop expectation across the four frozen siblings when it calls.

The gate has only `ANSWER` and `CALL_VISUAL_TOOL` outputs. It must keep
`spatial_action_id=None`; using a learned refocus program or learned crop ranker
would change the estimand and is forbidden in this protocol.

## Primary estimand and criterion

For question `i`, define

`utility_i = score_i(policy) - score_i(answer_now) - 0.05 * call_i`.

The policy score for a call is the exact mean official score of the four crop
siblings. This removes arbitrary crop-seed variance while retaining the cost of
one deployed call. The primary result is the mean paired utility on the formal
target.

The untouched-target confirmation passes only if the transferred frozen gate
has all of the following:

- positive mean utility;
- a 95% question-bootstrap utility lower endpoint above zero;
- a 95% image-cluster-bootstrap utility lower endpoint above zero;
- positive mean official-score gain before cost; and
- lower tool use than unconditional one-crop and exhaustive four-crop policies.

Answer-only, unconditional uniform one-crop, exhaustive entropy search, the
source-frozen entropy gate, and oracle value are reported under the same formal
rollouts. Question-type strata, call calibration, transition counts, and cost
frontiers are secondary and cannot change the primary decision.

## Decision boundary

A pass would establish that the project's central stopping result transfers
beyond the ChartQA family split used to construct it, and would justify a
bounded VTool Stage A when-to-call evaluation. It still would not validate
spatial action selection or localized action-token credit.

A failure leaves the existing ChartQA high-power stopping replication intact
but limits the paper claim to in-family generalization. The formal target must
not be reused to tune a replacement primary. Representation or calibration
changes require a new development target and another untouched confirmation.

High-cost RL and any spatial-action advantage remain on hold until this
when-to-call transfer is resolved and a spatial selector independently beats
matched random or fixed crops.

## Completed target freeze

The pinned Parquet contains 1,948 rows and 1,252 unique decoded-RGB image
contents. Exact audits find zero image, normalized final-question, or joint-key
overlap with the ChartQA development, validation-confirmation, train-replication,
or VTool test sources. Fourteen malformed targets are removed by the rules
above, leaving 1,934 scorer-self-consistent questions over 1,250 images.

The deterministic split contains the same state and image identities in both
freeze versions. Version 2 changes only the data contract: `question` is the
core gate-visible task context, while `model_prompt` contains the full official
benchmark prompt shown only to the VLM. This prevents benchmark instructions
and long paragraph wrappers from becoming stopping-model features.

The current version-2 freeze contains:

- compatibility pilot: 309 questions over exactly 200 images, manifest
  SHA-256 `b5a61ebc91e8ac94686af13af47ca8714df9b290bae239d820d699c510f7fe4d`;
- untouched formal target: 1,625 questions over 1,050 images, manifest SHA-256
  `5a3ddca2e6476196aac8ad4fa7bc00033f2ac9c39d2011fe21fa070e965b97d4`;
- identity audit SHA-256
  `7737888c136ebc71cc2edce6f632c43c3d726a0fa5163d420047cce170a5f13e`;
  and
- normalized image bundle SHA-256
  `c4946970db3576cab6f136a72465cf4bf1c63cad5d0734d57af92d2870d35fd1`.

An independent post-export pass re-decodes all 1,250 PNGs, verifies every RGB
digest against its filename, confirms zero pilot/formal image overlap, and
confirms that both frozen scorers assign all 1,934 retained gold answers a
self-score of exactly 1.0. It also verifies that all version-2 VLM prompts are
byte-identical to the version-1 gate-visible strings and that every version-2
core question differs from its backend prompt. The export is bound to code
revision `3cd17c3ee345bd0348038ac866717a64d7eb65e7`.

The first compatibility rollout used the version-1 conflated state and is kept
only as a VLM/scorer diagnostic. It completed all 1,545 sibling records with no
empty output and max-token cap rates below 1%, but raw constrained-format
compliance was 87.9% rather than the registered 95%. The 46 deviations were
short, structurally recognizable outputs such as an option letter followed by
the option text or bracketed booleans; no explanatory answer was observed. The
version-1 gate result is invalid because the full benchmark prompt entered its
semantic features. No formal-target outcome has been read, and formal execution
remains blocked until prompt isolation and the constrained-answer handling are
frozen from a version-2 compatibility run.

Before inspecting a version-2 rollout, the compatibility handling is frozen as
a conservative parser. For Multi Choice it canonicalizes only an explicit
leading option letter followed by a closing bracket or punctuation delimiter;
for Fact Checking it canonicalizes only a fully bracketed `true` or `false`.
It does not extract an answer from prose, and every unmatched response remains
unchanged. This parser covers all 380 constrained version-1 pilot outputs while
changing exactly the 46 raw-format deviations. Compatibility acceptance uses
at least 95% conservative parse coverage and zero obvious explanatory outputs.
The released-code scorer on raw outputs remains the primary benchmark result;
raw paper-spec exact match and canonicalized paper-spec exact match are both
reported as sensitivities.
