# InfographicVQA baseline-attention where protocol v1

Status: frozen on 2026-09-02 after the relative-where source-OOF gate and its
registered action-generalization diagnostic were opened, but before extracting
or scoring any InfographicVQA question-region attention. All validation, test,
and other protected outcomes remain sealed.

## Motivation and scope

The existing question/global/region embedding bank fails to generalize crop
location across official-train sources. Four target/weighting variants achieve
chance four-way, row, column, and pairwise agreement despite strongly falling
training losses. This protocol therefore does not train another crop ranker on
that representation.

The candidate changes the spatial evidence: use raw question-query attention
to original-image visual tokens from the frozen Qwen baseline forward pass,
pool that map into the same four UG candidate boxes, and choose the box with
maximum attention density. No candidate crop or post-action signal is used to
construct the score.

This is an official-train method-development gate. It cannot revise any prior
negative result. Passing it authorizes only a separate deployment/calibration
freeze; it does not by itself establish a validation/test claim.

## Bound prior evidence

```text
42cce38d6eb62396d4cfb38534537c1517ba72ae37e25561f4ef80732b853ca8  relative-where OOF result
dba870a7b350ed245c230a42b494e526fff58f64b8f64237097c85e99f27e783  action-generalization diagnostic protocol
1fc3c78174a7f0b2479c6f56ae1586b8c31e4c8b23d919cd12f708b3d9b3a428  action-generalization diagnostic result
1363d5f148a8624741a973c3de1930034901ed7c7fe70095d70c1d4cf772d198  action-generalization audit JSON
6ef0869b453e1a70ad5f479e8a9604aa04ac0419a1dc92d1ce353c54f66f3025  entropy-when/oracle-where factorization evaluation
```

The independent DocVQA formal diagnostic found 53.23% raw-attention rescue on
helpful states versus 40.32% expected for a random crop. The independent fresh
TextVQA diagnostic found 52.14% versus 41.77%. Those opened results motivate
the fixed raw-attention construction but do not determine any InfographicVQA
threshold or outcome.

## Frozen official-train population

Use exactly the completed full InfographicVQA official-train bank:

- 23,946 decisions, 4,406 images, 2,204 whole-source groups;
- one `ANSWER` and four actions `ug-grid-00` through `ug-grid-03` per decision;
- model `Qwen/Qwen2.5-VL-7B-Instruct`, revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`;
- `min_pixels=200704`, `max_pixels=602112`, `bfloat16`, system prompt
  `You are a helpful assistant.`;
- visual proposer `ug-grid`, crop ratio 2.0, visual cost 1.0.

Bound merged inputs:

```text
9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e  merged rollouts
d0508726a50b4c1e54778392d08329b242a680fc13292cac1ebec8b42a175300  merged label-free semantic features
884de4ebc1ba83226393871cdfff819bc7553a61eee6ffa11697db03332ac646  merged answer NLL
17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6  reused source-bootstrap indices
5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0  reused source order
```

The four frozen source-disjoint feature/rollout shard pairs are:

| Shard | Decisions | Sources | Base features SHA-256 | Rollouts SHA-256 |
|---:|---:|---:|---|---|
| 0 | 6,014 | 538 | `2ef27cfe17b5d8d36bd4410850a24e982b23f7686a9fa064c48db73c1ba0f3da` | `1e130c22a2b1ba85e41c12a18dbef66e717fafc4a996f6dde90577a839d1b6da` |
| 1 | 6,036 | 597 | `ed643ad1d4b82500db3dd3cec6f7d6d01412cef90e7f55690ad2e57b70cabdeb` | `506928e10c66bd47caab2be0631a63a0f63abb83e520a8769c826b84f5fe35b9` |
| 2 | 5,910 | 547 | `6cf284fec70ad2873ff05a1ef17ab0958ab57d92bc1b340c03a749fdb470a69b` | `a486e8d5e34b7ac7846b074782738826ee1bd81a1ec5833c4a18011a0597c8da` |
| 3 | 5,986 | 522 | `4eb20a4d9ca35b693889406eb82c74f1c635e93ab07ec74917fe0976773d948e` | `e8bd28e33f7d72135728fb567ea9d3407fb325c7e0e49af8f860fb37c4606d83` |

No validation/test path or outcome may be read by the extraction, merger,
auditor, evaluator, or submitter.

## Frozen attention construction

For every decision, replay only the original image and exact user question:

1. load the pinned Qwen model with eager attention and no generation;
2. identify the exact question-token span in the frozen chat template;
3. from the final four language layers, average attention over layers, heads,
   and question query tokens;
4. retain attention whose key positions are original-image tokens;
5. restore the merged visual-token grid;
6. clamp attention nonnegative and normalize total image mass to one;
7. ROI-mean pool the four frozen UG boxes and normalize the four densities to
   sum to one.

Store exactly four finite nonnegative `question_region_attention` values and
the scalar `question_image_attention_mass`. The attention values must sum to
one within float32 tolerance `1e-6`. Do not replace the original question
embedding. Ties select the lexicographically first action ID.

The feature is inference-visible and outcome-free. Candidate crops are not
executed. The forward uses eager attention rather than the original SDPA
generation forward, so runtime and memory overhead must be reported; it may
only be described as baseline-forward reusable, not as zero compute.

## Extraction and engineering contract

- One Slurm job with four NVIDIA H800 GPUs, one frozen source shard per GPU.
- `bfloat16`, `device-map=cuda:0`, offline local model files, no CPU/disk
  offload, no network credentials or proxy variables.
- Resume-safe checkpoint interval 512 decisions per shard.
- All-state email to `yihangc@connect.hku.hk`.
- Hash the worker, protocol, code revision, all shard inputs, every shard
  output, merged output, audit, evaluation, decision, and execution record.
- A pre-scoring smoke may use at most eight decisions and may inspect only
  shapes, finiteness, normalization, identity alignment, and leakage fields.
  It may not inspect action outcomes or select scientific settings.
- Scientific evaluation may start only after all 23,946 rows merge exactly and
  the label-free audit reports `outcomes_included=false` and no outcome field.

Any partial extraction is engineering state only. A failed or interrupted job
must resume the same outputs and settings; it must not create a scientific
result.

## Frozen policy family

The sole candidate action is
`argmax(question_region_attention)`. The sole stopping score is the already
observed answer-now mean normalized token entropy. Evaluate the exact global
entropy call sets with nominal question rates:

```text
0.005, 0.01, 0.02, 0.05, 0.10
```

The call count is `ceil(rate * 23946)`, with descending entropy and
`(state_id, replicate_id)` tie breaking. Tool cost is `lambda=0.05` per crop.
There is no fitted model, attention-layer search, head selection, threshold
search, temperature, spatial smoothing, score fusion, or post-hoc abstention.

Report at each operating point:

- source- and question-balanced ANLS gain, call rate, cost-adjusted utility,
  gain per call, induced harm, helpful-call precision, helpful-state recovery,
  negative-utility call mass, and action-selection regret;
- exact crop agreement, row/column agreement, helpful-state rescue, and NLL
  regret for the raw attention action;
- paired whole-source intervals for candidate utility and differences from all
  registered baselines;
- call concentration by source and attention confidence/margin deciles.

## Frozen comparators

At exactly the same entropy call set, compare:

1. answer now;
2. fixed `ug-grid-00`;
3. deterministic uniform-random crop using the existing frozen comparator;
4. the original source-OOF DECAR action;
5. the failed relative-teacher-entropy action, as a same-bank learned-ranker
   diagnostic;
6. privileged best-NLL crop and task-outcome oracle, both clearly marked
   non-deployable ceilings.

All pre-existing comparator aggregates must reproduce exactly before any new
metric is accepted. Use the already frozen 20,000 paired whole-source bootstrap
indices and source order for every policy and difference.

## Registered advancement rule

An operating point qualifies only if all of the following hold:

1. every population, identity, hash, normalization, outcome-exclusion,
   comparator-reproduction, and bootstrap audit passes;
2. candidate source-balanced utility is positive and its 95% bootstrap lower
   endpoint is strictly above zero;
3. the 95% paired lower endpoints of candidate-minus-fixed,
   candidate-minus-random, candidate-minus-original-DECAR, and
   candidate-minus-relative-where utility are all nonnegative, with at least
   one of the four strictly positive;
4. candidate induced harm does not exceed the best deployable comparator at
   that rate by more than `0.00025` source-balanced mass;
5. the candidate closes at least 25% of the source-balanced action-selection
   utility gap from deterministic random to the task oracle.

If multiple points qualify, choose the one with the largest candidate utility
lower endpoint, then largest point utility, then least induced harm, then lower
nominal rate. Emit `attention_where_train_supported` and freeze only that
nominal rate for a later calibration protocol.

If none qualifies, emit `attention_where_train_not_supported`; keep every
protected role sealed and do not tune attention construction on this result.

No GitHub push is authorized by this protocol.
