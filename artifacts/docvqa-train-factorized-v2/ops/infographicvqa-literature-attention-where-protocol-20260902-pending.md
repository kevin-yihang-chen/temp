# InfographicVQA literature-attention where protocol (pending integration)

Status: frozen blind on 2026-09-02 while raw-attention extraction job 203257
was running, before its feature merge, action evaluation, or any protected
outcome was opened. This is a separately declared literature-comparator branch;
it does not alter the raw-attention protocol or its pass rule. The file remains
untracked only because dependent evaluator 203262 is revision-bound and
requires a clean tracked worktree at startup.

Primary method sources audited before this freeze:

- ViCrop / MLLMs Know Where to Look (ICLR 2025):
  https://arxiv.org/abs/2502.17422
- LASER / Beyond Static Cropping (2026):
  https://arxiv.org/abs/2602.04304
- ENCORE (ICASSP 2026): https://arxiv.org/abs/2608.22996

Official ViCrop code binding:

```text
a64c5d5a0ece793a2f7dc96e926d283a199033c1  repository commit
618fec20571c060705ce1ddf0930241e8bbf529ce51b380de38f450e19b23512  qwen2_5_methods.py
6d27b7430d3d6046c4ce708b57cb2aa903fa512c73a023422d0c8c60329c750a  run.py
```

LASER arXiv source binding:

```text
a6fb5c55be5df434404a8d13a23b3e239d23ac9e6df68f08b8e39c0786d38d10  source archive
0d329f129be640b860a72f62fdc6f13b023e26118ce11f99e0a69c8eb9a72e7c  section/method.tex
8486bca1a1699da06e606b55d311e25cb69a16ba08505306901c96fc00e5c693  section/exp.tex
```

The paper and source never instantiate numeric `K_head`, and no public LASER
code repository was found in a 2026-09-02 GitHub repository search. The
all-head rule below is thus an explicit preregistered adaptation, not a guessed
reproduction parameter.

## Purpose and non-claim

This branch asks whether externally established attention-localization
constructions solve the action-proposal bottleneck under the existing signed,
costed, source-balanced gate. It is a strong-baseline and component-selection
study, not a novelty claim for attention-guided cropping. It must run once with
the settings below whether the simpler raw-attention branch passes or fails.
No setting may be selected from job 203257/203262 outcomes.

## Frozen population and bindings

Use exactly the existing InfographicVQA official-train bank:

- 23,946 decisions, 4,406 images, 2,204 whole-source groups;
- one `ANSWER` and four `ug-grid-00` through `ug-grid-03` actions;
- `Qwen/Qwen2.5-VL-7B-Instruct` at revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`;
- identical image preprocessing, chat template, system prompt, candidate boxes,
  tool cost, source order, and 20,000 source-bootstrap draws as the frozen
  raw-attention protocol.

Bound inputs:

```text
9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e  merged rollouts
d0508726a50b4c1e54778392d08329b242a680fc13292cac1ebec8b42a175300  merged label-free semantic features
884de4ebc1ba83226393871cdfff819bc7553a61eee6ffa11697db03332ac646  merged answer NLL
17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6  source-bootstrap indices
5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0  source order
```

Extraction and model inputs remain outcome-free. Candidate crops are never
executed while constructing the attention scores. Validation and test remain
sealed.

## Three frozen prefills

For each original image, run eager-attention prefills in `bfloat16` under the
same frozen image preprocessing. Preserve the project system prompt so the
proposer observes the same deployed context, but bind the ViCrop question
suffix exactly:

1. **query:** the exact user question followed by
   ` Answer the question using a single word or phrase.`;
2. **generic:** the exact string
   `Write a general description of the image. Answer the question using a single word or phrase.`;
3. **no-query:** the image-only user message with no text content.

For every condition, retain the final prefill/assistant-prefix query position's
attention to the merged original-image token positions for every language layer
and head. Do not generate an answer. Audit identical visual-token count and
spatial grid across the three conditions. Attention outside image-token key
positions is discarded only after the model forward.

## Frozen action-score variants

### Primary: `vicrop_relative_bank`

- Use language-layer index 22 (zero-based), matching the audited official
  Qwen2.5 code, and mean all returned attention heads.
- Let `a_q` and `a_g` be query and generic image-token attention maps.
- Compute the official elementwise ratio `r = a_q / a_g` in float32, without
  adding an unreported epsilon. A zero denominator or nonfinite result is a
  hard audit failure; no clipping, temperature, smoothing, or threshold is
  allowed.
- ROI-mean pool `r` into the four frozen UG boxes, normalize the four finite
  nonnegative densities to sum to one, and choose the lexicographically first
  argmax.

This preserves the official Qwen2.5 prompt, layer, final-position, head pooling,
and relative-attention formula while retaining the project's pinned model
revision and image preprocessing. It is still a fixed-candidate-bank
projection, not an exact reproduction of ViCrop's adaptive sliding-window crop,
224-base crop geometry, or original-plus-crop answer input.

### Secondary: `laser_contrastive_all_head_bank`

- For every layer and head, extract query and no-query image-token attention
  vectors without separately renormalizing their image mass.
- Compute `c[l,h] = relu(a_q[l,h] - a_empty[l,h])`.
- Define the layer score as the mean across all heads of the L2 norm of
  `c[l,h]`; choose the maximum layer, breaking ties toward the lower index.
- Average `c` across every head at the selected layer, then ROI-mean pool and
  normalize the four candidate densities to sum to one.
- If every candidate density is zero, select `ug-grid-00` and record the event;
  any nonfinite value is a hard failure.

LASER reports a top-`K_head` rule but its audited paper does not state a numeric
`K_head`. This protocol therefore fixes an all-head projection rather than
inventing or tuning that missing constant. It is not an exact LASER
reproduction and excludes LASER's crop geometry and counterfactual decoding.

### Descriptive only: `encore_early_entropy`

At layers 0 and 1, mean the query image attention across all returned heads,
renormalize that image map to unit mass, and report its Shannon entropy and
association with baseline correctness, helpful-crop existence, and selected
action regret. Do not use it to choose a four-way action or stop threshold in
this branch: ENCORE scores alternative input layouts, whereas the current bank
contains fixed spatial actions, so an action mapping would require an
unregistered design choice.

The descriptive association is fixed as Spearman rank correlation with the
continuous baseline ANLS, the binary presence of any positive-gain crop, and
the selected-crop NLL regret for each of the ViCrop and LASER variants. Also
report mean entropy separately for helpful and non-helpful states. None of
these summaries may enter qualification or parameter selection.

## Frozen stop sets and comparators

For the two action-score variants, reuse the exact global entropy call sets at
nominal rates `0.005, 0.01, 0.02, 0.05, 0.10`, tool cost `lambda=0.05`, and all
metrics/comparators from the raw-attention protocol. Reproduce answer-now,
fixed `ug-grid-00`, deterministic random, original DECAR, relative-where,
privileged NLL teacher, task oracle, and the raw-attention result exactly before
accepting any new endpoint.

For qualification, the completed raw-attention policy is an additional fifth
deployable comparator alongside fixed, random, original DECAR, and
relative-where. Each literature candidate must be noninferior to all five
under the corrected paired intervals and strictly superior to at least one.

## Multiplicity-corrected advancement rule

The primary and secondary variants are jointly registered. Use central 97.5%
paired whole-source bootstrap intervals (Bonferroni correction for two
candidates) for candidate utility and candidate-minus-comparator differences.
A candidate qualifies only if it satisfies every audit, positivity,
noninferiority/superiority, induced-harm, and 25% random-to-oracle gap-closure
condition from the raw-attention protocol under these corrected intervals.

If both qualify, choose the candidate/operating-point pair with the largest
corrected utility lower endpoint, then largest point utility, least induced
harm, lower nominal call rate, then prefer `vicrop_relative_bank`. A qualifying
result authorizes only a separately frozen calibration protocol. If neither
qualifies, emit `literature_attention_where_train_not_supported` and keep all
protected roles sealed.

## Resource and audit contract

- Two H800 GPUs in two deterministic shard waves, at most 8 hours wall time;
- three prefills per decision, checkpoint every 256 decisions;
- no network credentials/proxies, all-state email, exact code/input/output
  hashes, resume-safe outputs, and measured extraction/merge/evaluation time;
- any partial output is engineering state only and cannot be scored;
- no GitHub push is authorized by this protocol.
