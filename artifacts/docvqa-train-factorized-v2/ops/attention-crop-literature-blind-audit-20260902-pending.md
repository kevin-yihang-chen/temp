# Attention-crop literature blind audit (pending integration)

Status: written on 2026-09-02 while InfographicVQA raw-attention extraction
job 203257 was still running and before any of its action outcomes were opened.
This note does not modify the frozen experiment. It remains untracked until the
revision-bound dependent evaluator has passed its clean-worktree startup check.

## Material overlap

**ViCrop / MLLMs Know Where to Look (ICLR 2025)** already establishes the
training-free attention-guided crop baseline that the current raw-attention
InfographicVQA branch most closely resembles. It maps starting-answer-token
attention back to image patches, proposes relative attention by normalizing a
question map with a generic-description map, and converts the map to a crop
through sliding-window search. It also tests gradient-weighted attention and
input-gradient variants, keeps the original image alongside the crop, and
reports broad VQA gains. Therefore neither "the VLM already knows where to
look," model-internal attention/gradient crop localization, nor a
training-free attention crop can be claimed as new. ViCrop does not decide
whether the crop has positive signed task value, charge action cost, compare
every same-state action, or prospectively constrain induced harm.

The official repository was audited at commit
`a64c5d5a0ece793a2f7dc96e926d283a199033c1`. Its Qwen2.5 implementation fixes
zero-based language-layer index 22, averages heads at the final prefill
position, divides question attention elementwise by generic-description
attention, and uses the exact generic instruction
`Write a general description of the image.`. The implementation files audited
have SHA-256 `618fec20571c060705ce1ddf0930241e8bbf529ce51b380de38f450e19b23512`
(`qwen2_5_methods.py`) and
`6d27b7430d3d6046c4ce708b57cb2aa903fa512c73a023422d0c8c60329c750a`
(`run.py`).

**LASER / Beyond Static Cropping (arXiv:2602.04304)** is an even stronger
collision for any proposed attention-map extension. It contrasts visual
attention with and without the question to suppress query-invariant attention
sinks, selects a query-specific layer through Visual Activation by Query, uses
the selected contrastive map for constrained cropping, and optionally performs
counterfactual decoding. It explicitly compares raw attention, ViCrop-style
relative attention, and dynamic contrastive attention on Qwen-VL and LLaVA.
Consequently, query/no-query attention contrast, dynamic attention-layer
selection, prefill attention localization, and attention-sink correction are
not available as novelty claims. LASER still always constructs its enhanced
inference path and optimizes VQA accuracy rather than estimating the signed
costed effect of a fixed candidate bank or independently gating deployment
harm.

The official arXiv source archive was audited with SHA-256
`a6fb5c55be5df434404a8d13a23b3e239d23ac9e6df68f08b8e39c0786d38d10`.
Both the rendered paper and source retain `K_head` only as a symbol and never
state its numeric value; `section/method.tex` and `section/exp.tex` have
SHA-256 `0d329f129be640b860a72f62fdc6f13b023e26118ce11f99e0a69c8eb9a72e7c`
and `8486bca1a1699da06e606b55d311e25cb69a16ba08505306901c96fc00e5c693`,
respectively. A public GitHub repository search on 2026-09-02 found no LASER
implementation. An all-head fixed-bank projection is therefore more auditable
than inventing a hidden top-head constant, but must not be called an exact
reproduction.

**ENCORE (ICASSP 2026; arXiv:2608.22996)** uses early-layer image--text
attention entropy over predefined crop aspect ratios, chooses the
minimum-entropy crop at inference, and adds attention-entropy regularization
during fine-tuning. It reports ten-benchmark VQA gains and measured latency and
FLOP overhead. This directly removes novelty from generic claims about
attention-entropy crop selection or prompt-relevant entropy as a crop-quality
signal. ENCORE selects an input layout by semantic-fragmentation entropy and
does not retain realized signed task effects for answer-now and every concrete
candidate, impose an explicit tool cost, or prospectively control
right-to-wrong harm.

The frozen current branch remains useful as a proposer diagnostic: it uses a
single original-image Qwen prefill, pools final-four-layer question-query
attention into four already frozen UG boxes, executes no candidate during
selection, and combines that proposer with the separately frozen entropy stop
set and harm gate. It must be described as a deliberately simple
ViCrop/LASER-adjacent baseline or component, not as a standalone novel
attention-guided cropping method. Its gate must not be modified after this
audit; any relative-attention or dynamic-layer follow-up requires a new
predeclared branch and fresh extraction.

## Required paper consequences

- Add ViCrop raw/relative-attention and gradient-guided crop localization at
  matched call sets, with original-plus-crop compute reported.
- Add LASER query/no-query contrastive attention and dynamic-layer
  localization, separated from optional counterfactual decoding.
- Add ENCORE-style early-layer image--text entropy crop selection, clearly
  distinguishing layout/aspect-ratio selection from action-value gating.
- Do not claim first training-free attention-guided crop, first use of a VLM's
  internal attention to locate a crop, first query-contrastive attention crop,
  first dynamic-layer attention crop, or first attention-entropy crop
  selection.

## Primary sources audited

- ViCrop / MLLMs Know Where to Look: https://arxiv.org/abs/2502.17422
- ViCrop official code: https://github.com/saccharomycetes/mllms_know
- LASER / Beyond Static Cropping: https://arxiv.org/abs/2602.04304
- ENCORE: https://arxiv.org/abs/2608.22996
