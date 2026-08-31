# Proxy-NLL hardware consistency protocol v1

Status: frozen on 2026-08-31 after one 4090 decision and 64 H800 decisions had
been scored for engineering validation, but before the matched 64-decision 4090
benchmark and before any full-bank proxy audit.  The observed one-decision
maximum absolute mean-NLL difference was `0.008493825793266296`; the four crop
loss-gap signs and their top-one ordering agreed.  This observation motivates
the audit below and is disclosed rather than treated as a scientific endpoint.

## Matched engineering population

- Use the same 64 opened ScreenQA ranker decisions at sorted positions
  `position % 227 == 0`.
- Score every ANSWER and four ZOOM siblings with the frozen Qwen2.5-VL-3B
  revision, prompt, target rule, bfloat16, SDPA, and pixel limits.
- Run one benchmark on RTX 4090 and one on H800.  Both runs must use the same
  scoring implementation revision after runtime provenance was added.
- Record accelerator name, compute capability, requested and parameter dtype,
  requested and actual attention implementation, pixel limits, prompt, PyTorch,
  CUDA-runtime, and Transformers versions in the configuration hash.

## Consistency endpoints

For each of the 256 matched ZOOM actions, compute the answer-loss gap within its
own hardware run: `ANSWER mean NLL - ZOOM mean NLL`.  Report:

1. Pearson and Spearman correlation between H800 and 4090 loss gaps;
2. positive/nonpositive loss-gap sign agreement;
3. top-one crop agreement across the 64 decisions, with lexicographic tie-break;
4. median, 95th percentile, and maximum absolute loss-gap difference;
5. end-to-end benchmark time and projected four-GPU full-bank wall time and
   GPU-minutes for each hardware type.

These are numerical-stability diagnostics, not task results and not a new model
selection endpoint.

## Hardware decision rule

- Prefer 4 x RTX 4090 for the full audit when its measured projection is at most
  four hours and fits the live remaining account quota, because the stored
  ScreenQA sibling outcomes were generated on RTX 4090.
- If that condition fails, H800 is eligible only when loss-gap Spearman is at
  least `0.99`, sign agreement is at least `0.95`, and top-one agreement is at
  least `0.95`.  Otherwise retain 4090 and use resumable shards rather than mix
  hardware inside the full score artifact.
- All four full-bank shards must use one hardware type.  A partial shard may be
  resumed only on the same recorded accelerator class and numerical contract.

The selected hardware and this audit must be disclosed in the final report.
