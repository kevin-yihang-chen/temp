# ScreenQA Qwen-7B backbone mechanism diagnostic protocol v1

Status: frozen on 2026-08-31 after the opened ScreenQA Qwen-3B proxy audit
was reported and while the preregistered DocVQA cross-domain proxy replication
was still running, but before the 512-state population below was selected and
before any Qwen-7B outcome or likelihood was computed on that population.

This is an opened ranker-development diagnostic, not independent validation.
It cannot revise the failed DocVQA formal decision, select a deployment
threshold, or authorize opening ScreenQA calibration, formal, reserve,
untouched, validation, or test outcomes.

## Fixed question

Does the hierarchy observed with Qwen2.5-VL-3B—target-answer loss gap being
more aligned with signed crop utility than entropy reduction—replicate when the
acting and scoring backbone is Qwen2.5-VL-7B?

The Qwen-7B acting policy and the Qwen-7B answer-likelihood measurement must be
self-consistent. Reusing Qwen-3B task outcomes with Qwen-7B likelihood is
forbidden. Existing Qwen-3B outcomes may be used only as disclosed prior
motivation and never for population selection, hardware choice, or tuning.

## Outcome-blind population

- Parent manifest:
  `artifacts/screenqa-train-factorized-v1/ranker-manifest-v1/manifest.jsonl`.
- Parent manifest SHA-256:
  `a2b6941e2a073b24571d2ccb50960f7c1cd70cb0ce53dc8339c7ec44a47f67ec`.
- The parent contains 14,511 states from 1,510 opened ranker-development
  source groups.
- Select exactly 512 source groups by ascending
  `SHA256(namespace + NUL + seed + NUL + "source" + NUL + source_id)`, with
  source ID as a tie breaker.
- Namespace: `beyond-entropy-screenqa-backbone-7b-v1`; seed: `20260901`.
- Within each selected source, select exactly one state by ascending
  `SHA256(namespace + NUL + seed + NUL + "state" + NUL + source_id + NUL +
  state_id)`, with state ID as a tie breaker.
- Preserve the selected rows in parent-manifest order. Selection may read only
  `source_id` and `state_id`; target text is copied only after selection so the
  opened development task can be scored.

Bound selector:

- code revision: `7b8f5e803224e10f426c2de5257b40af23058c05`;
- module SHA-256:
  `2f43906dc818a5e734e7498d4997ebb35f6821cbca25354e193f541abe138cbd`;
- CLI SHA-256:
  `a6c002913e276be052e8bccf1271708a3dfe8dd9edaa0eb90b0ca6094770ad70`;
- test SHA-256:
  `e144ffa5085df1c5c013d98379d42451277f849bcc364f43352f9cc3cdc01939`;
- targeted tests: `2 passed` before this protocol was written.

## Frozen acting and measurement model

- Model: `Qwen/Qwen2.5-VL-7B-Instruct`.
- Revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- System prompt: `You are a helpful assistant.`; preserve each stored
  `model_prompt` exactly.
- Deterministic decoding, generation seed `0`, bfloat16, SDPA, min/max pixels
  `200704/602112`, and `max_new_tokens=32`.
- Actions: one ANSWER and four deterministic UG-grid ZOOM siblings with the
  frozen `visual_crop_ratio=2` and `tool_cost=1`.
- Primary outcome: official ScreenQA Short exact match under the existing
  scorer. Cost-adjusted utility remains `task_gain - 0.05 * tool_cost`.
- After all five Qwen-7B sibling outcomes exist, measure target-answer mean NLL
  under the same Qwen-7B revision, prompt, crop, dtype, attention, and pixel
  contract. Never write raw target text into the likelihood artifact.

## Hardware and quota rule

The full 512-state run is expected to exceed one hour on one consumer GPU, so
it is eligible for four-GPU execution and an advanced accelerator. Immediately
before submission, re-query live quota, the one-job/four-GPU account limits,
and queue state for RTX 4090, H800, and H100.

1. Run a maximum 32-state engineering smoke without reporting or using task
   endpoints for model or hardware selection.
2. Prefer four H100 or H800 GPUs when the projected queue-plus-runtime is below
   the four-4090 projection, the complete reserve fits the live GPU-minute
   quota, and the 7B model loads without quantization or offload.
3. Otherwise use four RTX 4090 GPUs if the same checks pass. Do not shrink the
   model, reduce the fixed population, change pixels, or mix accelerator types
   merely to fit a queue.
4. Every full rollout shard and every likelihood shard must use one common
   accelerator class. Record accelerator name, compute capability, package
   versions, requested/actual dtype, attention implementation, wall time,
   queue wait, and peak allocated memory.
5. Use deterministic source-aligned shards, atomic complete-state checkpoints,
   exact-prefix resume, and a byte-identical merge audit. Failed partial shards
   remain available for same-contract resume.
6. Configure Slurm email to `yihangc@connect.hku.hk` for all execution-state
   changes. Never write an access token to a script, log, manifest, or artifact.

If the pinned 7B snapshot is absent, download only that official revision into
the existing private Hugging Face cache after verifying at least 20 GiB free.
The user's credential may be supplied through a private environment variable
or authenticated cache and must never be printed or committed.

## Frozen analysis

For each of 2,048 Qwen-7B ZOOM actions define task gain, entropy reduction,
answer-loss gap, and utility exactly as in the ScreenQA Qwen-3B audit. Report:

1. Pearson and Spearman correlation of entropy reduction and answer-loss gap
   with signed task gain;
2. top-one task gain, utility, rescue, harm, and oracle regret for answer loss,
   entropy, exact uniform random, and oracle;
3. descriptive fixed call-rate grids for answer loss and entropy, without
   selecting a deployment rate or threshold;
4. 5,000 iid whole-source percentile bootstrap resamples with seed `20260903`
   and two-sided 95% intervals; and
5. answer-loss/task disagreement counts and identifier-only examples.

Because there is one state per source, the bootstrap has exactly 512
independent source units. Sparse call-rate endpoints are descriptive only and
are not part of the replication decision.

## Preregistered backbone decision

Declare **strong backbone replication** only if all conditions hold:

1. the 95% lower endpoint of answer-loss Spearman is greater than zero;
2. the 95% lower endpoint of answer-loss top-one mean task gain is greater than
   zero;
3. the answer-loss top-one task-gain point estimate exceeds both entropy and
   exact uniform random; and
4. the answer-loss top-one induced-harm point estimate is lower than both
   entropy and exact uniform random.

Declare **partial backbone replication** if conditions 1 and 2 pass but either
condition 3 or 4 fails. Otherwise declare **backbone non-replication**.

Strong replication supports only the claim that the proxy mechanism persists
across the 3B/7B Qwen backbone scale on opened ScreenQA development sources.
It does not establish policy generalization, SOTA accuracy, formal utility, or
generality to another model family. Partial or non-replication must be reported
and cannot be hidden by expanding the population or changing hardware.
