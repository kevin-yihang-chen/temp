# Qwen2.5-VL-7B backbone result and next-step decision v1

Status: completed and frozen on 2026-09-01. The preregistered mechanical
decision is **strong backbone replication**. This is an opened ScreenQA
ranker-development mechanism diagnostic, not candidate selection, independent
validation, a deployable policy, or authorization to open a protected role.

## Bound result

- Population: 512 distinct opened-development sources, one state per source,
  2,048 ZOOM actions, and 2,560 ANSWER/ZOOM likelihood records.
- Model: `Qwen/Qwen2.5-VL-7B-Instruct` revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, bfloat16, SDPA, no
  quantization or offload, on four NVIDIA H800 GPUs.
- Bootstrap: 5,000 iid whole-source resamples, seed `20260903`, two-sided 95%
  percentile intervals.
- Merged rollout SHA-256:
  `33e9330c144f65a148820cbe008fb9b469bdb1e7a1add9301302367dcc02c76a`.
- Merged answer-NLL SHA-256:
  `6c9fcae676d6d9aa5a7e13739b545575c013e61987914c1aaad167bbc6cb76c7`.
- Analysis report SHA-256:
  `7c890f308a2647b3ced9d98979dbee3d26e23fbff406e863a69797a6ab0f571b`.
- Analysis completion SHA-256:
  `e1c6e955770b71e6d80ecf4afbb1a834139abc49fb2696c3b3b577cb76064067`.
- Decision JSON / completion SHA-256:
  `58a6b524b01fec05b27651b63f72c519aa69ac12fa9426b78c4c8bfe2aa5e505`
  / `e0e5f2b2928b19e0118c4c712661b87ace26cf96d0bd377c33a9d6523fbc2a8e`.

No raw answer target was written to the likelihood artifacts. Candidate search
was not reopened. Calibration, formal, reserve, untouched, validation, and
test inputs were not used.

## Frozen four-condition decision

All conditions passed:

1. Answer-loss Spearman with signed task gain is `0.11619`, with 95% interval
   `[0.04468, 0.18175]`; its lower endpoint is above zero.
2. Answer-loss top-one mean task gain is `0.01758`, with 95% interval
   `[0.00391, 0.03125]`; its lower endpoint is above zero.
3. Answer-loss top-one task gain `0.01758` exceeds entropy `0.00391` and exact
   uniform-random `-0.00391`.
4. Answer-loss top-one induced-harm rate `0.00391` is below entropy `0.01563`
   and exact uniform-random `0.02197`.

Answer-loss top-one recovers 68.75% of the 16 helpful states, compared with
62.5% for entropy and 57.81% in exact uniform expectation. The answer-loss
selector is still imperfect: ten harmful actions have a positive loss gap and
ten helpful actions have a nonpositive loss gap in this population.

Entropy is not wholly uninformative at 7B: its Spearman interval is positive,
while its Pearson interval and top-one task-gain interval cross zero. The
defensible cross-scale claim is therefore that answer-loss supervision yields
the stronger top-one task-gain and harm profile under the frozen comparison,
not that every entropy statistic is zero.

## Execution and recovery disclosure

- `q-hgpu-small` rejected the initial four-H800 request before enqueue with
  `QOSMaxGRESPerUser`. Association inspection showed that four H800 GPUs were
  permitted; a zero-execution `--test-only` request passed in the dedicated
  `q-h800` partition. Only the partition/QOS was corrected.
- Job `199141` began the full rollout and was manually interrupted after an
  overly cautious but incorrect interpretation of unequal rollout-shard
  counts. The four rollout shards use stable state hashes and are expected to
  be unequal; the NLL shards use position modulo and are exactly 128 each.
  Atomic checkpoints were retained and no result was selected from this event.
- Resume job `199143` completed all 512 rollout states and 2,560 records, then
  failed before provenance because the worker passed zero resamples to a
  diagnostic whose implementation requires a positive count.
- The worker was corrected to 100 auxiliary per-shard resamples without
  changing any model call, prompt, action, target, population, scorer, or
  final-analysis setting. Job `199148` wrote byte-stable rollout provenance,
  merged the unchanged rollout bytes, and completed all four 128-decision NLL
  shards and their resume checks. It then stopped because a complete rollout
  resume does not instantiate the backend, so the original rollout peak-memory
  telemetry remained unavailable.
- Recovery job `199179` performed one deterministic state replay per original
  rollout shard on four H800 GPUs. Every five-record replay matched its main
  rollout records exactly before telemetry was attached. The original merged
  rollout was archived recoverably, strict merge reproduced the identical
  SHA-256 above, and the final analysis ran only after this equality check.
  Recovery took 21 seconds for replay and one second for analysis after a
  two-second queue wait.
- The recovery record explicitly states
  `original_process_peak_reconstructed=false`: replay peaks are hardware and
  configuration attestations, not invented measurements of the terminated
  original processes. Scientific rollout and NLL component hashes remained at
  their frozen values throughout.

## Scientific interpretation

The 3B mechanism result now replicates at Qwen2.5-VL-7B under a self-consistent
7B acting and scoring backbone. This materially strengthens the evidence that
target-answer loss contains crop-ranking information beyond post-action
entropy in this Qwen family and opened ScreenQA population.

It does not yet make the project a successful deployed method. Only 16 of 512
states contain any helpful crop, answer loss needs the accepted answer and is
unavailable at inference, and always calling has negative utility even for the
oracle because every tool call costs `0.05`. The remaining research problem is
selective pre-action stop/action prediction with explicit harm control.

## Next authorized step

Do not tune another ScreenQA threshold, feature, model, or call rate on this
opened bank. Freeze a separate non-ScreenQA development protocol for a
deployable joint stop/action value model that:

1. consumes only pre-action and candidate-geometry information at inference;
2. learns signed realized task effect with answer-loss as training-only
   auxiliary supervision;
3. has an explicit harm head or constrained decision rule;
4. uses source-disjoint out-of-fold predictions and calibration;
5. compares against entropy gate, random/fixed crop, exhaustive UG,
   task-value-only, and loss-only baselines at matched call budgets; and
6. preserves a one-shot untouched confirmation population.

The next paper claim should combine signed sibling outcome supervision, joint
stop/action selection, and prospective harm control. Cross-scale proxy
replication is supporting mechanism evidence, not the headline method by
itself. No GitHub push is authorized by this result.
