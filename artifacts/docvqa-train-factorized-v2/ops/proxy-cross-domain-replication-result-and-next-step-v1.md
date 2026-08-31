# DocVQA proxy cross-domain replication result and next-step decision v1

Status: completed and frozen on 2026-09-01. This is a retrospective
cross-domain diagnostic on the opened DocVQA ranker-development bank. It is not
candidate selection, independent validation, or authorization to open any
protected DocVQA or ScreenQA role.

## Execution and integrity

- Slurm job `197943` completed with exit code `0:0`, zero restarts, and a
  runtime of `01:50:00` on four NVIDIA GeForce RTX 4090 GPUs.
- Each of the four score shards contains exactly `3,395` decisions and
  `16,975` records. The strict merge contains `13,580` decisions, `67,900`
  records, `54,320` ZOOM actions, and `3,500` source groups.
- Every decision contains one ANSWER sibling and four ZOOM siblings. All NLLs
  are finite and nonnegative, all token counts are positive, sibling metadata
  is consistent, and no raw target field was written.
- All shards record Qwen2.5-VL-3B revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`, code revision
  `05a4d4907039029f8372dcff80239c9cf9e14dfd`, bfloat16, SDPA, and the RTX
  4090 measurement contract.
- Merged score SHA-256:
  `f23e32bfbfa264f0362dd43881443c9c6ed507400d1fe7c2577688db5767e938`.
- Audit report SHA-256:
  `38c2147d9a9dc22597f038bd091874b9d2d44be829029885ac3a6f4f6dbd1954`.
- Markdown report SHA-256:
  `0e653397478660e55dae9bba2ee29f039da31de3cda873dd41894bbecabb09e8`.
- The report uses 2,000 iid whole-source bootstrap resamples, seed `20260901`,
  and two-sided 95% intervals. Its outcome-use record confirms that no
  calibration, formal, reserve, validation, test, or other protected role was
  used and that candidate search was not reopened.

## Cross-domain result

Target-answer loss again ranks realized visual actions substantially better
than entropy reduction. The effect is at least as clear on DocVQA as on the
previous ScreenQA opened-development audit.

| Diagnostic | Answer-loss gap | Entropy reduction | Exact random crop |
|---|---:|---:|---:|
| Spearman with signed task gain | 0.24343 [0.22995, 0.25684] | 0.13185 [0.11532, 0.14854] | n/a |
| Top-one crop mean task gain | 0.01911 [0.01644, 0.02191] | 0.00648 [0.00364, 0.00948] | -0.00533 [-0.00726, -0.00336] |
| Top-one rescue within helpful states | 0.89754 [0.87728, 0.91508] | 0.79507 [0.76821, 0.82049] | 0.39507 [0.38122, 0.40950] |
| Top-one induced-harm rate | 0.01119 [0.00941, 0.01303] | 0.02135 [0.01900, 0.02379] | 0.02826 [0.02591, 0.03061] |

Answer-loss sparse-policy utility has a strictly positive 95% lower endpoint
at every preregistered rate from 0.5% through 10%. At the descriptive 10% grid
point, its mean policy utility is `0.012256` [`0.010241`, `0.014339`]. No rate
or score threshold was selected; these numbers cannot be reused as a frozen
deployment policy.

## Preregistered decision

The hash-bound decision runner returned **`replicated_alignment`**. All five
conditions passed:

1. answer-loss Spearman lower endpoint is positive: `0.229946`;
2. answer-loss top-one task-gain lower endpoint is positive: `0.016435`;
3. answer-loss top-one gain `0.019106` exceeds entropy `0.006477` and random
   `-0.005328`;
4. all fixed sparse rates from 0.5% through 10% have positive answer-loss
   utility lower endpoints;
5. answer-loss top-one harm `0.011193` is below entropy `0.021355` and random
   `0.028258`.

Decision JSON SHA-256:
`f7f95a79e8d1aa386f4f10a909968a844bf94e5bb01ae6a2cf805fc8c2d10ca2`.
The decision explicitly records that no score threshold, call rate, or
protected outcome was selected.

## Hardware interpretation

RTX 4090 was a measurement-control choice, not the model. A matched 64-decision
engineering audit found H800 faster (`85` versus `100` measured seconds), but
small hardware-dependent NLL differences failed every preregistered
sign/ranking-stability gate. Because both the prior sibling outcomes and the
DocVQA pipeline used the 4090 execution family, retaining one accelerator class
prevented a hardware change from confounding the cross-domain conclusion.

This constraint applies only to the frozen Qwen2.5-VL-3B comparison. The
separate Qwen2.5-VL-7B mechanism diagnostic is explicitly allowed to prefer
H100 or H800 after an endpoint-blind engineering smoke, provided all rollout
and likelihood shards use the same accelerator class.

## Paper asset

The cross-domain figure combines the frozen ScreenQA and DocVQA reports without
selecting a threshold or call rate:

- PDF SHA-256:
  `3f07404ae5bc4f510e2760cc7772792e247903fea28cbf30c8c31f5616c167b7`;
- PNG SHA-256:
  `1e5d7427256bd81cba5d4ed6d927762d3af3cdcc154dfd57fc4487cdf9514e59`;
- CSV SHA-256:
  `5e2fa71a838279441b193395dfe651f8b3ac10a2ebb411aa8669e131abf294a3`;
- provenance SHA-256:
  `f8408c99f3a8f6e5f61972701cf029fc5ff05d981b5470fe08dd38652618f1ce`.

## Frozen next route

The positive cross-domain result supports a method-transfer route, not a claim
that a deployable gate already exists. The next method must predict visual
action utility before using the target answer and must be trained on a
different development population. The preferred design remains a low-capacity
joint stop/action model with:

1. signed realized task gain as the primary action target;
2. teacher-forced target-answer loss gap only as training-time auxiliary
   supervision;
3. a separate induced-harm target or constraint;
4. pre-action features only at inference;
5. matched entropy, random/fixed crop, exhaustive UG, task-value-only,
   loss-only, and joint-supervision comparators;
6. source-disjoint calibration and a one-shot untouched confirmation whose
   population, metrics, and failure rule are frozen before outcomes are opened.

Before fitting or opening a protected destination role, a separate transfer
protocol must bind the source/destination identities, feature schema, exact
model family and capacity, folds, fixed call-budget grid, multiplicity control,
and success/failure rule. In parallel, the already frozen Qwen2.5-VL-7B opened
development diagnostic tests whether the mechanism survives a stronger acting
and scoring backbone. Neither branch may reopen the failed ScreenQA semantic
candidate or alter any earlier formal conclusion.

No GitHub push is authorized by this result.
