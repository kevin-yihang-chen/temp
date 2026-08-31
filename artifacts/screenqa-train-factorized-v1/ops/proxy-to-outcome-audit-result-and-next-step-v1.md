# ScreenQA proxy-to-outcome audit result and next-step decision v1

Status: completed and frozen on 2026-08-31. This is a retrospective diagnostic
on the opened ScreenQA ranker-development bank. It is not candidate selection,
independent validation, or authorization to open any protected ScreenQA role.

## Execution and integrity

- Slurm job `197357` completed with exit code `0:0`, zero restarts, and a runtime
  of `01:28:10` on four NVIDIA GeForce RTX 4090 GPUs.
- The four complete shards contain `3,628 / 3,628 / 3,628 / 3,627` decisions and
  `18,140 / 18,140 / 18,140 / 18,135` records.
- The strict merge contains exactly `14,511` decisions, `72,555` score records,
  `58,044` ZOOM actions, and `1,510` source groups. Every decision contains one
  ANSWER sibling and four ZOOM siblings.
- Every score is finite and non-negative. Every answer target is represented by
  a SHA-256 digest; no raw target field was written.
- All shards record the frozen Qwen2.5-VL-3B revision, code revision
  `4680dc3bca467b04a01717d71252322f60b79522`, bfloat16, SDPA, and RTX 4090
  runtime contract.
- Merged score SHA-256:
  `be0fc8c25109f9617f525fabecc370eac51e5990f4deaf7df4eaa33e36ca4e66`.
- Audit report SHA-256:
  `438a5e64072826480aa41a5ccf78224bb4b8191a04976bf26760f4278181276a`.
- Markdown report SHA-256:
  `9ef7223bd601a726de364b4c08fc8eea5a791827b93eed45bd56e3ed00fe2e66`.
- The protocol, implementation contract, and 2,000-resample whole-source
  bootstrap are bound by `analysis/audit.complete.json`.

## Primary result

Teacher-forced target-answer loss is materially more aligned with realized
signed task gain than post-action entropy reduction on this opened bank.

| Diagnostic | Answer-loss gap | Entropy reduction |
|---|---:|---:|
| Pearson with signed task gain | 0.2577 [0.2297, 0.2855] | 0.0991 [0.0760, 0.1213] |
| Spearman with signed task gain | 0.2035 [0.1917, 0.2155] | 0.0698 [0.0527, 0.0860] |
| Top-one crop mean task gain | 0.01730 [0.01408, 0.02044] | -0.00179 [-0.00495, 0.00151] |
| Top-one rescue within helpful states | 0.8455 [0.8117, 0.8803] | 0.6073 [0.5603, 0.6537] |
| Top-one induced-harm rate | 0.00985 [0.00822, 0.01167] | 0.02129 [0.01887, 0.02370] |

The exact uniform-random crop expectation has mean task gain `-0.00929`, rescue
`0.4576`, and induced-harm rate `0.02398`. Answer-loss therefore provides real
crop-ranking information rather than merely matching random or entropy.

Always calling remains cost-inefficient. Even answer-loss top-one has mean
utility `-0.03270` under the frozen `lambda=0.05`, and the oracle crop has mean
utility `-0.02292`, because only 466 of 14,511 decisions contain any helpful
crop. The scientific opportunity is selective gating, not unconditional crop
selection.

On the fixed descriptive call-rate grid, answer-loss ranking has a strictly
positive 95% source-bootstrap lower endpoint for mean policy utility from 0.5%
through 25% call rate. The largest observed point estimate is at the 10% grid
point: mean policy utility `0.006371` [`0.004616`, `0.008120`], task gain per call
`0.1137`, and induced harm per call `0.00414`. This is descriptive only: the
10% point and its score threshold are not a frozen policy and must not be used
for ScreenQA deployment or formal evaluation.

No entropy call-rate grid point has a positive 95% lower endpoint for mean
policy utility. This directly supports the empirical phrase "beyond entropy",
but not a novelty claim by itself.

## Important failure modes

- There are 125 harmful actions whose target-answer loss improves and 175
  helpful actions whose target-answer loss does not improve. Answer-loss is an
  informative but imperfect proxy, not ground-truth action value.
- Target-answer loss uses a canonical accepted answer that is unavailable at
  deployment. It can be a training signal or audit variable, not an executable
  inference-time gate.
- The original ScreenQA semantic candidate remains failed. This diagnostic does
  not permit another model, feature, threshold, or relaxed risk gate to be
  selected on the same ScreenQA ranker-development bank.
- The result currently covers one target VLM family and scale. A Qwen2.5-VL-3B
  result alone is insufficient for a three-conference generality claim.
- Existing answer-loss/crop-supervision work prevents claiming that loss-gap
  supervision alone is novel. The defensible contribution must include signed
  sibling outcome supervision, joint stop/action selection, and prospective
  harm control.

## Frozen scientific decision

The strong-alignment branch in `proxy-to-outcome-audit-protocol-v1.md` is
activated, with the caveats above. This is a positive mechanistic result and a
motivation for a new method, not a successful deployed method.

The next method must be developed on a different development population and
must predict a deployable pre-action quantity. The preferred hypothesis is a
joint stop/action value model trained with both signed realized task effect and
teacher-forced answer-loss as auxiliary supervision, with an explicit harm
head or constraint. It must never consume the answer target at inference.

Before fitting that method, a new protocol must freeze:

1. the non-ScreenQA development population and its source grouping;
2. the pre-action feature contract and which features are available before a
   crop is executed;
3. the target decomposition for task gain, loss-gap auxiliary supervision,
   harm, and cost;
4. the sole model family, capacity, folds, hyperparameters, and deterministic
   selection rule;
5. entropy, random/fixed crop, exhaustive UG, task-value-only, loss-only, and
   joint-supervision comparators under matched call budgets;
6. a source-disjoint calibration rule and a one-shot untouched confirmation;
7. a pre-result second-backbone check, prioritizing the locally cached
   Qwen2.5-VL-7B or another modern VLM when memory and quota permit.

ScreenQA calibration, formal, reserve, validation, test, and untouched roles
remain sealed. No GitHub push is authorized by this result.
