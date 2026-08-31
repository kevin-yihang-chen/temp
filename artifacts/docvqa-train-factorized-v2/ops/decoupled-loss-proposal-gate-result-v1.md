# Decoupled loss proposal plus incumbent gate result v1

Status: completed on 2026-09-01. Registered decision:
**`decoupled_loss_proposal_gate_not_advanced`**. No ScreenQA protected role was
opened.

## Bound result

- Completed job: `199856`, one H800, 2 minutes 21 seconds, `ExitCode=0:0`.
- Population: 3,500 opened DocVQA development sources and 13,580 decisions.
- Both policies call exactly 225 questions by outcome-blind tie-preserving score
  order.
- Report SHA-256:
  `c12ff60ac157519cd5e9a8dfce0639ba8a408902c50e6a57faa3106304b4fd60`.
- Score report SHA-256:
  `efbfcd06d89e16d105d3e5586b6c7cc1ab08940123d1227cd65bca273eff22f1`.
- Outcome-free scores SHA-256:
  `e82ea5b94f45c7e5c961ec61248b9cca514897a0b2b291e5c9110fd1c4673cc3`.
- Completion SHA-256:
  `9fe5cad70b9c8c5fcb75c68fe0e81ff0aa9e697b6a1872db5f14e4685faa00c4`.

The score audit reproduces the incumbent 225-call set and frozen pooled gain,
utility, and call rate. Thresholds use no outcome. Serialized score rows contain
only identities, actions, scores, and call Booleans.

## Mechanical decision

Only proposal recovery and audits pass. The loss-distilled proposal recovers
64.34% of helpful states versus 51.64% for the incumbent proposal, but the old
factorized gate does not exploit that improvement:

| Source-balanced metric | Incumbent | Decoupled |
|---|---:|---:|
| Utility | 0.00317324 | 0.00247865 |
| Gain | 0.00397661 | 0.00332560 |
| Gain per call | 0.24750 | 0.19633 |
| Induced harm | 0.00034156 | 0.00042958 |
| Negative-value call mass | 0.00977110 | 0.01082291 |
| Helpful-call precision | 0.39441 | 0.36106 |

The paired decoupled-minus-incumbent utility difference is `-0.00069459`, with
95% whole-source interval `[-0.00134406, -0.00010288]`. The entire interval is
negative. Four performance clauses fail, so this composition cannot advance.

## Diagnosis

Proposal actions disagree on 64.10% of decisions, but gate decisions disagree
on only 0.353%. The incumbent factorized score was learned while optimizing
over all crops and then taking its own argmax. Applying it to a different
loss-distilled pending action creates a selection mismatch: the gate's high
score tail remains aligned with its own proposals, not with the loss proposer's
additional helpful actions.

This closes the naive reuse route. It does not invalidate the loss proposer;
it shows that a gate must be trained on the distribution of actions that the
loss proposer actually emits.

## Frozen next direction

Train one low-capacity proposal-conditioned rescue/harm gate on the registered
OOF loss actions. Each gate fold must exclude the test source, and every
training action must itself come from a loss proposer that excluded its source.
Keep the loss proposer fixed; do not tune its architecture or loss. Compare at
the same 225-call budget and retain the existing utility, harm, paired interval,
and audit rules before any ScreenQA activation.

No GitHub push is authorized by this result.
