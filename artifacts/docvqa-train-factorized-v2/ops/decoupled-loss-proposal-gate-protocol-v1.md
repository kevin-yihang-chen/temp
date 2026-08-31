# Decoupled loss-distilled proposal and harm gate protocol v1

Status: frozen on 2026-09-01 after the registered joint shared-trunk failure
was opened, but before the forced-action factorized gate scores below were
computed. This is a new development diagnostic on already opened DocVQA data.
It cannot revise any prior conclusion, and all ScreenQA protected roles remain
sealed.

## Motivation and single hypothesis

The registered joint model shows statistically negative transfer from sharing
one representation across task and loss objectives, while its separately
trained `loss_only` ablation has higher point gain and helpful-state recovery
than the incumbent factorized proposal. Test one decoupled hypothesis:

> use distilled answer-loss only to choose **where**, then use the existing
> source-held-out factorized error/rescue/harm model to decide **whether** to
> execute that pending crop.

No new feature, neural architecture, loss weight, alpha, fold assignment, or
teacher target may be tried in this diagnostic.

## Frozen inputs

- DocVQA rollouts SHA-256:
  `9109d5c896e59959937c2171e897dd81341237ff51fc5745fb4c56325ddd8fc3`;
- label-free semantic features SHA-256:
  `05b3aaeabeef84993787c0ce676d175ab51a95bae34cabb5e950b0bd4f906686`;
- registered joint report SHA-256:
  `689b7471943a24ddd44f5972642d2f3071ee58f009474b308040bf2e758ee932`;
- registered full-refit model SHA-256:
  `a69a3d1a58e5bbac525035c10b2d76ea9d652b858567ce4191fbec846cf023f3`;
- registered outcome-free OOF predictions SHA-256:
  `d73b976b72101f2815dc89fd9d472ac91b680aa195beb032deef116600db572e`;
- incumbent factorized model/report SHA-256:
  `ce8e534cfcf5bf4f08e565b4b88112c1768d6b7d9cc0a1eaa8226ffc424b697e`
  / `fd17ef0863ea21ad6eae646e08deb936da449f5ee05dda9cb94d6d1687aef888`.

The input inventory must reproduce 3,500 sources, 13,580 decisions, four crops
per decision, and exactly one `loss_only_action_id` per decision. OOF prediction
rows must not contain correctness, target, reward, gain, harm, answer, oracle,
or post-action entropy fields.

## Frozen forced-action gate reconstruction

Reconstruct the incumbent factorized heads exactly:

- five whole-source folds from `_source_folds`, seed `20260829`;
- alpha `1.0`, feature mode `hybrid-context-semantic`;
- error/rescue/harm head seed `20260829 + fold`;
- equal-domain, equal-source, equal-row weighting;
- rescue and harm magnitudes and cost `lambda=0.05` from the existing code.

For every decision, do **not** maximize the factorized value across crops.
Instead, take its registered OOF `loss_only_action_id` and compute the expected
net value of that single pending action with factorized heads trained without
the decision's source. This produces one decoupled OOF gate score per decision.

The incumbent comparator is its original OOF top action and original OOF net
value from the same reconstructed heads. Both predictions must exclude each
source. Reproduction must match the frozen incumbent report before evaluation.

## Outcome-blind matched budget

Set the primary call count to the incumbent OOF policy's frozen 225 calls
(`0.016568483063328424` pooled rate). For each score family independently:

1. sort distinct scores from high to low;
2. retain complete tie groups;
3. select the threshold minimizing absolute distance to 225 calls;
4. break equal distance toward fewer calls, then the higher threshold.

This threshold construction may use only scores and identities. It may not use
correctness, gain, utility, harm, entropy, or teacher targets. Report achieved
pooled and source-balanced call rates and any unavoidable tie mismatch.

## Registered evaluation

At the frozen matched budgets, report source- and question-balanced gain,
utility, call rate, gain per call, induced harm, negative-value call mass,
helpful-call precision/recall, action disagreement, and gate disagreement.
Compute decoupled-minus-incumbent source-balanced utility using paired
whole-source bootstrap with 20,000 percentile resamples, seed `20260905`, and a
two-sided 95% interval.

Also report proposal-only loss-distilled minus factorized top-one gain as a
secondary diagnostic. It is not the deployment primary and cannot change the
matched-budget rule.

## Mechanical advancement rule

Advance the decoupled full refit to a separately frozen ScreenQA calibration
candidate only if all clauses hold:

1. decoupled source-balanced utility is at least incumbent utility plus
   `0.00025`;
2. the paired 95% lower endpoint for decoupled minus incumbent utility is above
   `-0.0005` (development non-inferiority margin);
3. decoupled source-balanced gain per call is strictly higher;
4. decoupled induced harm and negative-value call mass are each no greater
   than the incumbent at the matched budget;
5. loss-distilled proposal helpful-state recovery is strictly higher than the
   incumbent proposal;
6. every input, reconstruction, OOF exclusion, score-schema, and threshold
   audit passes.

Failure yields `decoupled_loss_proposal_gate_not_advanced`; do not open
ScreenQA calibration. Passing only authorizes serialization and freezing of a
candidate and its finite risk-threshold sequence. It is not independent
validation or permission to skip prospective calibration.

Every submitted task uses all-state email notifications to
`yihangc@connect.hku.hk`. No GitHub push is authorized by this protocol.
