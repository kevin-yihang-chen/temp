# CV Method GO / NO-GO

Final decision: **NO-GO at Phase B** (2026-09-06, Asia/Hong_Kong).

This decision applies to the frozen method in
`docs/cv_counterfactual_method_protocol_v1.md`: lightweight
Qwen2.5-VL-3B post-training for a fixed binary STOP/CONTINUE decision using
explicit counterfactual gain preference. It does not claim that every possible
counterfactual learning method is impossible.

## Four required answers

1. **Did Counterfactual Utility improve matched-cost accuracy over the strongest
   uncertainty baseline? No.** At the frozen 25% call rate, ChartQA was
   `-0.781pp` (`95% CI [-5.469,+3.906]`) versus confidence. DocVQA was only
   `+0.062pp` (`95% CI [-0.439,+0.684]`) versus entropy. Neither meets the
   Phase-B `>+1pp` threshold.
2. **Did it improve over the matched Outcome-only control? No.** It was
   `-3.906pp` (`95% CI [-7.812,-0.781]`) on ChartQA and `-0.521pp`
   (`95% CI [-1.681,0.000]`) on DocVQA. Thus the explicit Phase-B stop rule,
   nonpositive deltas on both domains, is triggered.
3. **Is there a consistent positive direction across ChartQA and DocVQA? No.**
   The proposed method is below Outcome-only on both domains and does not form
   a cross-domain advantage over the strongest uncertainty baselines.
4. **Is scaling allowed? No.** Phase C, three-seed confirmation, a new test
   transaction, 7B, RL, continuous boxes and multi-turn acquisition are not
   authorized under this protocol.

## Primary matched-cost result

All non-Answer policies below make exactly 32 CONTINUE decisions among 128
states (`25%`, average incremental visual cost `0.25`).

| Domain | Answer-only accuracy | Strongest uncertainty | Outcome-only | Counterfactual Utility |
|---|---:|---:|---:|---:|
| ChartQA | 46.094% | 49.219% (confidence) | **52.344%** | 48.438% |
| DocVQA | 91.500% | 91.690% (entropy) | **92.273%** | 91.753% |

At `lambda=0.05`, Counterfactual incremental net utility is `+0.01094` on
ChartQA and `-0.00997` on DocVQA. Outcome-only is higher on both (`+0.05000`
and `-0.00476`). Counterfactual useful-call precision/recall is only
`9.38%/18.75%` on ChartQA and `6.25%/25.00%` on DocVQA; unnecessary-call rates
are `90.63%` and `93.75%`.

The full call-rate frontier (`0, .1, .25, .5, .75, 1`) and utility sweep
(`lambda=0, .025, .05, .1, .2`) do not reverse the primary decision. Their
machine report and plots are in
`artifacts/cv-method-v1/phase-b-pilot/evaluation/job-209090/`.

## Training and failure diagnosis

Phase A Job `209085` passed all engineering gates. Phase B Job `209090` then
used the full frozen train banks (256 states per domain), one matched
outcome-independent pass (512 steps), seed 17 and identical architecture,
optimizer and schedule for both arms. Slurm recorded `COMPLETED`, `ExitCode=0:0`,
no restart, 2×RTX 4090 and 11m08s runtime. Peak memory was 9.51 GiB per arm.

The Counterfactual objective receives supervision from only the 63 non-neutral
train pairs: 50 beneficial and 13 harmful; it ignores 449 neutral pairs by
definition. By the end of Phase B it naturally selected CONTINUE for all 256
validation states, and its fixed train audit loss increased from `0.68236` to
`1.64377`. Exact top-count evaluation still supplies a nonconstant ranking and
forces matched cost, but that ranking is inferior to Outcome-only. This action
collapse is a measured method failure, not a pipeline or leakage failure.

No proposed crop was executed by the policy input path; all scores were finite;
the head, vision merger and last language block received gradients and changed;
the two arms used the identical schedule SHA-256
`7620e38321829c3da5a850c40c80e76aa2abe08a00908ecf953307c9e039d434`;
and every report records `test_accessed=false`. Therefore the negative result
cannot be dismissed as an unrun model, unmatched compute, post-action leakage,
or cost mismatch.

Class balancing or alternative treatment of neutral pairs might address the
collapse, but trying it now would violate the pre-registered Phase-B stopping
rule and become a new method search. It is not run in this task.

## Immutable evidence

- Implementation commits: `32bd83c` and Phase-A record `da5fdfe`.
- Frozen Phase-B plan SHA-256:
  `49cdcb5b7556fb986bcca3b7d5ba8fbf65f570d31406b5bc9b95da87d5d764fd`.
- Outcome-only report SHA-256:
  `81722c92ee3fb0b8441ada273b86241b4ba2cd319a2088b5897127396f4ce5e5`.
- Outcome-only selector SHA-256:
  `8233b28670198701ed3c55c5fb7f394d46ec073a7a7b6557b4249ead46672088`.
- Counterfactual report SHA-256:
  `dae1ee0babef7bd89a2b25bf6f801ca08086ab223db7a20c9318cd46e4e6f144`.
- Counterfactual selector SHA-256:
  `9b026f7ed73f33156f8daca5108b12a1f8c4784013fa753f045d1bf7f5511c03`.
- Evaluation report SHA-256:
  `b0008061ce0db16bf257ede408be239d10d6993df3a80f2d498c3c01999605a4`.

## Stop reason

**Counterfactual Utility is worse than Outcome-only on both registered domains,
and it fails the registered transition threshold versus the strongest
matched-cost uncertainty baseline. The supervised hypothesis did not translate
the existing acquisition headroom into a paper-worthy end-to-end policy gain.
Stop without Phase C or scaling.**
