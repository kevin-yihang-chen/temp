# Joint auxiliary action-proposer result v1

Status: registered development result completed on 2026-09-01. The mechanical
decision is **`joint_auxiliary_proposer_not_advanced`**. ScreenQA calibration,
formal, reserve, and untouched roles remain sealed.

## Bound execution

- Job `199850`, one NVIDIA H800, completed in 2 minutes 38 seconds.
- 3,500 DocVQA development sources, 13,580 decisions, 54,320 candidate crops.
- Five whole-source OOF folds, 200 fixed epochs, 20,000 whole-source bootstrap
  resamples, seed `20260904`.
- Registered code revision:
  `6431c65a6f0ace4b53af35d575ce6ea5214340b0`.
- Report SHA-256:
  `689b7471943a24ddd44f5972642d2f3071ee58f009474b308040bf2e758ee932`.
- Model SHA-256:
  `a69a3d1a58e5bbac525035c10b2d76ea9d652b858567ce4191fbec846cf023f3`.
- Outcome-free OOF prediction SHA-256:
  `d73b976b72101f2815dc89fd9d472ac91b680aa195beb032deef116600db572e`.
- Completion SHA-256:
  `9005e7291703e6a36f4e78b23e9a64b699269a93ea32347ebcc23e550d1b7fb1`.

The formal prediction rows contain only identities, method scores, and selected
action identifiers. The implementation-only one-epoch smoke was not used to
change any registered setting.

## Registered decision

All four advancement clauses fail:

1. joint minus task-only source-balanced gain is `-0.0011810`, with 95%
   interval `[-0.0030205, 0.0005712]`;
2. joint gain `-0.0012387` is below frozen factorized gain `0.0014373`;
3. joint helpful-state recovery `0.56770` is below task-only `0.57173`, though
   above frozen factorized `0.51641`;
4. joint induced harm `0.0146189` exceeds task-only `0.0140979` and frozen
   factorized `0.0128565`.

The registered conclusion is negative. The joint shared representation may not
be promoted, and no protected ScreenQA role may be opened from this model.

## What the ablations reveal

The failure is not evidence that teacher loss contains no useful signal. The
`loss_only` distillation ablation has source-balanced top-one gain `0.0025233`,
helpful-state recovery `0.64336`, and induced harm `0.0134485`. It exceeds the
frozen factorized proposal's gain `0.0014373` and recovery `0.51641`, although
its paired gain improvement `0.0010859` has a 95% interval
`[-0.0013943, 0.0036243]` and therefore is not independently significant.

By contrast, sharing one 32/16-dimensional trunk across balanced rescue,
balanced harm, and continuous loss-gap objectives causes negative transfer:
joint is worse than loss-only by `-0.0037620`, with 95% interval
`[-0.0064855, -0.0011498]`. This interval is entirely negative. The task-only
network is also weaker than the incumbent factorized decomposition.

Loss-only and factorized proposals disagree on 64.10% of decisions. Their task
outcomes are equal on 12,979 decisions; loss-only is better on 364 and
factorized is better on 237. This suggests a potentially useful proposal signal
with a modest net edge, but not a successful joint method.

Entropy top crop has still higher descriptive gain and recovery, but it uses
post-action entropy from all candidate executions and is therefore an
exhaustive acquired-evidence comparator rather than a one-call pre-action
proposal.

## Frozen next direction

Do not tune the failed shared trunk or its loss weight on this result. The next
registered branch may retain the successful pre-action `loss_only` proposal but
must decouple it from stopping and harm control:

1. use loss distillation only for `where`;
2. score that pending action with the separately trained, source-held-out
   factorized error/rescue/harm heads for `whether`;
3. compare at an outcome-blind matched call budget;
4. advance only if utility is non-inferior or better while harm does not rise;
5. keep all ScreenQA protected roles sealed until that rule is evaluated and a
   complete deployment refit is frozen.

No GitHub push is authorized by this result.
