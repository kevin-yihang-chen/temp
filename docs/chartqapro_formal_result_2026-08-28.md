# ChartQAPro formal result (2026-08-28)

## Status

The frozen 1,625-question ChartQAPro confirmation **failed** its registered
six-part primary criterion. This result is retained as negative cross-domain
evidence. The formal split is now permanently closed to fitting, threshold
selection, feature selection, prompt changes, action-geometry changes, and any
replacement primary claim.

The run used `Qwen/Qwen2.5-VL-3B-Instruct` at model revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`, rollout-code revision
`d9b35b8e735848872e5ea315cfd56cd0398512a6`, four candidate crops, seed 0,
and the released ChartQAPro scorer. It completed all 1,625 states and 8,125
sibling action records with no empty output.

## Frozen inputs and products

| Product | SHA-256 |
| --- | --- |
| Formal manifest | `5a3ddca2e6476196aac8ad4fa7bc00033f2ac9c39d2011fe21fa070e965b97d4` |
| Frozen factorized model | `5d5c0f781a7141726e786d6ad87b861a6395c489bcf9ad8567a2e9ca825c3330` |
| Rollouts | `d086808e09bddd0b09d0aa4851030fbcc2889cfd6910fa21961a29df1c6bc45e` |
| Rollout provenance | `a15a8e884056e9815bd9af58c6b98236074a38ab4286fd84b06c67df24472dce` |
| Frozen analysis JSON | `6053b4db14043f76b18a810e4c1d5c8da1450ae3e4f4fd6e2535206a42c994e5` |
| Frozen analysis Markdown | `8322672a66a34f65777f694a5c98ce5ba11f1b3eebbf772c673682860fc718cf` |

The analysis implementation was frozen at commit
`849088b` before inspecting the rollout outcomes. Released, paper-spec raw,
and conservative canonical scoring all agree on the primary policy's sign and
decision.

## Registered decision

At visual cost `lambda=0.05`, the frozen primary called a crop on 151/1,625
states (tool rate `0.092923`), gained `0.002363` released score, and obtained
mean utility `-0.002283`. Its question-cluster bootstrap interval was
`[-0.004932, 0.000545]`; the image-cluster interval was
`[-0.004989, 0.000491]`.

| Registered criterion | Passed |
| --- | --- |
| Positive mean utility | no |
| Question-bootstrap utility lower bound above zero | no |
| Image-bootstrap utility lower bound above zero | no |
| Positive released-score gain | yes |
| Lower tool use than unconditional one-crop | yes |
| Lower tool use than exhaustive four-crop | yes |

Therefore the primary confirmation is `FAILED`, not a near-pass or an
inconclusive success.

## Descriptive diagnosis, not model selection

The result does not show that crops lack value. The one-crop random baseline
gained `0.016459` but had utility `-0.033541`; the four-crop oracle gained
`0.084600` at call rate `0.096615` and utility `0.079769`. The frozen gate raised
gain per call from `0.016459` for unconditional random cropping to `0.025433`,
but this remained below the registered cost `0.05`. Among its calls, the
unnecessary-call rate was `0.874172`.

The deployed primary is explicitly
`frozen_factorized_context_uniform_random_expectation`: its source-frozen model
contains an error head, a conditional-rescue head, and a stopping threshold,
but no crop-action head. Conditional on calling, it averages uniformly over the
four candidate crops. This makes both calibration of *when* to call and
selection of *where* to look plausible sources of cross-domain regret. These
observations explain the next development direction; they must not be used to
fit or retest a replacement on ChartQAPro formal.

## Activated preregistered branch

Branch B of `chartqapro_formal_decision_protocol.md` is active:

1. ChartQAPro formal remains negative, descriptive-only evidence.
2. ChartQAPro pilot may join ChartQA and the registered DocVQA/TextVQA
   development sources.
3. The next method directly estimates cost-adjusted value for each candidate
   action and retains answer-now as a zero-cost action, rather than composing a
   source-only gate with uniform-random crops.
4. Selection is by source-group-held-out multi-domain utility. Formal
   DocVQA/TextVQA and paired HRBench outcomes remain unseen until the complete
   model class, candidate set, cost, and decision rule are frozen.
5. High-cost RL remains deferred until this revised method passes a new
   untouched confirmation.
