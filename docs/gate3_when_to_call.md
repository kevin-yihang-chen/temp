# Gate 3 when-to-call integration boundary

## Scope

The independently confirmed result supports adaptive stopping, not spatial
action selection. The first Gate 3 adapter therefore exposes exactly two
decisions:

```text
ANSWER
CALL_VISUAL_TOOL
```

It does not emit a bounding box, quadrant, crop index, or tool argument. A
separate fixed or randomized controller must supply the visual action after a
call. This prevents the failed/unresolved where-to-look hypothesis from being
silently folded into the confirmed when-to-call result.

## Frozen runtime adapter

`FrozenWhenToCallGate` loads the confirmed factorized model, optionally checks
its byte-level SHA-256, and accepts only `PreActionGateInput`:

```python
from beyond_entropy.stopping import FrozenWhenToCallGate, PreActionGateInput

gate = FrozenWhenToCallGate.load(
    "artifacts/gate2-transfer-chartqa-vstar/factorized-context-v2-quantile/model.json",
    expected_sha256="5d5c0f781a7141726e786d6ad87b861a6395c489bcf9ad8567a2e9ca825c3330",
    registered_lambda_cost=0.05,
)
decision = gate.decide(
    PreActionGateInput(
        state_id="example-1",
        question="What is the largest value?",
        answer_before="42",
        entropy_before=0.31,
        normalized_token_entropies=(0.28, 0.34),
    )
)
```

The runtime input has no correctness labels, post-action entropy, candidate
outcomes, bbox, or ground truth. The returned decision deliberately has
`spatial_action_id=None`.

As an integration check, the adapter was replayed over the completed 4,500-state
independent replication using the byte-verified frozen model. It emitted 288
tool calls (6.4%), exactly matching the registered analysis, and all 4,500
decisions retained `spatial_action_id=None`. This is a parity check, not a new
experiment or additional model result.

## Matched-compute ablation

The initial VTool-R1/training-v2 experiment should compare:

1. outcome-only training with the existing tool policy; and
2. the same training budget with the frozen binary call/answer decision exposed
   at the visual-tool token boundary.

Keep model revision, target data, number of optimizer steps, sampled
trajectories, and tool budget matched. Report answer accuracy, tool-call rate,
registered utility at `lambda=0.05`, and token/latency cost. Any localized
visual-action advantage must remain disabled in this experiment. This module is
an interface scaffold and does not itself constitute an RL result.
