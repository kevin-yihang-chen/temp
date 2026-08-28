# DocVQA attention action-value secondary preregistration

Outcome: failed on 2026-08-28. See
`docs/docvqa_attention_action_value_formal_result_2026-08-28.md`.

Frozen: 2026-08-28 13:30 Asia/Hong_Kong, before job `190296` completed and
before any DocVQA formal-v2 outcome was evaluated or inspected. Rollout
generation had already started; only checkpoint counts and Slurm state were
observed.

## Role and multiplicity

The context-by-geometry policy in
`docs/docvqa_action_value_formal_preregistration.md` remains the sole primary
analysis and must be evaluated first under its original 95% interval rule. This
document freezes one secondary policy selected after additional development-
only diagnostics. Because two frozen policies will be tested on the same
formal partition, the secondary policy uses a conservative source-bootstrap
97.5% confidence interval. It is confirmatory only if both its mean utility and
the 97.5% lower bound are strictly positive. Otherwise it is directional or
failed evidence and cannot be revised on this partition.

## Frozen development policy

The policy was selected from the outcome-disjoint 824-decision, 200-source
DocVQA development bank using five source-grouped OOF folds. It is a factorized
error/rescue/harm logistic model with `semantic-context` features:

- state head: compact original-image semantic state plus pre-action context;
- action heads: question/global/ROI similarities, candidate geometry, and
  frozen question-to-region attention;
- decision: maximum predicted rescue value minus predicted harm and
  `0.05 * tool_cost`, called only above the frozen margin.

| Artifact or parameter | Frozen value |
|---|---|
| Development rollouts SHA-256 | `4d3d3a33f644d1f5122aabecd47a8168d2dce2db5014692b508ba76ae4ddbe52` |
| Development attention features SHA-256 | `a4055bc8306321c0ca98577407e3d9ed1f4b983474178cb68422702bd6f9407a` |
| OOF report SHA-256 | `a3e1387c04c760c5bac2483da724a36a144f9c5dc21807d4f8e743786e7a3420` |
| Serialized model SHA-256 | `1f8b6cf5d026bcd9921434c1c6ef0c753259d36504dedc040b8145c76bd06ff3` |
| Training code revision | `96076cc0321c1813f6e9a3dad74bdbf27ab888f7` |
| Feature mode | `semantic-context` |
| OOF folds / seed | 5 / `20260828` |
| Regularizer / call margin | `alpha=10.0` / `0.03311381598522578` |
| Development utility | `+0.00460893`, 95% source CI `[+0.0000557, +0.0102646]` |
| Development gain / tool rate | `+0.00655068` / `3.8835%` |

No threshold, regularizer, feature layer, head pooling, crop set, or cost may be
changed after this freeze.

## Label-free formal feature contract

Formal features are produced only after the already registered rollout job
finishes. They may read state/action IDs, original image, question, baseline
entropy, candidate boxes, and fixed tool costs. They must not serialize or
validate against answers-after-action, correctness-before/after, score deltas,
or post-action entropy.

The frozen chain is:

1. one-pass Qwen original-image ROI features, with `--exclude-outcomes`;
2. exact question-token final hidden-state mean conditioned on the original
   image, never on a candidate crop;
3. mean attention from the final four language layers and all heads, from exact
   question tokens to original-image visual tokens, ROI pooled and normalized
   over the four fixed candidate boxes;
4. `scripts/audit_label_free_semantic_features.py`, which must report no
   forbidden outcome fields;
5. frozen evaluation with `--require-label-free-features` and exact model,
   rollout, and feature hashes.

This offline replay represents signals available from the baseline prompt
forward pass and never executes candidate crops. The paper must separately
report replay latency and the engineering path for harvesting the tensors from
the baseline pass; visual-tool utility continues to charge one unit for each
executed crop.

## Formal target and exact analysis

The target is the same outcome-unseen formal-v2 manifest already frozen at
`data/cross-benchmark-v1/docvqa-formal-v2/manifest.jsonl`, SHA-256
`9ceb28d05df5feecedf6cf61fbbb27ce281b94dd027e5d6d6da43ddc091081ac`:
1,608 decisions from 400 source documents, with zero state, source, or decoded-
RGB overlap with development. Job `190296` produces 8,040 sibling records.

The secondary metric is mean ANLS gain minus `0.05 * mean tool calls`. Evaluate
with 10,000 source-cluster bootstrap resamples and seed `20260828`. The report
must bind the exact serialized model, rollout, and label-free feature hashes.
The model is evaluated exactly once after the primary context policy report is
locked. Secondary diagnostics include ANLS gain, tool rate, gain per call,
unnecessary-call rate, correct-stopping rate, and predicted-value distribution;
they cannot alter the pass decision.
