# InfographicVQA DECAR nested-OOF implementation freeze v1

Status: implementation-frozen on 2026-09-01 before the corrected pilot rerun
and full rollout. No InfographicVQA pilot or full scientific endpoint was read
while constructing or testing this implementation.

## Frozen implementation

```text
937dcd29deed4e671b4969a30b8521b685c326619fbf907f673240853b25ac3d  src/beyond_entropy/qwen_backend.py
5729228f02ac5fa316f9a8549acedec0643c14ef555455a8fa61b20b79c260ce  src/beyond_entropy/infographicvqa_decar.py
3731934019d99fb28990a7625e2616957cebb9d0c5438d658bb7ef67890aed55  scripts/fit_infographicvqa_decar_oof.py
2b502bf40197a69f5c182d8e7e35e7d24aadcab91b4580f4a16bc55bb54d3cf8  scripts/smoke_infographicvqa_decar_torch.py
0001e6f48841d82b82461766f275d975bf0cbb706230a4e18f06f7f4da44bc2b  scripts/slurm_infographicvqa_decar_torch_smoke.sh
f580c2a77a64de52e9802a65f88939fd0507634e3593a79dd2870820d757661e  tests/test_infographicvqa_decar.py
ede9e208b7d7b56ead155fcd389e47a5f81d332f6863f8c4b739cb7e207007ef  tests/test_qwen_runtime.py
```

## Fit and exclusion contract

For each of five outer source-held-out folds, the runner fits four inner
source-held-out `where` models, cross-fits the selected action/gap/margin for
every outer-training row, refits `where` on all outer-training sources, and
fits the registered outer `when` gates. It executes exactly the four frozen
variants: `decar`, `task_value_only`, `loss_only`, and `no_harm_head`.

The registered seed equation is implemented literally. Inner where fits use
`inner_fold=0..3`; the outer where refit uses the pre-endpoint pseudo fold
`inner_fold=4`; and the outer when fit uses pseudo fold `inner_fold=5`.
Variant indices are `0,1,2,3` in the registered variant order. `loss_only`
shares the primary loss-distilled where fit and therefore creates no separate
neural fit with index 2. `no_harm_head` likewise shares the primary where fit
but uses variant index 3 for its distinct binary when fit.

Every standardizer, target standardizer, class mass, rescue/harm magnitude,
and neural fit uses only the applicable training sources. The runner fails on
any missing/duplicated fold context, source overlap, non-finite value, missing
class, incomplete feature/NLL/action coverage, generated-token statistic
misalignment, or changed full population.

Predictions are serialized separately from the evaluation audit. Prediction
rows contain only identifiers, outer fold, selected action, predicted
gap/margin, probabilities, predicted delta, score, and eligibility. They never
contain correctness, task delta, post-action entropy, teacher NLL, target
answer, or another outcome. Model-state SHA-256 digests and zero-overlap source
counts are retained in the audit.

## Verification

- Complete repository regression suite passed.
- Focused mypy, Python compilation, Black formatting, shell syntax, and
  whitespace checks passed.
- The qwen-vl CPU smoke passed under PyTorch `2.4.0+cu121`.
- Exact-hash H800 nested-OOF smoke job `200068` completed in five seconds,
  with zero restarts and exit code zero. It exercised all 65 registered neural
  fits on synthetic source-disjoint folds and read no InfographicVQA task
  endpoint.
- All-state email was configured for `yihangc@connect.hku.hk`.

This freeze authorizes implementation verification and the registered
corrected pilot rerun only. It is not a scientific success claim, does not
authorize opening validation/test, and does not authorize a GitHub push.
