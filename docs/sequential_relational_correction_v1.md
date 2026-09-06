# Sequential acquisition: single representation correction v1

Status: **frozen before correction results**. Test access remains unauthorized.

The 256/128-state pilot uses a 18,461-dimensional raw frozen-state vector. Its result is
promising only on ChartQA, is negative on DocVQA, and is statistically inconclusive on
HRBench; ChartQA-to-HRBench transfer also collapses to a greater-than-80% acquisition rate.
The dimensionality is underdetermined relative to the number of training states and is the
only permitted correction target.

Exactly one representation change is allowed: replace raw hidden coordinates with a
label-free relational summary. The summary contains the unchanged uncertainty, geometry,
history, and shallow question features; four distribution summaries for each of question,
global, acquired ROI, proposed ROI, current language, current visual, and fused states; and
cosine, mean-product, and RMS-distance for eleven fixed semantic pairs. No outcome,
CONTINUE-side feature, benchmark-specific feature, learned projection, new backbone,
hidden-size change, loss change, extra seed, or threshold search is allowed.

The correction reuses the exact frozen pilot rollout/features and the existing linear/MLP,
three-seed training configuration. It is evaluated at the same matched acquisition rates,
lambda grid, and 10,000 paired source bootstraps. Continue beyond this correction only if
the validation evidence improves cross-domain stability without a trivial policy. If at
least two benchmarks still have learned-minus-strongest-baseline CI lower bounds at or
below zero, the route is NO-GO and no test, RL, 7B, larger head, or further representation
search is permitted.
