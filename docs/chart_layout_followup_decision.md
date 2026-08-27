# Chart-layout follow-up decision rule

## Freeze point

This decision rule is frozen before the outcome of the 2,137-state
image-disjoint chart-layout confirmation is inspected. It governs whether a
second chart-layout rollout is collected on the separate 4,500-image ChartQA
train replication target.

## Go/no-go rule

Proceed only if the registered chart-layout confirmation primary passes all of
the following for treatment minus the four-crop UG baseline under exact
uniform-random one-crop evaluation:

- positive paired utility point estimate;
- 95% state-bootstrap lower endpoint above zero; and
- 95% image-bootstrap lower endpoint above zero.

If any condition fails, do not launch the 4,500-image chart-layout treatment.
Record action proposal as unresolved and retain the UG rollout as the stopping
replication only.

## Conditional follow-up estimands

If the go rule passes, freeze and collect four chart-layout candidates on the
already frozen 4,500-state target using the same Qwen revision, prompt, seed,
visual cost, and pixel budget as its UG rollout. Before either target outcome is
inspected, register:

1. the paired chart-layout-minus-UG difference for unconditional exact
   uniform-random one-crop accuracy gain/utility;
2. the paired difference after applying the byte-identical frozen factorized
   stopping gate and absolute threshold; and
3. the absolute utility and state/image intervals of the composed stopping plus
   chart-layout policy.

The first estimand confirms proposal quality; the second isolates whether it
improves the end-to-end policy at equal calls and cost; the third checks whether
the composed deployable policy has positive utility. All require both state- and
image-cluster intervals. Fixed candidates, entropy search, oracle VOI, and
strata are secondary. No target threshold tuning is permitted.
