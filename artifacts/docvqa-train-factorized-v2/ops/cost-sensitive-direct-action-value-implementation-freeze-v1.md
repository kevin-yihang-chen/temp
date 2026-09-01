# Cost-sensitive direct action-value implementation freeze v1

Status: frozen on 2026-09-01 before any registered candidate fit or score was
opened. This document binds the sole implementation allowed by
`cost-sensitive-direct-action-value-protocol-v1.md`.

## Bound implementation

- core module SHA-256:
  `d9764456a71017d48bc98553c49589bfe4ad76820a5b6f5a068a90b8e8c3dd0d`;
- runner SHA-256:
  `42890bfbd24929f4d0ec36535a15e1c59001f9f63edbf5484e729df1f024df05`;
- tests SHA-256:
  `d0a70aa13568d92cc16473fb6cf85231cc5f6bccb0a5f7e2e93e7b8981c3741f`;
- protocol SHA-256:
  `d803f09ec795b54ac88127c464059b63e76e3e45ab29f3d1cd2a98458ecd2dad`.

## Implemented invariants

The implementation:

1. verifies the exact 3,500-source, 13,580-decision, 54,320-action population
   and all registered utility/gain class counts before fitting;
2. constructs only the frozen 46-dimensional label-free semantic-context
   action features;
3. gives every source equal total training mass and distributes that mass
   within source in proportion to absolute net utility;
4. uses no class balancing, class resampling, hyperparameter selection, or
   alternate target;
5. fits one scaler and one `C=0.01` logistic head per whole-source OOF fold;
6. selects among all four actions by the raw decision score with smaller-ID tie
   breaking;
7. matches exactly 225 calls without outcomes, reproduces the incumbent call
   set and pooled metrics, and applies the registered 20,000-source bootstrap;
8. serializes one full-development refit and outcome-free OOF score rows; and
9. fails closed on hash, population, dimension, source exclusion, weight mass,
   convergence, score, call-count, reproduction, or leakage disagreement.

## Completed pre-fit verification

- Module, runner, and tests compile.
- Nine focused current-branch tests passed, including equal source mass,
  within-source utility weighting, zero-utility rejection, source-disjoint OOF
  coverage, finite scores, and deterministic tie breaking.
- The full repository test suite passed with only existing optional-runtime
  skips.
- Runner help and `git diff --check` passed.
- The real-input no-fit preflight reproduced 54,320 rows, 46 features, 1,442
  positive and 52,878 negative net-utility targets, total weight 54,320, 3,500
  sources, utility range `[-1.05, 0.95]`, and per-source mass equal to 15.52
  within floating-point tolerance.

The formal runner must verify every frozen input and implementation hash before
fitting and refuse an existing output directory. Engineering recovery may only
restore compliance with this protocol and must receive new hashes; it may not
change the feature family, target, cost, weighting, C, folds, seed, call budget,
bootstrap, or pass rule.

No ScreenQA or protected input is authorized. Every submitted compute task
must use all-state email notifications to `yihangc@connect.hku.hk`. No GitHub
push is authorized.
