# Externally fixed pairwise signed-value implementation freeze v1

Status: frozen on 2026-09-01 before any DocVQA candidate fit or score was
opened. This document binds the sole implementation allowed by
`external-pairwise-signed-value-protocol-v1.md`.

## Bound implementation

- core module SHA-256:
  `d1b094d0129b85b2ab74ac0f12a9236adf5b713709c9d5af0dd3b65371096653`;
- runner SHA-256:
  `7e6ed4df863f617c3129a25ef5b023ce13c25309c1d9366e9481ff2043168cb3`;
- tests SHA-256:
  `0663e0a1ad63845c16a2f399030aa75e4c98b216316c2cfd9fc9acf2a75e7dad`;
- protocol SHA-256:
  `f09e742fb4b30058712b22134fdded89f84c3446e7be454544494f1aa252794a`.

The implementation imports the already tested pairwise feature preparation,
source-balanced fitting, call-feature construction, ridge fitting, and linear
serialization primitives. It supplies a new exact protocol layer that:

1. uses the transferred singleton settings `semantic-context`, `C=0.01`, and
   `alpha=100`;
2. performs five outer whole-source folds and five inner whole-source folds on
   each outer-training population;
3. breaks exact action-score ties toward the lexicographically smaller action
   ID;
4. trains each outer call head only on actions selected by inner source-held-out
   rankers;
5. trains the full-development call head only on outer-OOF selected actions;
6. matches the candidate and audited incumbent to exactly 225 pooled calls
   without outcomes;
7. serializes no correctness, answer, gain, harm, reward, target, post-action
   entropy, oracle, or utility field; and
8. evaluates the fixed 20,000-resample source bootstrap and mechanical pass
   rule only after score serialization is complete.

## Completed pre-fit verification

- Python compilation passed for module, runner, and tests.
- Seven focused pairwise/scaled-action-value tests passed.
- The full repository test suite passed with only the existing optional-runtime
  skips.
- Runner help and `git diff --check` passed.
- A real-input, no-fit preflight verified 3,500 sources, 13,580 decisions,
  54,320 ZOOM rows, exactly four aligned candidates per decision, 60 state
  features, 46 action features, 110 call features, and
  `outcomes_included=false` in semantic storage.
- Synthetic nested-OOF tests verify complete prediction coverage, zero outer
  source overlap, complete inner source-held-out action coverage, finite
  predictions, deterministic smaller-ID ties, and rejection of outcome-bearing
  incumbent score rows.

The formal runner must verify every frozen input and implementation hash before
fitting and refuse an existing output directory. Any engineering recovery may
change only code required to satisfy this protocol, must receive a new hash and
implementation audit, and cannot alter the model, features, folds, seeds,
regularization, call budget, bootstrap, or pass rule.

No ScreenQA or protected input is authorized. Every submitted compute task
must use all-state email notifications to `yihangc@connect.hku.hk`. No GitHub
push is authorized.
