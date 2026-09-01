# Minimum-rank consensus gate implementation freeze v1

Status: frozen on 2026-09-01 after implementation and synthetic tests, but
before constructing or evaluating any real DocVQA consensus score. ScreenQA
and every protected role remain sealed.

## Bound implementation

- evaluation module SHA-256:
  `348e08d3d1038ab760227af7129311f85020b101805826693e6f705d86e9cb3f`;
- command-line runner SHA-256:
  `a2f8ad151d24cfb1c5c6baa45bbef529a78f11cba628a6c0bece91a81bd3b554`;
- focused test SHA-256:
  `ec96396537d699d5b73093ce3db7aed8196aa2c6257ada3ac253c1edfb25e77f`;
- frozen protocol SHA-256:
  `e5424107b0a92a15364e9cd2137ffbe29bb932bbb966ff7379de8f3e6c59f591`.

The focused consensus, cost-sensitive, and decoupled tests pass. The complete
repository test suite passes with only the existing optional-runtime skips.
Compilation, runner argument parsing, focused mypy with imported modules
skipped, and `git diff --check` pass. Full import-following mypy still exposes
five pre-existing errors in `decoupled_loss_gate.py`; none is introduced by
this branch, and focused mypy reports no issue in either new source file.

## Bound scientific inputs

- sibling rollouts SHA-256:
  `9109d5c896e59959937c2171e897dd81341237ff51fc5745fb4c56325ddd8fc3`;
- cost-sensitive report SHA-256:
  `941181e02e48352ea4f10ca20f9b10b3ed85afa790ba7d25b2138c0be984c464`;
- cost-sensitive score report SHA-256:
  `5ae3623879a96c46ac5ffce40b4acc0b84b53e30c58dc2fe2d3f8006fbedf83d`;
- cost-sensitive model SHA-256:
  `762005d1124ea68ca993f5a58e56317bdec0e026f24751504b8d5e295c3e6bb1`;
- cost-sensitive OOF score rows SHA-256:
  `9512d000ca3c2567fd36711f20eca10619acd390622ccf1f05cc9930145dcaec`;
- incumbent report SHA-256:
  `fd17ef0863ea21ad6eae646e08deb936da449f5ee05dda9cb94d6d1687aef888`;
- incumbent model SHA-256:
  `ce8e534cfcf5bf4f08e565b4b88112c1768d6b7d9cc0a1eaa8226ffc424b697e`;
- audited incumbent OOF score rows SHA-256:
  `e82ea5b94f45c7e5c961ec61248b9cca514897a0b2b291e5c9110fd1c4673cc3`.

The runner fails closed on every hash before constructing the consensus score.
It also requires the preceding cost-sensitive decision to remain
`cost_sensitive_direct_action_value_not_advanced`.

## Fail-closed implementation properties

The implementation accepts only the exact registered outcome-free schemas and
requires 3,500 sources, 13,580 decisions, four valid actions per decision,
complete unique identities, exact source alignment, exact reproduction of the
incumbent fields embedded in the cost-sensitive artifact, and exact
reproduction of both frozen 225-call sets.

For each complete raw-score vector, equal values receive the same empirical
percentile `count(score_i <= score) / 13,580`. The candidate score is exactly
the smaller of the two percentiles, while its action is exactly the frozen
cost-sensitive action. A complete-tie threshold must yield exactly 225 calls;
otherwise evaluation stops before reading the outcome metrics. No fit, mixture
weight, fallback, tie split, alternate call budget, or ScreenQA input exists.

The output contract stores both rank lookup tables and only the registered
outcome-free per-decision fields. Evaluation uses 20,000 whole-source bootstrap
resamples with seed `20260916` and the unchanged advancement rule. Every
submitted compute task must email `yihangc@connect.hku.hk` for all state
changes. No GitHub push is authorized.
