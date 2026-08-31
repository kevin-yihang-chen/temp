# DocVQA proxy-to-outcome cross-domain implementation contract v1

Status: frozen on 2026-08-31 before any DocVQA ranker-development
teacher-forced answer likelihood was computed. This contract resolves the
implementation details of `proxy-to-outcome-cross-domain-protocol-v1.md`.

## Bound implementation

- Code revision:
  `dabb6973e01fd24266f828a3c227cd42965557cf`.
- Scorer `scripts/score_visual_action_answer_nll.py` SHA-256:
  `d278b8cd50a58133d6f512467dce8b53a38a690ade3e874b9721c61adabe523d`.
- Answer-likelihood module SHA-256:
  `10c2b647b6ebbc036d6ce06b046521476b4f3d26e73e66b63b7d3f32382b51e4`.
- Merger `scripts/merge_visual_action_answer_nll.py` SHA-256:
  `4e5c8f2a97e9bdfed835f592e6cc9e52138134e4b6d1cfcd855c013b05f5974d`.
- Analysis module SHA-256:
  `7ad2fe4a710e60ca3d1d7f69584c9344c2eee6533e176ddb5e28063b16dae5a4`.
- Analyzer CLI SHA-256:
  `0147a7215ac4956eb908322cce880512e6961ee8ba1cf6ce4321c5084c22e266`.

The job orchestration commit may differ only by adding the frozen protocol,
this contract, tests, and Slurm wrappers. The component hashes above may not
change during scoring, merging, or analysis.

## Sharding and merge

- Assign sorted complete decisions by zero-based position modulo four.
- Shard indices must be exactly 0 through 3 and share the same manifest,
  rollout, model revision, target rule, numerical measurement contract, and
  accelerator class.
- Atomic checkpoints contain only complete one-ANSWER/four-ZOOM decisions and
  may resume only an exact prefix with the same configuration hash.
- Merge exactly four shards into `13,580` decisions, `67,900` rows, and `3,500`
  source groups. Sort by `(state_id, replicate_id)`, then ANSWER first and
  lexicographic `action_id`.
- Verify every shard output hash and provenance hash. Reject duplicates,
  missing actions, raw targets, nonfinite NLL, negative NLL, identity drift, or
  a non-4090 accelerator name.

## Point estimands and uncertainty

- Compute correlations across all `54,320` ZOOM actions. Pearson uses raw
  values; Spearman uses exact average ranks for ties.
- Maximize each proxy within decision for top-one selection and break ties by
  lexicographically first `action_id`.
- Random is the exact uniform expectation over four crops, not a Monte Carlo
  sample. Oracle maximizes signed task gain with the same tie-break.
- A helpful decision contains at least one positive-gain crop. Induced harm is
  negative signed task gain. Unnecessary execution is realized utility at most
  zero.
- Convert fixed call rates to the nearest integer with half rounded upward and
  at least one call. Rank by proxy score, then
  `(state_id, replicate_id, action_id)`.
- Whole-source bootstrap uses 2,000 resamples, seed `20260901`, exact integer
  source multiplicities, weighted sufficient statistics for Pearson, and exact
  weighted midranks for Spearman.
- Grid intervals condition on the full-bank descriptive ranking rather than
  reselecting top-k inside each resample. State this in the report.

## Reporting and outcome isolation

- Study label: `DocVQA ranker development`.
- Scientific status: `retrospective cross-domain replication on opened DocVQA
  ranker-development data; not candidate selection or independent validation`.
- Interpretation boundary: `Only the opened DocVQA ranker-development manifest
  and sibling outcomes are inputs. Existing DocVQA calibration/formal results
  and every ScreenQA protected role are excluded. Thresholds are descriptive
  only and cannot revise either prior candidate branch.`
- Emit report JSON, Markdown, and a completion record binding score, protocol,
  contract, report, Markdown, analysis-module, and code-revision hashes.
- Record that candidate search was not reopened and no calibration, formal,
  reserve, validation, test, or other protected-role input was used.
