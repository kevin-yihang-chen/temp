# InfographicVQA DECAR nested-OOF evaluation freeze v1

Status: frozen on 2026-09-01 while full official-train sibling generation job
`200130` was still running and before any full-train task, teacher-NLL, policy,
ablation, or baseline endpoint was inspected. Official validation and test
remain absent and sealed.

## Bound scientific inputs

```text
d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342  infographicvqa-decar-method-protocol-v1.md
7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6b8cdb6d0af5a4da60  outer-folds.jsonl
8b4977776841794185eedd8cbbfdb8e23ec2c44cc0be9931cc2e672b232f344c  inner-folds.jsonl
0916a6b5a32e15c4f5b3bf920e1ecd4f304aeb97ae186e0e3e846391e2304203  full image-manifest.jsonl
```

The OOF worker receives the completed rollout, answer-NLL, label-free feature,
strict input-audit, and generation-execution hashes only after generation
finishes. It fails before fitting unless the generation record proves exact
23,946-question, 2,204-source, 4,406-image, five-action coverage, complete
generated-token statistics, no predictions, no endpoint-based selection, and
no validation/test input.

## Frozen fit and evaluation implementation

```text
5729228f02ac5fa316f9a8549acedec0643c14ef555455a8fa61b20b79c260ce  src/beyond_entropy/infographicvqa_decar.py
608ec39d15a0f96da4465dfd1331eae5838f534df412991e4d9ec303a5ef6795  src/beyond_entropy/infographicvqa_decar_evaluation.py
3731934019d99fb28990a7625e2616957cebb9d0c5438d658bb7ef67890aed55  scripts/fit_infographicvqa_decar_oof.py
995332a67049548c68fe062f381f30f6673f90dd678b15ced7ca9b1db413c231  scripts/evaluate_infographicvqa_decar_oof.py
a8cd00cea342158f89a449938e825104f38e3456af89eeb5ab4b7fb97a5c8bf6  scripts/slurm_infographicvqa_decar_oof_h800.sh
87534145c318379c4c4f260d8b54444405bc19541cc4d3ba9c25749710300b1a  scripts/submit_infographicvqa_decar_oof_h800.sh
f580c2a77a64de52e9802a65f88939fd0507634e3593a79dd2870820d757661e  tests/test_infographicvqa_decar.py
4d07b64f8eb86293005d2640d36e3efa50be237f1d460ab5eb45bd59056ce7b8  tests/test_infographicvqa_decar_evaluation.py
```

The existing nested-OOF fitter executes all 65 registered deterministic neural
fits for `decar`, `task_value_only`, `loss_only`, and `no_harm_head`, using 200
epochs and the frozen source-exclusion and seed contracts. Prediction rows are
written and hashed without outcomes before evaluation joins the sibling bank.

## Operating points and comparator accounting

The primary DECAR curve uses nominal question-balanced call rates `0.005`,
`0.01`, `0.02`, `0.05`, and `0.10`. It takes the ceiling nominal count among
eligible rows and retains every exact boundary-score tie. The three learned
ablations are evaluated at the resulting identical call count using their
registered score/gap/state ordering. An insufficient eligible ablation count
fails that operating point's audit.

Random and fixed `ug-grid-00` use one common answer-now-entropy threshold at
the primary count. The threshold retains complete ties and must reproduce the
count exactly. Entropy-gated UG similarly targets `floor(primary_calls/4)`,
executes all four crops on every called decision, selects the minimum-entropy
answer, and pays four costs. A tie-induced mismatch is retained and reported
but prevents qualification. Charged exhaustive UG executes four crops on all
questions. The two task-oracle references choose the best-ANLS crop with
one-crop cost, respectively with and without oracle stopping; neither is a
deployable advancement comparator.

Uniform random is the exact expectation over all four actions and contains no
Monte Carlo crop draw. Costs are `0.05` per executed crop. Source-balanced
statistics first average questions within each source and then average
sources. Normalized exact accuracy is the indicator that the DocVQA ANLS score
equals one, which is equivalent to normalized prediction/reference equality
under the frozen scorer.

At every operating point, the evaluator reports question- and source-balanced
baseline/final ANLS, ANLS gain, baseline/final exact accuracy, exact-accuracy
gain, utility, executions, calls, gain per call, helpful-call precision,
helpful-state recovery, induced-harm magnitude, harmful-call mass,
negative-utility call mass, action-selection regret, oracle-stop regret,
entropy disagreement, and SCGR. Raw calls and distinct called sources are
also retained.

For every primary point, retain a failure decomposition even when the
advancement rule fails: action-choice regret, gate false-positive mass and
negative utility magnitude, gate false-negative mass and missed positive
utility, source call HHI, top 1/5/10-percent source call concentration, and
source-utility quantiles. This analysis is descriptive only and cannot alter
the frozen candidate or advancement rule.

## Bootstrap and advancement

Generate exactly 20,000 iid whole-source bootstrap resamples with NumPy
`default_rng(20260917)`, sorted source order, and int32 indices. Save the exact
`[20000, 2204]` index matrix and source-order file with SHA-256 hashes. Every
policy metric and every paired utility difference uses those same draws.
Report two-sided 95% percentile intervals.

The evaluator applies the six registered train advancement conditions
literally. A point must have at least 100 calls and 50 called sources, a
strictly positive source-utility lower endpoint, strictly beat all feasible
non-oracle baselines and all three ablations, not exceed the no-harm and
strongest-baseline harm quantities, and pass every fit/join/tie/cost/bootstrap
audit. If multiple points qualify, choose higher source utility, then lower
induced harm, then lower nominal rate. Otherwise emit `decar_not_advanced` and
leave validation sealed.

## Execution and security

Run fit plus evaluation in one four-hour `q-h800` job on one H800 with 32 CPUs
and 384 GiB. The submitter requires a clean tracked revision, absent fit and
evaluation output roots, a successful Slurm admission test, and at least 240
remaining GPU-minutes. All supported state emails go to
`yihangc@connect.hku.hk`.

The worker exports no credential, downloads nothing, and cannot read
validation or test. It records the train decision but does not open validation
automatically. No GitHub push is authorized.

## Pre-endpoint verification

- Focused evaluator regression: 5 passed.
- Complete repository regression: 426 passed, 22 skipped.
- Focused mypy: no issues in the evaluator module and runner.
- A synthetic 2,204-source, 39-policy, 21-metric bootstrap stress test took
  1.877 seconds for 1,000 shared resamples on the login CPU, projecting about
  38 seconds for the frozen 20,000 resamples; it used no task endpoint.
- Python compilation, Black formatting, shell syntax, and whitespace checks
  passed.
