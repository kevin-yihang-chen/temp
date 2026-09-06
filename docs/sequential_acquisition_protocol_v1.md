# Sequential Visual Acquisition Protocol v1

Status: preregistered development protocol; held-out test is not authorized.

## Estimand

For each image-question state, an outcome-blind deterministic proposer first supplies one
UG-grid observation. At the resulting shared prefix, the only legal decision is:

- `STOP`: answer from original image plus the already acquired crop;
- `CONTINUE`: acquire exactly one fixed, geometrically farthest UG crop and answer from
  original plus both crops.

Both branches use the same model, prompt, scorer, decoding configuration, and generation
seed. The cost-independent target is

`gain = correct(CONTINUE) - correct(STOP)`.

Policy-time utility is `gain - lambda * incremental_visual_cost`. No cost is included in
the learned target. This protocol studies *when* to acquire; there is no candidate ranking,
free-form box generation, or exhaustive test-time search.

## Fixed proposer

`sequential-opposite-ug-v1` constructs four valid UG-grid actions. A SHA-256 hash of the
state ID chooses the already acquired action, spreading prefixes across spatial locations
without outcomes. The proposed next action is the farthest remaining crop by center
distance with action-ID tie breaking. Its output is a deterministic function of the legal
pre-action state and image geometry.

## Data and leakage boundary

Development uses only the existing source/RGB-disjoint train and validation manifests for
ChartQA, DocVQA, and HRBench. Every paired row saves state/source/image IDs, question,
prefix observation, proposed bbox, seed/replicate, STOP and CONTINUE answer/confidence/
entropy/correctness, incremental and total costs, and cost-independent gain.

The critic receives a typed allowlist containing current-prefix confidence, question/global
visual representations, original-image ROI pools for acquired/proposed boxes, the frozen
Qwen state of original plus the acquired crop, bbox/history/step, and costs. It cannot read
the proposed crop pixels, CONTINUE answer/confidence/entropy, correctness, gain, reward, or
sibling outcomes. The proposed ROI is pooled from the original encoding; it is not executed
as a crop to construct inference features.

## Diagnostic gate before critic training

Before any critic fit, each benchmark must report beneficial, harmful, and neutral rates;
mean/oracle gain; entropy-gain correlation; entropy sign mismatch; and useful-acquisition
precision/recall. If beneficial events or oracle utility are negligible, that benchmark
does not proceed to a more complex critic.

## Models and baselines

The backbone is frozen Qwen2.5-VL-3B. Two small critic families are allowed: linear and a
two-hidden-layer MLP. Critic A predicts remaining STOP risk; Critic B predicts signed
counterfactual gain. No additional architecture/loss search is permitted.

Policies include always STOP/CONTINUE, deterministic matched-rate random, entropy,
confidence, margin, a compatible static VOI score if one exists for the partial-prefix
information set, learned gain, risk-plus-gain, and oracle. Missing static scores must be
reported as unavailable rather than substituted from a different information set.

## Metrics and inference

Report accuracy, acquisition rate, incremental/total visual cost, incremental/total net
utility, oracle gap, useful precision/recall, harmful and unnecessary acquisition rates.
Risk reporting includes AUROC, Brier, ECE, AURC, and risk-coverage. Learned-versus-baseline
comparisons use at least 10,000 paired whole-source bootstrap replicates. Lambda is swept at
policy time without retraining to produce the accuracy-cost frontier.

## Test transaction and verdict

Train and validation APIs reject test. A held-out test can be opened only after all model,
feature, threshold, lambda, baseline, bootstrap, and verdict rules are frozen in a
hash-bound `sequential_test_freeze_v1` one-shot authorization. Previously consumed static-
router test outcomes are not reusable evidence for this new estimand.

GO requires a nontrivial 5--80% call rate and a positive lower confidence bound over the
strongest deployable matched-rate baseline on at least two benchmarks, including consistent
cross-domain evidence. Otherwise the result is PARTIAL GO or NO-GO under the user-specified
stop rules. No RL, 7B, multi-step, continuous boxes, or architecture search is allowed in
this task.
