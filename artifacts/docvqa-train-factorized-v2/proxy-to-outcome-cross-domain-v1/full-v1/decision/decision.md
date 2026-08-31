# DocVQA proxy replication decision

Decision: **replicated_alignment**.

This mechanically applies the five conditions frozen before the full
DocVQA answer-likelihood bank was observed. It selects no score threshold
and does not authorize use of protected outcomes.

## Conditions

- `answer_loss_spearman_ci_low_above_zero`: **PASS** — ci_low=0.22994635
- `answer_loss_top_one_gain_ci_low_above_zero`: **PASS** — ci_low=0.01643523
- `answer_loss_top_one_gain_exceeds_entropy_and_random`: **PASS** — answer=0.01910627, entropy=0.00647653, random=-0.00532801
- `positive_sparse_utility_lower_endpoint`: **PASS** — qualifying_rates=[0.005, 0.01, 0.02, 0.05, 0.1]
- `answer_loss_top_one_harm_below_entropy_and_random`: **PASS** — answer=0.01119293, entropy=0.02135493, random=0.02825847

## Boundary

The frozen cross-domain replication gate passed. This authorizes writing a separate DocVQA-development to ScreenQA-untouched surrogate protocol; it does not validate or select that future method.
