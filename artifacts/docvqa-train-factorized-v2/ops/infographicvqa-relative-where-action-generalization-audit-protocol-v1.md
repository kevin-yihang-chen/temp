# InfographicVQA relative-where action-generalization audit protocol v1

Status: frozen after the official train decision
`relative_where_train_not_supported` and before running this diagnostic.

This is a post-gate official-train diagnostic. It cannot change the parent gate,
select a call rate, open validation/test, or justify a performance claim. It
reuses the fixed 23,946 source-OOF prediction rows and their exact parent result.

For each decision and each of the four registered variants, report:

- exact teacher-action agreement and the 25% four-action reference;
- hit rate when the chosen crop is within NLL tolerance 0, 1e-4, 1e-3, and
  1e-2 of the best crop;
- top-2, 2x2 grid row, and 2x2 grid column agreement;
- chosen-action and probability-weighted NLL regret;
- pairwise crop-ranking concordance after excluding exact teacher ties;
- predicted and teacher action priors plus the full confusion matrix;
- per-outer-fold summaries;
- deterministic population-rank deciles for maximum predicted probability,
  teacher best-vs-second NLL gap, and baseline entropy;
- source-frequency strata 1, 2--4, 5--9, and at least 10 decisions.

Also report exact and near teacher ties, best-vs-second gap and crop-range
quantiles, the rate at which the best crop beats answer-now in teacher NLL, and
source-frequency quantiles. Question- and source-balanced means are both
required. Decile 0 is the lowest-valued tenth and decile 9 the highest; ties are
ordered deterministically by `(state_id, replicate_id)` only for bin allocation.

The diagnostic may motivate a separately frozen train-only family. In
particular, strong row/column generalization with weak four-way generalization
would motivate a factorized locator; high tie-aware hit with weak exact accuracy
would motivate tie-aware objectives; failure of confidence/stability strata to
improve regret would argue against further selection on this representation.
These interpretations are not predeclared scientific pass/fail rules.

Bind the parent-result SHA-256 and all data/code hashes before execution. Write
atomically to a new directory, never overwrite, hide the reserved GPU from this
CPU-only diagnostic, remove credentials/proxies, and use Slurm mail type `ALL`
for `yihangc@connect.hku.hk`. No GitHub push is authorized.
