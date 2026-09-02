# InfographicVQA relative-where action-generalization result (job 203241)

Date: 2026-09-02

## Decision

The registered post-gate official-train diagnostic completed successfully and
does not change the parent decision `relative_where_train_not_supported`.
Validation, test, and every other protected outcome remained sealed.

The result rules out another learned crop ranker on the existing question,
global-image, four-region, and geometry representation. The source-held-out
models are confidently wrong at chance rate: changing the loss, weighting,
tie handling, or row/column factorization does not expose a transferable
localization signal.

## Bound execution

- Slurm job `203241`: `COMPLETED`, exit `0:0`, zero restarts, 15 seconds.
- The diagnostic reserved one RTX 4090 through the `debug` partition but hid
  the accelerator because all calculations were deterministic CPU analysis.
- Four source-OOF variants, 23,946 decisions, 4,406 images, and 2,204 sources.
- Prediction rows were outcome-free. No validation or test input was used.
- All-state email notification was enabled.

Bound SHA-256 values:

```text
1363d5f148a8624741a973c3de1930034901ed7c7fe70095d70c1d4cf772d198  audit.json
e8c9c54c011061d1c295de7773c87032531f056690b4df9e7d22983e12dedfff  complete.json
b9cb5a9c2cc2953b41280e46e36a920db2fb2140cb128e5d1c34dd7a6037b7bd  job-203241-action-generalization-audit.json
d3b484cfdf94bd6d6478a0083d70dc13034072a6e89d39f110d12f7703832834  slurm-infovqa-relative-where-action-audit-203241.out
```

## Teacher-label audit

The crop target is not collapsed to one grid position:

| Teacher-best crop | Count | Rate |
|---|---:|---:|
| `ug-grid-00` | 6,023 | 25.15% |
| `ug-grid-01` | 5,843 | 24.40% |
| `ug-grid-02` | 6,254 | 26.12% |
| `ug-grid-03` | 5,826 | 24.33% |

The best crop beats answer-now NLL on 39.07% of decisions. Exact best-crop
ties are absent. The best-versus-second NLL gap has median `0.00855`; 5.24%,
20.76%, and 52.50% of states lie within `1e-4`, `1e-3`, and `1e-2`,
respectively. Thus near-ties are common but cannot explain the full failure.

## Source-OOF generalization

Overall metrics are source-balanced unless marked question-balanced (`Q`).
Chance is 25% for exact crop, 50% for top-two/row/column/pairwise agreement.

| Variant | Exact Q / source | Top-2 | Row | Column | Pairwise | NLL regret Q / source | Mean max probability |
|---|---:|---:|---:|---:|---:|---:|---:|
| absolute teacher, entropy weighted | 25.49 / 25.74% | 50.38% | 50.53% | 50.22% | 50.31% | 0.09769 / 0.09001 | 84.79% |
| relative teacher, entropy weighted | 25.38 / 25.25% | 50.22% | 50.56% | 50.06% | 50.25% | 0.09934 / 0.09401 | 85.57% |
| relative teacher, uniform weighted | 25.06 / 25.42% | 50.09% | 49.89% | 50.25% | 49.98% | 0.10249 / 0.09408 | 84.88% |
| relative task, entropy weighted | 25.14 / 24.60% | 49.92% | 50.26% | 50.11% | 50.06% | 0.09902 / 0.09384 | 76.05% |

For the primary relative-teacher model, tie-aware exact hit remains only
34.88% at NLL tolerance `1e-3` and 53.49% at `1e-2`. Pairwise concordance is
also chance, so a tie-aware pairwise objective does not recover a hidden
ordering signal.

## Failure localization

- **Not arbitrary ties:** exact teacher ties are zero, and agreement remains
  chance in large-gap teacher-stability deciles.
- **Not four-way factorization:** row and column agreement are both chance.
- **Not the chosen target:** teacher-NLL, task outcome, relative, absolute,
  entropy-weighted, and uniform variants all fail together.
- **Not a confidence-selection opportunity:** predictions range from diffuse
  to nearly one-hot, but higher confidence does not improve exact or pairwise
  accuracy. The highest-confidence primary decile has only 27.09% exact
  source-balanced agreement and 51.31% pairwise concordance.
- **Not a hidden high-value subset:** the highest-entropy deciles degrade below
  chance while NLL regret rises; the largest teacher-gap decile remains about
  25% exact and 50% pairwise.

Training losses did fall strongly in every fold, while held-out-source
agreement stayed at chance. This is overfitting to source-specific embedding
patterns, not an optimization failure or action-index/sign error.

## Consequence

Do not fit another MLP, bilinear head, row/column classifier, confidence gate,
or tie-aware ranker on the same feature bank. The next admissible branch must
change the pre-action spatial evidence itself. The best-supported candidate is
Qwen baseline-forward question-to-visual-token attention pooled into the four
UG regions: it is outcome-free at inference and has previously shown
within-state localization signal on independent DocVQA and TextVQA branches.
It requires a new official-train-only protocol and feature extraction before
any protected InfographicVQA role can be considered.

No GitHub push is authorized by this result.
