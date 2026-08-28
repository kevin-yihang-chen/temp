# TextVQA source-balanced factorized OOF result

Status: post-failure development diagnostic. The original 3,000-source risk-
calibration bank had already been inspected before this branch was implemented.
No calibration or formal outcome was used by these fits, and this result cannot
revise the failed scaled primary or authorize opening the sealed formal bank.

## Question

The factorized OOF heads were source-grouped at evaluation time but their loss
weights were decision-balanced. In the 5,000-source development bank, 2,912
sources have two questions and 2,088 have one. The old loss therefore gave
58.24% of sources twice the influence of the rest, while downstream risk and
formal estimands average within source first.

This experiment changes only the loss weighting. Every domain receives equal
total mass, every source within a domain receives equal mass, and rows within a
source share that source mass. It compares the existing two state heads:

- `hybrid-context-semantic`: 27-dimensional pre-action context for baseline
  error, with frozen semantic crop features for rescue and harm;
- `semantic-context`: frozen semantic plus context state for baseline error,
  with the same frozen semantic crop features for rescue and harm.

Both modes are tested at fixed `alpha in {1, 10}` using the same five source
folds. The 0.5%, 1%, 1.5%, 2%, 3%, and 5% score tails are development-only
diagnostics and are not independent calibration.

## Reproducibility

- code revision: `ae8e340c3309b14bb9c3b8691cdad7e7c2ff6edf`;
- development rollouts: 39,560 records / 7,912 decisions / 5,000 sources,
  SHA-256
  `1c1d5b67010b5ddfbdabe47072291336b34dcc54928e5db7a12727daa4f14c8e`;
- frozen label-free semantic features SHA-256:
  `93cdfa91b570fcc67f16bdd4e39d59489fa160e26c2797abf16d684f2f44a504`;
- training protocol:
  `source_grouped_oof_domain_source_balanced_v2`;
- Slurm jobs: `191643`, `191644`, `191645`, and `191646`, all completed with
  exit code zero;
- formal outcomes used: false.

## Results

| State head | Alpha | OOF utility | OOF gain | Tool rate | Unnecessary calls | 1% source utility | Tail risk | Tail selection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| context | 1 | 0.003191 | 0.005814 | 0.05245 | 0.8193 | **0.001175** | pass | selected |
| context | 10 | 0.003153 | 0.005359 | 0.04411 | 0.8080 | 0.000790 | pass | no selection |
| semantic + context | 1 | 0.002648 | 0.005384 | 0.05473 | 0.8360 | 0.000870 | pass | no selection |
| semantic + context | 10 | 0.002439 | 0.007647 | 0.10415 | 0.8568 | 0.000975 | pass | no selection |

For context/alpha=1, the selected development tail calls on 1.03% of sources,
has induced-harm mass 0.00040 and net-negative-call mass 0.00770, and satisfies
the same fixed risk limits of 0.005 and 0.02. Relative to the old context/alpha=1
fit, its 1% tail utility increases from 0.000930 to 0.001175, about 26%, and its
overall OOF utility increases from 0.003097 to 0.003191.

## Decision

Source-balanced training is retained. Adding the current frozen semantic state
to the error head is rejected: both semantic-state variants are worse on OOF
utility and neither produces a selectable high-precision tail. The surviving
context/alpha=1 branch is a narrow positive development result, not a stable
success: its utility clears the 0.001 floor by only 0.000175 and was selected
after several development diagnostics.

It must never be evaluated on the already opened calibration bank. Before any
new outcomes are generated, the model, score-to-call rule, risk family, and a
new source/RGB-disjoint calibration allocation must be frozen. The existing
5,000-source formal allocation remains sealed. If a fresh calibration bank
does not select a non-degenerate policy, close this factorized branch without
opening formal outcomes.
