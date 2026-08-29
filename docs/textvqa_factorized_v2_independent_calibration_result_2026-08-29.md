# Factorized-v2 TextVQA independent calibration result

Calibration decision: **FAIL**.

This branch is closed as a negative independent calibration. The sealed formal split must not be materialized or evaluated for this candidate.

The threshold order, safety constraints, non-degeneracy floors, source allocation, and reporting template were frozen before calibration output. No formal outcome was used.

## Decision summary

- Selection status: `no_non_degenerate_safe_threshold`
- Selected threshold: none (answer now)
- Tested thresholds: 9 / 11
- Fixed-sequence stopping threshold: `0.086497`
- Source count / decision count: 3,000 / 4,747
- Safety family error / per-step cutoff: 0.05 / 0.025
- Non-degeneracy floors: source call rate >= 0.01 and source utility >= 0.001

## Frozen threshold sequence

| Step | Threshold | Source call | Source utility | Harm mean | Harm p | Negative-call mean | Negative-call p | Risk | Non-degenerate | Selected |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: |
| 1 | 0.404608 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | pass | no | no |
| 2 | 0.190024 | 0.003667 | 0.000117 | 0.000250 | 0.000006 | 0.003000 | 0.000000 | pass | no | no |
| 3 | 0.151428 | 0.005667 | 0.000183 | 0.000583 | 0.000073 | 0.004500 | 0.000000 | pass | no | no |
| 4 | 0.131245 | 0.006667 | 0.000433 | 0.000583 | 0.000073 | 0.005167 | 0.000000 | pass | no | no |
| 5 | 0.119666 | 0.008333 | 0.000550 | 0.000583 | 0.000073 | 0.006500 | 0.000000 | pass | no | no |
| 6 | 0.112072 | 0.010000 | 0.000750 | 0.000633 | 0.000101 | 0.007833 | 0.000000 | pass | no | no |
| 7 | 0.101727 | 0.012500 | 0.000992 | 0.000633 | 0.000101 | 0.009833 | 0.000060 | pass | no | no |
| 8 | 0.093879 | 0.014333 | 0.000900 | 0.000633 | 0.000101 | 0.011667 | 0.001948 | pass | no | no |
| 9 | 0.086497 | 0.016667 | 0.000783 | 0.000633 | 0.000101 | 0.014000 | 0.046206 | fail | no | no |

## Integrity and provenance

- Calibration JSON SHA-256: `e23ec5752800b4a40d56a3b5151798d8fc06d46bcd874eeacf24ba398c33a849`
- Calibrated model SHA-256: `1b765595a46e3c8578a2dc77902606eae8541f0373af2c48ea09bdb52cdd912c`
- Candidate SHA-256: `9a6c9d032ebdbc271b7d3c829fbb3d6ff167cac01b54ce75adc8da86e3063342`
- Allocation SHA-256: `bc0ecb4b6f49a5b0e92b90b4c30620f72246722370d59c8078753d5846f5e9b6`
- Rollouts SHA-256: `59a9de3a758400d1c86b7f4498e21bcce30a15e80ed8e9f1683653a286fc8403`
- Rollout audit SHA-256: `88f3ff91d70ec61994eaab19c9936cc7bf2ff399b1e9d602398028e3521ad866`
- Label-free features SHA-256: `8ac08a7a9e8d6d72b0e8db71625c9fedfc18a1f7ed4843b2e1aa9be89f8aae5c`
- Protocol SHA-256: `babf01d4090263d1cfcb28c42f86f7b13ae9de4bb6bab0ca10d6e4707f02e2ca`
- Code revision: `d85c8d57db2b0c663f760e1fc43a0a9920297422`
- Formal outcomes used: `false`
