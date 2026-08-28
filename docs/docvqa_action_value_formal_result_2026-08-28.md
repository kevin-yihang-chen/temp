# DocVQA context action-value formal result

Date: 2026-08-28

## Verdict

**Primary confirmation failed.** The frozen context-by-geometry policy has
negative point-estimate gain and utility on the untouched DocVQA formal-v2
partition. It does not satisfy the preregistered requirement that mean utility
and the source-cluster 95% confidence-interval lower bound both exceed zero.

## Frozen evaluation

| Metric | Estimate | 95% source-bootstrap CI |
|---|---:|---:|
| ANLS gain | -0.001563 | [-0.006790, +0.003705] |
| Tool-use rate | 7.587% | [5.905%, 9.453%] |
| Mean utility (`gain - 0.05 * calls`) | **-0.005357** | **[-0.010857, +0.000104]** |
| Gain per call | -0.02060 | [-0.08697, +0.05126] |

Answer-now ANLS is `0.900441`; the policy ANLS is `0.898878`. The frozen gate
calls on 122 of 1,608 decisions and 90.16% of calls are classified as
unnecessary. Thus failure is not only the explicit cost penalty: the selected
crops reduce raw task score on average.

## Integrity and provenance

- formal manifest SHA-256:
  `9ceb28d05df5feecedf6cf61fbbb27ce281b94dd027e5d6d6da43ddc091081ac`
- formal rollouts: 8,040 records, 1,608 decisions, 400 source documents;
  1,608 ANSWER plus 6,432 ZOOM siblings
- formal rollouts SHA-256:
  `a7f44c267b11c12f6cbf8f1e714350174c4dfd7e4ab3866fde0dbd84fe0b5aa3`
- rollout provenance SHA-256:
  `304059080d3b2232ccddfed791858dae5e17938b91e194c6359aa395d9b96886`
- frozen model SHA-256:
  `33f2e0b1fd29e52c878bbbf2cd9819cd3c7e65e12afbabbdc5fa1f6687c8496b`
- rollout job: `190296`, `COMPLETED`, `ExitCode=0:0`, runtime `01:11:06`,
  Slurm mail type `ALL`
- evaluation: 10,000 bootstrap resamples, seed `20260828`, clustered by
  `source_id` (400 clusters)

The primary evaluation was written before any formal action-bank or
decomposition analysis. The policy and threshold are permanently frozen and
will not be revised on this partition.

## Consequence

Compact question/confidence context and crop geometry are insufficient for
out-of-source action value. The next registered analysis is the separately
frozen question-conditioned regional-attention policy. It was selected and
hashed before this primary outcome was evaluated, uses an outcome-free formal
feature contract, and remains a multiplicity-adjusted secondary result rather
than a replacement primary.
