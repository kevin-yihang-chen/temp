# Minimum-rank consensus gate result v1

Status: completed on 2026-09-01 under the frozen opened-DocVQA development
protocol. The mechanical decision is
**`minimum_rank_consensus_gate_not_advanced`**. All ScreenQA and other
protected roles remain sealed.

## Bound execution and integrity

- Slurm job `199972`, one NVIDIA H800, 12 CPUs, 96 GiB, completed in
  `00:00:25`, exit `0:0`, zero restarts, with all-state email enabled.
- Code revision:
  `a527b6831848986d3b0387c15895499543dcb722`.
- Report SHA-256:
  `7ed71b943cae14eb21e1fd56cef80703c0e3771a98f2b0e1ca0d1eb4bd295efa`.
- Score-report SHA-256:
  `a40a01c6e9ec3364b8f5e7db5175792259ec2ebe494d5a85934701b4b668c4d7`.
- Outcome-free score rows SHA-256:
  `303c16d715f616bcf9e476ae655affd6854f890d77d44ddb47b1680a0c899326`.
- Rank contract SHA-256:
  `014f0b13be3697cec91474b94ea636f87065193b3469d901c35f3e41125011af`.
- Completion SHA-256:
  `eb10934b5fc2c15b802c1e001bcf7ed7017326a42d70a4d20d37f35589641142`.

Every bound hash, population, source/identity alignment, embedded-incumbent
reproduction, frozen 225-call set, finite score, empirical-percentile tie and
monotonicity, exact minimum rule, complete-tie 225-call threshold,
serialization, incumbent metric, and no-leakage audit passes. The two raw
scores have Pearson correlation `0.366820`, while their empirical percentiles
have Pearson correlation `-0.002416`; agreement is concentrated in the extreme
tail rather than across the full ordering. The consensus and incumbent call
sets share 180 decisions and have 45 exclusive decisions each.

## Registered result

Both methods make exactly 225 pooled calls. Source-balanced metrics are:

| Metric | Incumbent | Minimum-rank consensus |
|---|---:|---:|
| Utility | 0.00317324 | 0.00296593 |
| Raw gain | 0.00397661 | 0.00382959 |
| Gain per call | 0.247496 | 0.221709 |
| Helpful-call precision | 0.394407 | 0.395863 |
| Induced harm | 0.000341563 | 0.000663457 |
| Negative-value call mass | 0.00977110 | 0.0104924 |
| Helpful-state proposal recovery | 0.516410 | 0.701761 |

Consensus-minus-incumbent utility is `-0.000207303`, with 95% whole-source
interval `[-0.00105372, 0.000496744]`. Helpful-call precision and every
integrity audit pass. Utility margin, paired lower endpoint, gain per call, and
the joint harm/negative-call clause fail. Requiring both OOF scores to be high
therefore does not calibrate realized harm.

## Fixed realized-policy failure decomposition

A post-hoc descriptive decomposition evaluates no new policy, action
combination, threshold, oracle, or protected input. It only partitions the
calls already made by the two frozen policies. Report SHA-256:
`6c98be56455b710f031dac2f1e17263e3b786ec59aeb5e43a47aaf044ebc39b4`.

- On the 180 shared calls, only ten selected actions differ. Relative to the
  incumbent actions, the realized consensus actions are better on one call,
  equal on 176, and worse on three; their mean utility difference is
  `-0.005556` per shared call. The source-balanced consensus contribution on
  this intersection is `0.00308382`, versus `0.00315387` for the incumbent.
- The 45 consensus-only calls have 28.89% helpful-call precision, 73.33%
  negative-value calls, mean utility `0.0171424` per pooled call, and mean harm
  `0.0646392` per pooled call. Source weighting makes their aggregate utility
  contribution negative (`-0.000117883`) and their harm contribution
  `0.000412413`.
- The 45 incumbent-only calls contribute source-balanced utility
  `+0.0000193667` and harm `0.000100043`.

The failure is thus mostly the supposedly conservative gate replacing 45 calls
with a source-concentrated harmful tail, with a smaller additional loss from
the retained cost-sensitive actions on the shared calls. This rules out score
agreement as a sufficient harm-control mechanism on this population.

## Frozen branch decision

Opened-DocVQA candidate selection is closed. Do not try another mean, product,
weight, veto, threshold, call budget, or action recombination on these outcomes.
The positive cost-sensitive point estimate remains a near-positive diagnostic,
not a promoted model.

The next method-development population must be newly declared before any task
outcome is opened. ScreenQA calibration, formal, reserve, untouched, official
validation, and official test remain sealed and cannot absorb this failure.
The selected next route is an InfographicVQA-train development branch with
image-grouped source exclusion, a self-consistent Qwen2.5-VL-7B actor/scorer,
explicit rescue/harm/neutral modeling, and independently sealed official
validation/test roles. No GitHub push is authorized by this result.
