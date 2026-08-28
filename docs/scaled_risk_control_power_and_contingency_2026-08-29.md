# Scaled risk-control power audit and prospective contingency

Date: 2026-08-29

Status: written before reading any risk-calibration rollout outcomes. This note
does not alter the frozen primary protocol in
`scaled_textvqa_risk_control_preregistration.md`.

## Primary gate remains unchanged

The current primary calibration uses 3,000 independent source groups, at most
32 thresholds, two risk constraints, family error `0.05`, and a Bonferroni
cutoff of

`0.05 / (32 * 2) = 0.00078125`.

Numerically inverting the implemented bounded-mean Bernoulli-KL lower-tail
test gives the following largest empirical source-balanced risks that can
reject their respective nulls:

| Risk limit | Largest empirical mean that rejects | Fraction of limit |
| ---: | ---: | ---: |
| induced-harm mass `0.005` | `0.00100853` | `0.2017` |
| net-negative-call mass `0.02` | `0.01113334` | `0.5567` |

For intuition only, if every source-level loss were binary, these boundaries
correspond to at most 3 and 33 events among 3,000 sources. Actual induced-harm
losses are continuous source-level means, so event counts are not the test
statistic.

This audit explains an important power cost: finite-sample certification is
substantially stricter than checking whether the empirical mean is below the
nominal risk limit. It does not justify weakening the current test after its
calibration outcomes exist. If no threshold passes both risk tests, the 1%
minimum source call rate, and the `0.001` empirical utility floor, the primary
branch selects answer-now, is declared failed, and must not open its formal
role.

## Prospective fallback on a new calibration bank only

The threshold call sets are nested: lowering a score threshold only adds
calls. Both registered losses are non-negative and therefore monotonically
non-decreasing for every source as the call set grows. This structure permits
a more powerful *prospective* fixed-sequence Learn-Then-Test design:

1. Freeze a new model, threshold family, and descending strict-to-permissive
   threshold order before collecting or reading a new calibration bank.
2. At each threshold, jointly test the two registered risks with per-risk
   cutoff `0.05 / 2 = 0.025`.
3. Continue from stricter to more permissive thresholds and stop at the first
   threshold whose joint risk test fails.
4. Select the most permissive preceding threshold that also meets frozen
   non-degeneracy requirements.

At 3,000 sources, the same KL test and a `0.025` per-risk cutoff would move the
detectable empirical boundaries to approximately `0.00193829` for the `0.005`
induced-harm limit and `0.01346461` for the `0.02` negative-call limit. This is
still conservative, but avoids paying a 32-fold threshold-search penalty.

This fallback must not be evaluated on the current risk-calibration bank. It
requires a new source- and decoded-RGB-disjoint calibration role and a new
one-shot formal role. Otherwise it would be post-outcome method selection, not
prospective risk control.

## Further bound work

A second prospective option is a fixed-time betting confidence interval for
bounded source risks. Waudby-Smith and Ramdas derive variance-adaptive betting
intervals that can be tighter than Hoeffding-style bounds. Before such a bound
can enter a protocol, it needs an independently reviewed implementation,
deterministic tests, null simulations, and a frozen multiple-risk procedure.
It is not an authorized substitute in the current branch.

## Primary references

- Angelopoulos et al., [Learn then Test: Calibrating Predictive Algorithms to
  Achieve Risk Control](https://arxiv.org/abs/2110.01052).
- Angelopoulos et al., [Risk Control for Recommendation
  Systems](https://proceedings.mlr.press/v204/angelopoulos23a.html), which
  explicitly uses fixed-sequence threshold testing for a nested family.
- Bates et al., [Distribution-Free, Risk-Controlling Prediction
  Sets](https://arxiv.org/abs/2101.02703).
- Waudby-Smith and Ramdas, [Estimating means of bounded random variables by
  betting](https://arxiv.org/abs/2010.09686).
