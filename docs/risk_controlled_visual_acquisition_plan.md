# Risk-controlled visual acquisition: contingency method plan

Status: method-design note frozen while the fresh TextVQA confirmation is
running. It is not a claim, result, or authorization to tune on any formal
target.

## Motivation

The current factorized action-value model estimates rescue and harm before a
visual action. Its DocVQA confirmation failed because both components of the
decision transferred poorly: the call gate over-selected low-value states, and
the learned action ranking selected harmful crops. A higher-capacity predictor
alone would not address the central deployment question: how much evidence is
required before the system is allowed to acquire another visual observation?

The contingency direction is therefore **risk-controlled visual acquisition**.
Keep the action-value ranker, but replace a point-estimate call margin with a
calibrated selective rule whose allowed call set has a finite-sample upper
bound on acquisition harm. This is distinct from ordinary confidence
calibration: the loss being controlled is the paired, answer-now-relative harm
caused by the visual action.

## Statistical object

For state `x`, let the frozen ranker select action `a*(x)`, let `s(x)` be its
pre-action score, and let `C_t(x) = 1[s(x) >= t]`. Sibling rollouts provide

`Delta(x) = task_score(x, a*(x)) - task_score(x, ANSWER)`.

At deployment cost `lambda * cost(x)`, define three bounded losses for a
finite threshold family `T`:

- induced-harm mass:
  `H_t(x) = C_t(x) * max(-Delta(x), 0)`;
- net-negative-call mass:
  `N_t(x) = C_t(x) * 1[Delta(x) - lambda * cost(x) < 0]`;
- negative net-value magnitude:
  `V_t^-(x) = C_t(x) * max(lambda * cost(x) - Delta(x), 0)`.

The main risk should be expected induced-harm mass per decision. Conditional
harm among calls is reported but is unsuitable as the only constraint because
its denominator becomes unstable for conservative policies. Utility remains

`U_t = E[C_t(x) * (Delta(x) - lambda * cost(x))]`.

The independent calibration unit is a whole source, not an individual
question. For a source with multiple questions, first average each bounded
loss within the source and then apply the risk test across source-level rows.
This yields a source-balanced population risk and avoids treating correlated
questions as independent. Question-weighted utility remains an explicitly
separate benchmark metric unless a valid clustered concentration result is
specified in advance.

## Learn-then-test protocol

1. Split all labeled development data by whole `source_id` groups into ranker
   training and risk calibration. No sibling, image, or source may cross the
   split.
2. Fit the action-value ranker and candidate ordering on ranker-training
   sources only. Freeze it before reading calibration outcomes.
3. Construct a finite threshold grid from ranker-training scores only, always
   including the no-call rule. Candidate group-conditional variants must also
   be enumerated before calibration outcomes are read.
4. On calibration sources, use a simultaneous learn-then-test procedure to
   identify thresholds whose high-probability upper bound on `E[H_t]` is at
   most a pre-registered tolerance `rho`. Selection from this accepted set may
   maximize call coverage or a simultaneous lower bound on utility.
5. Refit no component and change no threshold on the formal target. Evaluate
   the selected rule once on a new source- and RGB-disjoint target with a
   source-clustered confidence interval.
6. Count abstention/no-call as a valid action, disclose the risk tolerance,
   family size, calibration sample size, accepted set, and realized formal
   risk. Report utility, raw gain, call rate, harm, unnecessary calls, and
   oracle headroom together.

The first implementation should use a transparent finite threshold family and
bounded-loss concentration or e-value tests. More elaborate continuous score
calibration is deferred until this minimal protocol works.

## Distribution-shift boundary

Ordinary conformal and learn-then-test guarantees rely on exchangeability.
The DocVQA development-to-formal reversal is evidence that this assumption
cannot be asserted casually. The paper must distinguish three settings:

- exchangeable source holdout: standard finite-sample risk statement;
- prespecified group mixture shift: group-conditional calibration and a
  disclosed bound for the target mixture;
- general covariate shift: importance-weighted risk control only when target
  density ratios can be estimated from outcome-free, pre-action covariates and
  the required overlap assumptions are defensible.

Without one of these conditions, the method is a calibrated heuristic and the
untouched-target result is empirical evidence, not a distribution-free
guarantee.

Relevant foundations are [Learn then Test](https://arxiv.org/abs/2110.01052),
[Conformal Selective Prediction with General Risk
Control](https://arxiv.org/abs/2603.24704), [High Probability Risk Control Under
Covariate Shift](https://proceedings.mlr.press/v266/almeida25a.html), and
[Adaptive Learn-then-Test](https://proceedings.mlr.press/v267/zecchin25a.html).
These works provide statistical machinery; the proposed contribution would be
the visual-action-specific paired loss, exhaustive sibling supervision, and
pre-execution acquisition policy.

## Required experiment and stop rules

This direction may proceed only after the fresh TextVQA result is locked.

- If the fresh policy passes its registered 97.5% interval, first report the
  simple frozen policy. Risk control becomes a robustness extension, not a
  replacement developed on that target.
- If it fails, the current attention policy family is closed on all consumed
  TextVQA and DocVQA targets. Risk-control development may reuse development
  banks but may not use formal outcomes to select features, thresholds, loss
  tolerances, or groups.
- A paper-level positive claim requires a new untouched benchmark or a newly
  reserved source bank. Re-testing a revised rule on any already opened formal
  bank is diagnostic only.
- If the risk-controlled rule reduces harm by merely approaching zero calls
  and has no positive utility lower bound, it is not a successful method.

## Paper-level hypothesis

The falsifiable hypothesis is:

> Paired sibling counterfactuals can calibrate a pre-action visual acquisition
> set with bounded induced harm, while retaining positive answer-now-relative
> utility on untouched source groups.

This is narrower than claiming a new cropper or a generally better visual
agent. It directly targets the failure mode exposed by the existing formal
experiments and remains separable from outcome-only visual-tool RL.
