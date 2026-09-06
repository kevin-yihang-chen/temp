# Factorized Potential-Outcome Visual Acquisition: paper method draft v1

Status: method text frozen before Phase-C held-out access; empirical claims marked
`TBD-FORMAL` must not be completed from development results.

## 1. Problem setting

We study one-step visual acquisition after a vision-language model has already
observed an image-question pair and one local crop. Let

\[
s=(I,q,o_1,b_2)
\]

denote the deployable state: original image `I`, question `q`, acquired local
observation `o_1`, and the geometry of one outcome-blind proposed crop `b_2`.
The policy chooses either `STOP`, which answers using the current evidence, or
`CONTINUE`, which acquires `b_2` and then answers. The selector never observes
the pixels of `b_2` before making this decision.

Let `Y_0,Y_1 in [0,1]` be the official task rewards of the paired STOP and
CONTINUE sibling executions. The per-state acquisition effect is

\[
\Delta=Y_1-Y_0.
\]

With a calibrated unconstrained policy, deployment can use

\[
\pi_\lambda(s)=\mathbb{1}[\widehat\Delta(s)>\lambda C(b_2)],
\]

where the cost `C` and trade-off `lambda` enter only at the policy layer. They
are not part of the supervision target. The registered formal comparison uses
an exact call budget instead: it selects the top-ranked states at each call
rate. Because every CONTINUE action in this experiment has the same unit cost,
subtracting `lambda C` does not change that ranking; the exact-rate protocol
isolates ranking quality without fitting a threshold on held-out outcomes.

## 2. Why direct gain prediction is statistically brittle

Direct signed-gain supervision discards every neutral pair under the binary
preference loss. In the frozen Phase-C train banks, beneficial/harmful/neutral
counts are respectively `84/20/920` for ChartQA, `34/23/955` for DocVQA, and
`28/21/339` for HRBench. Thus most paired executions provide no gradient to a
direct preference classifier, even though every pair reveals whether the
current answer is reliable.

Outcome-only imitation uses this dense correctness signal but does not
explicitly distinguish two different reasons not to call a tool: the proposed
crop may fail to rescue a wrong answer, or it may harm an answer that is already
correct. Our method retains dense current-answer supervision while separating
these asymmetric mechanisms.

## 3. Reward-mass factorization

Define current error and correct reward mass

\[
e^*=1-Y_0,\qquad c^*=Y_0.
\]

The normalized rescued and harmed fractions are

\[
r^*=\begin{cases}
\frac{\max(Y_1-Y_0,0)}{1-Y_0}, & Y_0<1,\\
0, & Y_0=1,
\end{cases}
\]

and

\[
h^*=\begin{cases}
\frac{\max(Y_0-Y_1,0)}{Y_0}, & Y_0>0,\\
0, & Y_0=0.
\end{cases}
\]

These targets satisfy the exact identity

\[
Y_1-Y_0=(1-Y_0)r^*-Y_0h^*.
\]

For binary correctness, the three terms reduce to the probability that STOP is
wrong, the probability that CONTINUE rescues an error, and the probability that
CONTINUE harms a correct answer. The reward-mass form also remains valid for
bounded soft metrics such as DocVQA ANLS; no arbitrary correctness threshold is
introduced.

## 4. Factorized selector

Qwen2.5-VL-3B encodes the original image, question, and already acquired crop.
The representation at the final attended token is concatenated with 15
outcome-free geometry and cost coordinates describing the acquired and proposed
regions. A two-layer MLP emits three logits, transformed into

\[
\hat e=\sigma(z_e),\quad \hat r=\sigma(z_r),\quad \hat h=\sigma(z_h).
\]

The predicted acquisition effect is

\[
\widehat\Delta=\hat e\hat r-(1-\hat e)\hat h.
\]

Training updates the visual merger, final language block and norm, and the
128-wide head. The proposed crop is not executed during selector training or
scoring. Consequently, the selector uses a fixed number of pre-action visual
observations rather than exhaustively evaluating candidate crops.

## 5. Objective

With binary cross entropy `BCE`, the per-pair loss is

\[
\mathcal L=\frac{1}{2}\left[
\operatorname{BCE}(z_e,e^*)
+e^*\operatorname{BCE}(z_r,r^*)
+c^*\operatorname{BCE}(z_h,h^*)
\right].
\]

The conditional weights sum to one. For hard rewards exactly one conditional
term is observed; for soft rewards both may carry fractional mass. This
weighting is necessary for the population targets to reconstruct expected gain
on the task-reward scale.

## 6. Matched controls

All trainable controls use identical Qwen initialization, trainable parameter
groups, optimizer, domain-balanced 3,072-step schedules, image processing and
three seeds (`17/29/47`).

1. **Outcome-only SFT** weights the STOP and CONTINUE action log-probabilities
   by their absolute branch rewards.
2. **Direct counterfactual SFT** predicts the sign of `Y_1-Y_0` and receives no
   loss on neutral pairs.
3. **Factorized potential outcomes** uses the three reward-mass targets above.

Evaluation also includes Answer-only, deterministic random calling,
entropy/confidence/margin gates, and a privileged oracle gain ranking. Every
non-oracle method is compared at the same exact call count.

## 7. Evaluation protocol

The one-shot held-out transaction contains 512 ChartQA states, 522 DocVQA
states from 128 documents, and 92 HRBench states from 20 image groups. Training
and held-out identities are source/image disjoint; allocation was frozen without
model outcomes.

The primary operating point is an exact 25% call budget and `lambda=0.05` for
reporting net utility. Call rates
`0/10/25/50/75/100%` and lambdas `0/.025/.05/.1/.2` form the registered
accuracy-cost frontier. Three trained seeds are treated as independent policy
deployments: action selection occurs separately for each seed, and paired
effects are averaged only afterward. Confidence intervals use 20,000 paired
cluster bootstrap samples, with source/document grouping for ChartQA/DocVQA and
image grouping for HRBench.

The formal result is GO only if Factorized exceeds Outcome-only in at least two
domains, has a positive source-cluster 95% CI lower endpoint in at least one,
does not trail the strongest uncertainty baseline by more than 0.5 percentage
points on successful domains, and passes all registered semantic controls.

## 8. Semantic controls

We score the frozen Factorized selector after deterministic outcome-independent
derangements of (i) question and prompt, (ii) source image and therefore the
already acquired visual observation, and (iii) proposed-region action,
geometry, and cost. A control passes only when the original policy has positive
mean task-score advantage and rankings/call sets change in at least two domains.
These tests distinguish image-question-region utility learning from a fixed
location or question-independent calling prior.

## 9. Registered result placeholders

The following fields must be populated only from the immutable Phase-C formal
report:

- Primary table: `TBD-FORMAL`.
- Factorized minus Outcome-only paired deltas and CIs: `TBD-FORMAL`.
- Factorized minus direct-CF and strongest-uncertainty results: `TBD-FORMAL`.
- Accuracy-cost frontiers: `TBD-FORMAL`.
- Question/image/region semantic gates: `TBD-FORMAL`.
- Final decision (`GO` or `NO_GO`): `TBD-FORMAL`.

Development Phase B may motivate the experiment but must not fill these
placeholders or support the paper's final empirical claim.

## 10. Claim boundary

The contribution is not a new general potential-outcome architecture. It is a
visual-acquisition formulation that converts real paired sibling executions
into dense answer-risk supervision plus asymmetric rescue/harm supervision, and
uses the resulting pre-action effect estimate under an explicit visual-cost
policy. The claim survives only if the frozen multi-domain evidence shows an
advantage over both dense Outcome-only imitation and strong uncertainty gates.
