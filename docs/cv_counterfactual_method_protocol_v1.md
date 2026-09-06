# Counterfactual Visual Utility Post-training Protocol v1

Status: frozen before Phase A execution (2026-09-06, Asia/Hong_Kong).

## Question and scope

This independent route asks whether lightweight post-training of
Qwen2.5-VL-3B can turn paired counterfactual visual-acquisition outcomes into a
better end-to-end STOP/CONTINUE policy. It reuses the existing paired
partial-prefix banks but does not revise or overwrite the earlier frozen-critic
NO-GO result.

Only ChartQA and DocVQA are in scope. The state contains the original image,
one already acquired fixed crop, and the question. The only action is STOP or
CONTINUE with the already frozen, geometrically opposite UG crop. The model may
read the proposed bbox and cost, but it must not execute the proposed crop before
selecting the action. HRBench, candidate ranking, new visual tools, OCR,
detectors, web tools, RL, continuous boxes, multi-turn acquisition, 7B, feature
search, architecture sweeps, and broad hyperparameter or seed searches are out
of scope.

## Data and leakage contract

Each paired record supplies

\[
G(s)=R_{continue}(s)-R_{stop}(s).
\]

The typed deployable input has only identities, image path, question/model
prompt, acquired crop specification, proposed action ID/bbox, and proposed cost.
STOP/CONTINUE answers, correctness, entropy after CONTINUE, reward, and gain are
labels or diagnostics and cannot enter `QwenSequentialPolicy.forward`.

The original and already acquired crop are encoded; the proposed crop is not.
Train and validation remain source-disjoint. Existing validation outcomes have
already been seen by earlier, different development routes, so Phase A/B are
development evidence only. A new sequential test transaction may be created
only after Phase B passes its transition rule; no test result may be used to
tune thresholds, losses, architecture, schedule, learning rate, or seed.

The frozen paired inputs are:

| Domain | Role | States | Manifest SHA-256 | Rollout SHA-256 |
|---|---:|---:|---|---|
| ChartQA | train | 256 | `f243e3f9...b0934ae1` | `bf456df...8980739` |
| ChartQA | validation | 128 | `81f1a5b0...064d550` | `5e200db...2987a8` |
| DocVQA | train | 256 | `a00dab30...699528a` | `abcf7e5f...cf3644` |
| DocVQA | validation | 128 | `96f24b92...726da5` | `68948ade...1cf7c` |

Full hashes and absolute paths are machine-bound in `configs/cv_method_*_v1.json`.

## Matched model and objectives

Both learned arms use the same Qwen2.5-VL-3B revision, multimodal state,
two-layer 128-wide head, trainable vision merger, last language block and norm,
optimizer, learning rates, deterministic state schedule, number of optimizer
steps, and seed. They differ only in the loss.

Outcome-only post-training uses absolute final rewards:

\[
L_{outcome}=-\frac{R_{stop}\log p(STOP)+R_{continue}\log p(CONTINUE)}
{\max(R_{stop}+R_{continue},1)}.
\]

This deliberately does not convert the two rewards to an argmax or softmax;
doing so would collapse into the same binary preference as the proposed method.

Counterfactual Utility post-training uses explicit paired gain:

\[
L_{cf}=\begin{cases}
-\log p(CONTINUE),&G>0\\
-\log p(STOP),&G<0\\
0,&G=0.
\end{cases}
\]

The two actions are valid by construction and never require JSON/tool-schema
generation. The final answer is the already generated paired branch outcome, so
policy evaluation changes only the selection mask.

## Locked training stages

Phase A is an engineering smoke, not scientific evidence. It selects 25 of 256
train states per domain (9.765625%) by a seed-17, outcome-independent state hash,
uses 64 alternating-domain optimizer steps, and evaluates a hash-selected
validation subset. It must show finite loss/gradients, updates in all three
trainable groups, no proposed-crop execution, nonconstant scores, and no complete
STOP/CONTINUE collapse for both arms. Engineering failures may be fixed without
changing the scientific estimand; any such fix must be recorded.

Phase B is a one-seed (`17`) small pilot using all 256 train states per domain,
one deterministic pass (512 optimizer steps), and the full 128-state validation
bank per domain. No hyperparameter search is allowed. The primary policy calls
CONTINUE for the top 25% of scores. This rate was frozen before the new model was
run and matches the favorable 25--27% region observed in the preceding
sequential diagnostic. Accuracy-cost frontiers additionally use call rates
`0, .1, .25, .5, .75, 1` without retraining.

Phase B advances only if matched-cost Counterfactual Utility improves accuracy
over the strongest of entropy, inverse confidence and inverse margin by more
than `+1` percentage point on at least one domain and remains above `-1`
percentage point on the other. It stops if both domain deltas versus the
strongest baseline are nonpositive, or both deltas versus Outcome-only are
nonpositive. Any result that does not satisfy the transition rule is NO-GO for
Phase C even if it misses those two descriptive stop patterns.

Phase C is generated only after Phase B GO. It uses three fixed seeds on
ChartQA and DocVQA and an untouched, pre-frozen test transaction. METHOD GO
requires at least three of:

1. ChartQA matched-cost gain over the strongest uncertainty baseline exceeds
   `+2` percentage points.
2. DocVQA matched-cost gain exceeds `+1` percentage point.
3. At least two of three seeds are positive on at least one domain with no
   catastrophic collapse.
4. Counterfactual Utility exceeds Outcome-only by `+1` percentage point on one
   domain and is above `-0.5` percentage point on the other.

No 7B or RL experiment is authorized unless Phase C is METHOD GO.

## Metrics and reporting

The primary endpoint is matched-cost accuracy difference at 25% incremental
tool calls. Reports also include accuracy, average incremental and total visual
cost, utility for `lambda = 0, .025, .05, .1, .2`, the accuracy-cost frontier,
paired 10,000-sample whole-source bootstrap intervals, beneficial-call
precision/recall, harmful and unnecessary continuation, missed beneficial
states, natural action rate, score variance, and score-vs-gain calibration bins.

The mandatory comparison is Answer-only, the strongest matched-cost uncertainty
baseline, Outcome-only post-training, and Counterfactual Utility post-training.
Every stage saves the frozen config, code revision, input hashes, seed, schedule
hash, metrics, tool calls, selector checkpoint, resumable checkpoint, logs, and
decision. `CV_METHOD_GO_NO_GO.md` is the sole scaling decision document.
