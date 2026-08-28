# Post-ChartQAPro decision protocol

## Status and purpose

This branching rule is frozen while the 1,625-question ChartQAPro formal
rollout is still running and before any formal target outcome is available.
Its purpose is to prevent the project from turning one untouched evaluation
into an iterative tuning set.

The formal analysis is bound to:

- ChartQAPro formal manifest SHA-256
  `5a3ddca2e6476196aac8ad4fa7bc00033f2ac9c39d2011fe21fa070e965b97d4`;
- compatibility report SHA-256
  `93e6f04989fa00c247406baaad2815a486b8d145bf8fa932b83648cf5995fe99`;
- prompt-isolation replay audit SHA-256
  `173ff249f1fb8c25b73abdc28f32d705bd3d25737dea6d3bd58b8ce042106480`;
- frozen model SHA-256
  `5d5c0f781a7141726e786d6ad87b861a6395c489bcf9ad8567a2e9ca825c3330`;
  and
- rollout code revision `d9b35b8e735848872e5ea315cfd56cd0398512a6`.

Released-code score is primary. Paper-spec raw exact match and the conservative
canonicalized paper-spec score remain sensitivities. The pass criterion is the
six-part criterion already registered in `gate3_untouched_target_protocol.md`.

## Branch A: frozen transfer passes

If every primary criterion passes:

1. Treat cross-domain when-to-call transfer as independently confirmed.
2. Keep the formal target permanently untouched by subsequent fitting.
3. Enter a bounded VTool-R1 Stage A experiment comparing outcome-only reward
   against counterfactual visual-action advantage under matched samples,
   updates, model initialization, and GPU budget.
4. Use stopping and local visual-action credit as the primary contribution.
   Spatial action selection remains a separate claim until a selector beats
   matched random and fixed crops on another untouched target.
5. Require at least one additional model scale or architecture and one
   non-chart task before making a general multimodal-agent claim.

## Branch B: frozen transfer fails

If any primary criterion fails:

1. Report and retain the result as a failed cross-domain confirmation. Do not
   tune the frozen model, scaler, threshold, prompt, cost, or action geometry on
   the formal labels.
2. Do not evaluate a replacement primary model on the same formal target. Once
   inspected, this target becomes descriptive evidence rather than a fresh
   confirmation set.
3. After recording the frozen result, ChartQAPro pilot labels may enter a new
   development pool, but the formal split must remain excluded from all model
   selection. Register a new identity-audited target before testing a revised
   method.
4. Develop a multi-domain action-value model rather than another source-only
   threshold. Candidate development directions are:
   - domain-normalized baseline-risk and expected-gain calibration;
   - semantic features stripped of benchmark prompt templates and dataset IDs;
   - multi-domain sibling rollouts with group-held-out validation;
   - worst-domain or lower-confidence-bound utility training rather than mean
     in-domain utility; and
   - explicit separation of call probability, action value, and spatial action
     ranking.
5. High-cost RL remains on hold until the revised when-to-call model has one
   new untouched confirmation. This avoids spending VTool-scale compute on an
   unverified reward signal.

## Paper-positioning boundary

The project keeps three possible packages, in descending methodological scope:

- **Action-value method paper:** counterfactual success supervision improves
  stopping and localized visual-tool credit across models and datasets.
- **Counterfactual visual-tool benchmark plus method:** a released sibling
  rollout bank demonstrates when entropy and outcome-only reward misassign
  value, paired with a domain-robust gate.
- **In-domain stopping study:** the current ChartQA confirmation alone. This is
  useful evidence but is not sufficient for the intended CVPR/ICCV/ECCV claim
  without a stronger cross-domain method or benchmark contribution.

Recent related work narrows the defensible novelty. UG already uses uncertainty
to score acquired visual candidates, VTool-R1 already trains visual tool use
with outcome reward, and general reducible-uncertainty formulations now exist
for agents. The project's claim must therefore remain the full combination of
pre-action, cost-sensitive, sibling-counterfactual task-success value and its
use for visual-action credit assignment—not the phrase “reducible uncertainty”
by itself.
