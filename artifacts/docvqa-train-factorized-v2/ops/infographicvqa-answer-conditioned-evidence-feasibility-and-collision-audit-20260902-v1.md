# InfographicVQA answer-conditioned evidence feasibility and collision audit v1

状态：2026-09-02 在任何 hidden-state feature 实现、outcome 拟合或 GPU 提交前完成。
本文是 outcome-free feasibility/literature gate，不是正负实验结果，也不授权读取
validation/test/reserve。

## 原候选

原计划是在 baseline answer 已经生成、但尚未执行 crop 时，复用生成过程中的
answer-token hidden states，并与原图区域表示计算 evidence consistency。目标是给
fixed raw-attention action 的 stopping 引入旧 80 维特征没有的 answer 语义与
answer-to-region grounding 信息。

## 代码可行性

代码审计结论是工程上可行：

1. `src/beyond_entropy/rollout.py::collect_sibling_rollouts` 保留 ANSWER sibling 的
   `answer_before`，且任何 ZOOM outcome 都在 baseline generation 之后产生；因此
   baseline answer 属于合法的 pre-tool observation，reference answer 不属于。
2. `src/beyond_entropy/qwen_backend.py::Qwen25VLBackend.infer` 已用
   `return_dict_in_generate=True`、`output_logits=True` 和 `use_cache=True`。固定运行
   环境的 Transformers `5.4.0` 中，`GenerateDecoderOnlyOutput` 原生包含
   `hidden_states` 与 `attentions` 字段；加入 hidden-state 返回不要求第二次完整
   generation，但仍需真实显存/吞吐 smoke 才能量化成本。
3. `src/beyond_entropy/qwen_semantic.py::Qwen25VLSemanticExtractor` 已输出 global
   visual embedding 与每个候选区域 embedding。旧 feature contract 只有 question、
   global/region 表示、baseline entropy、answer 长度/类型等表面量，没有生成答案
   token 的语义 hidden state。

所以该候选不是因为实现困难被否决。

## 一手文献碰撞

该候选没有通过顶会新颖性 gate：

- [ContextualLens](https://arxiv.org/abs/2411.19187) 已从 VLM 中间层的 contextual
  token embeddings 做 hallucination detection 与 visual grounding，并覆盖 OCR、
  spatial relation 与 grounded VQA。Answer/image token 的 contextual similarity 与
  本候选的核心观测高度重合。
- [Reading Between the Lines / LRP](https://arxiv.org/abs/2511.19806) 已在四个
  image/video text-rich benchmark 上训练 hidden-state/attention probes 做 VLM
  abstention，并明确比较跨层 hidden representations 与 visual-token attention。
- [Visuals Lie, Consistency Speaks](https://arxiv.org/abs/2606.17389) 系统比较
  spatial attention 与 generation-time hidden-state reliability probes，结论本身已
  把“生成隐状态比视觉 attention 更适合 reliability”作为中心发现。
- [V-Loop](https://arxiv.org/abs/2601.18240) 直接用 answer-conditioned verification
  question 与 visual-attention consistency 检查 VQA 答案。
- [Hallucinations Leave a Grounding Signature](https://arxiv.org/abs/2607.27823)
  进一步使用 generation-time grounding signature 和轻量 verifier 做选择性纠正。

项目仍可把这些方法实现成 strong reliability baselines，但不能把“answer hidden
state + region grounding 预测是否调用 crop”单独作为足以支撑 ECCV/ICCV/CVPR 的
新方法。其 deployment target 虽然从 abstain 变成 acquire，但核心表征、probe 和
grounding estimand 已被直接覆盖。

## 决定与完整性边界

决策码：`answer_conditioned_evidence_candidate_rejected_before_experiment`。

- 不实现该 feature，不拟合 probe，不提交 GPU job，也不打开任何新 outcome。
- 此决定是文献碰撞导致的 pre-experiment rejection，不能记录成“实验失败”。
- 不用 layer/head/pooling/hidden-state classifier sweep 规避新颖性问题。
- Generic group-DRO、IRM、conformal threshold 或跨域 calibrator 也不能单独成为
  替代新方法：仓库已有 shared/worst-domain/risk-control 实现和负证据，统计学上
  也有成熟直接先例。

下一候选必须改变学习对象而不是 feature：审计 same-prefix、signed、cost-aware
counterfactual visual-action credit 能否在视觉工具 RL 中只给 action tokens 分配
局部 advantage，并与 outcome-only、question-level necessity 和普通 decoupled RL
形成可验证区别。
