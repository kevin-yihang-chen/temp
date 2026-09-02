# 研究计划

更新时间：2026-09-02 22:00（Asia/Hong_Kong）

## 总目标与完成标准

目标不是获得一个局部正数，而是形成可投稿 ECCV/ICCV/CVPR 的完整证据链：

1. 与现有 uncertainty guidance、selective VQA、adaptive visual acquisition、
   attention crop 和视觉工具调用工作有清晰区别的新颖命题；
2. 至少一个预先冻结、无泄漏的主结果，优于 entropy、fixed/random crop、
   exhaustive/UG、现有学习式选择器及相关文献强基线；
3. 多数据集或明确的跨域证据，包含消融、风险/校准、置信区间和失败分析；
4. 从干净环境可复现的代码、环境、数据哈希、配置、种子、日志和产物；
5. 论文论证与实验支持一致，不把 exploratory/privileged ceiling 当正式结果。

当前尚未达到上述标准。

## 当前核心判断

InfographicVQA 上，raw attention 的 `where` 信号真实存在，但在相同 entropy
call set 上仍不能产生正净 utility。ViCrop 与 LASER 两个冻结文献 attention
候选也全部失败，且没有显著优于 raw attention。固定 raw action 后的
privileged stopping ceiling 很大，但 attention max/margin 与单一低容量
signed-value OOF stop 都无法利用它。

因此，当前失败不是“没有任何有用 crop”，而是现有 outcome-free representation
无法以足够精度预测哪些状态值得调用。Fixed four-box attention-localization、
entropy/simple-confidence stopping 与当前线性 signed-value stop 家族现已关闭。

原本唯一优先的 answer-conditioned evidence consistency 在实现前完成了代码与一手
文献审计：工程上可以从同一次 baseline generation 复用 hidden states，但
ContextualLens、LRP、VRP、V-Loop 等已直接覆盖 answer/image hidden-state probe、
grounding 与 reliability。该候选因此以
`answer_conditioned_evidence_candidate_rejected_before_experiment` 关闭，没有提交
GPU、拟合模型或读取新 outcome。

## 已完成的关键假设

### H1：文献 attention 方法能进一步改善 where

- 候选：固定 ViCrop Qwen relative attention 与 LASER contrastive all-head bank。
- 特征 Job：`203273`；评估 Job：`203340`。
- 结果：`literature_attention_where_train_not_supported`。两个候选在五个注册
  call rate 的 utility 均为负，Bonferroni-corrected 97.5% lower endpoint 均不
  大于零，也未显著优于 raw attention。
- 结论：关闭对 attention layer/head/ratio 与 fixed four-box attention scorer
  的继续搜索；不进入 calibration。

### H2：固定 raw-attention action 后，主要剩余 headroom 来自 stopping

- Job `203290` 的 unrestricted privileged fixed-action stop ceiling 为
  `+0.021318`，95% CI `[0.018447, 0.024444]`。
- Attention max/margin 在所有注册调用率均差于 entropy。
- 结论：stopping headroom 存在，但简单 attention confidence 不可用。

### H3：固定 raw action 的 signed net value 可在 source-held-out 条件下学习

- Job `203330` 在唯一 2% primary 上得到
  `fixed_action_signed_stop_train_not_supported`。
- Candidate utility `-0.000063`，95% CI `[-0.000739, 0.000655]`；相对 entropy
  改善 `+0.000522`，paired CI `[-0.000304, 0.001444]`。
- 结论：存在弱排序信号，但不允许在已打开 outcomes 上继续搜索 C、特征、权重、
  seed、call rate 或 classifier family。

## 下一主假设与路线选择

### H4：新的 pre-action hidden-state 信息足以形成独立方法

- 可行性：通过；Transformers `5.4.0` 可在 generation output 中返回 hidden
  states，旧 feature contract 确实没有 answer semantics。
- 新颖性：失败；与现有 hidden-state/grounding reliability probes 直接碰撞。
- 决定：实验前关闭，不实现、不提交作业、不把它记成负实验结果。

### H5：same-prefix signed action credit 能改善视觉工具 RL

下一唯一优先研究对象不再是 deployable pre-call classifier，而是对同一 agent
prefix 下 factual visual observation 与 no-op/固定 alternative observation 的最终
任务差定义 signed、cost-aware `A_visual`，并只分配给对应 tool/action tokens。
Final-answer/reasoning tokens 继续使用 outcome reward。

候选必须同时满足：

1. 与 VTool-R1 outcome-only、ToolVision committee evidence/MUT、AdaTooler-V
   query benefit 和 AdaptVision decoupled objective 明确区分；
2. 不是 question-level necessity label，而是具体 action/observation 的同 prefix
   signed rescue/harm/cost contrast；
3. 在 upstream 第一段 tool tokens 当前被 `response_mask=0` 的前提下，新增独立且
   可审计的 action mask/advantage 通路；
4. 先通过 synthetic sign/mask/unit test，再做极小 4×H800 smoke；
5. paired zero/shuffled controls 必须复用相同 pairs、mask、steps 与计算；另分别做
   outcome-only 的 trajectory/step-matched 和 GPU-hour-matched 比较，不虚构一个
   baseline 能同时匹配互相冲突的全部预算轴；
6. 短程学习曲线需同时改善 task score、cost-adjusted utility 与 harmful-call rate，
   否则不扩完整训练。

G0 当前已通过：protocol v1 冻结 arm-specific net-utility contrast、`lambda=0.05`、
`beta=1.0`、raw bounded action credit、zero/shuffled/outcome-only controls 与泄漏边界；
dependency-free core 已实现 action/answer/observation/padding masks、pair provenance、
token-local advantage、序列化与 deterministic derangement。G0 实现 commit 为
`56b990c767973a8a23060d63293db8657254b35d`；upstream-shaped adapter/overlay 的
pre-GPU contract 已通过，但尚未获得真实 rollout、optimizer step 或训练结果。

G0 后已做范围纠偏：VTool 只作为 Apache-2.0 的可运行 RL 骨架和 outcome-only
comparator，不再审计 thought、pixel 或内部实现是否与 VTool 等价；该问题与 H5 的
成败无直接关系。训练数据改为固定 revision/hash 的 Apache-2.0 official ReFocus train。
token-local autograd、隔离 runtime import、official-train converter/processor、paired
agent fake-server contract 与单卡 H800 vLLM model-load/真实首轮 generation 均已通过。
尚未获得真实 paired tool rollout 或 optimizer step，因此 H5 仍停在 G1 前，不是处于
训练中。

### 顶会约束

所有后续候选仍必须满足：

1. 引入可解释的新信息来源或 action proposer，而不是 attention 层/head、阈值、
   线性 head 或 call rate 的变体；
2. 明确分离 `whether/when` 与 `where`，并对每个具体 action 的 signed rescue/harm
   保留完整 sibling supervision；
3. 在任何结果读取前冻结唯一候选、调用成本、source split、primary endpoint、
   强基线与停止规则；
4. 先做小规模真实输入 smoke 和成本审计，再决定是否值得 GPU 完整运行；
5. official-train 只作 exploratory/source-OOF screen；validation/test/reserve 继续
   封存，只有严格 train gate 通过才允许新 calibration 协议。

路线优先级：

1. same-prefix counterfactual visual-action credit 的 novelty/implementation gate；
2. 只有 gate 通过才进行 matched-control 3B RL smoke 与短程曲线；
3. benchmark/causal audit 只能在规模、模型/工具广度与新 estimand 足以独立满足
   顶会标准时成为主路线，不能作为降低投稿档位的默认 fallback。

## 止损规则

- 不再运行 attention layer/head/ratio、entropy threshold、call-rate、随机种子或
  线性 classifier-family sweep。
- 不用 validation/test 帮助选路线，不把 privileged oracle 当部署结果。
- 下一候选必须先写 protocol，再实现，再 smoke；没有能区分科学假设的新信息时
  不提交 GPU job。
- 不再对 answer hidden-state/grounding probe、generic group-DRO/IRM 或 conformal
  threshold 进行局部变体搜索。
- 若 action-credit novelty audit 或短程 matched-control gate 失败，关闭该路线，
  重新选择实质方法/benchmark contribution；不以降低投稿目标作为完成条件。

## 紧接着的行动

1. Answer-conditioned candidate 已因文献碰撞关闭；VTool 仅保留为运行底座和
   outcome-only comparator，停止所有进一步 equivalence 审计。
2. Matched-control action-credit protocol 与 G0 synthetic implementation/tests 已完成；
   这只证明 arithmetic/schema，不是方法有效性证据。
3. 官方 Apache-2.0 ReFocus train 的 revision/shard hash 已固定；只从 official train
   构建 structural-group-disjoint 的最小 G1 数据，不使用旧 derivative 或 protected split。
4. token-local adapter、真实 autograd、pinned runtime、单行真实 processor 和 paired
   fake-server contract 已通过；两臂 shared prefix/seed、image-only delta、role mask、
   rescue/harm/failure/direct credit 均有运行证据。
5. outcome-only、paired-zero、paired-shuffled、paired-signed 四组配置，以及 task score、
   cost-adjusted utility、harmful-call rate 和 tool-call-rate stop rule 已在
   `configs/vtool_action_credit_g1_v1.json` 冻结。
6. 单卡 H800 model-load/单条 generation smoke（Job `205784`）已通过；下一步只实现、
   dry-run 并提交 4×H800、最多 2 optimizer-step 的 paired-signed G1。只有实际
   tool-call rate、pair validity 与训练稳定性通过冻结 stop rules，才以同一 revision
   顺序运行 zero/shuffled/outcome-only controls。
7. 其他 validation/test/reserve 继续封存；本地修改不 push GitHub。
