# 研究计划

更新时间：2026-09-02 17:25（Asia/Hong_Kong）

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

### H4：需要新的 pre-action 信息来源，而不是现有特征上的局部调参

下一方法候选必须同时满足：

1. 引入可解释的新信息来源或 action proposer，而不是 attention 层/head、阈值、
   线性 head 或 call rate 的变体；
2. 明确分离 `whether/when` 与 `where`，并对每个具体 action 的 signed rescue/harm
   保留完整 sibling supervision；
3. 在任何结果读取前冻结唯一候选、调用成本、source split、primary endpoint、
   强基线与停止规则；
4. 先做小规模真实输入 smoke 和成本审计，再决定是否值得 GPU 完整运行；
5. official-train 只作 exploratory/source-OOF screen；validation/test/reserve 继续
   封存，只有严格 train gate 通过才允许新 calibration 协议。

候选方向优先级：

1. 具有新观测语义的显式 counterfactual stop/where representation；
2. 能突破固定四格 action-bank 限制的 proposer，但必须对额外推理/视觉获取付费；
3. 若没有候选能在冻结 OOF gate 上给出正 utility，则停止正方法叙事，转为完整
   sibling bank、prospective risk 与跨域失败机制的 empirical audit 路线，并重新
   审计它是否达到顶会新颖性。

## 止损规则

- 不再运行 attention layer/head/ratio、entropy threshold、call-rate、随机种子或
  线性 classifier-family sweep。
- 不用 validation/test 帮助选路线，不把 privileged oracle 当部署结果。
- 下一候选必须先写 protocol，再实现，再 smoke；没有能区分科学假设的新信息时
  不提交 GPU job。
- 若下一次“机制上不同”的冻结 OOF 候选仍不能获得正 utility 或显著强于
  entropy/raw-attention，正方法路线关闭，不再用算力追逐局部变体。

## 紧接着的行动

1. 已将 Jobs `203273`/`203340` 的完整负结果、哈希与路线关闭规则写入不可变审计。
2. 同步 `PROJECT_STATUS.md` 与 `EXPERIMENTS.md`；保持 validation/test/reserve
   封存。
3. H4 候选矩阵已完成；唯一优先候选是复用 baseline generation 的
   answer-conditioned evidence consistency。先做 outcome-free feature/cost/literature
   feasibility audit，不拟合、不读 endpoints。
4. 只有 feasibility audit 通过才冻结单一 source-OOF protocol；当前 Slurm 队列
   为空，在 protocol 和真实输入 smoke 完成前不烧 GPU。
