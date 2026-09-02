# 研究计划

更新时间：2026-09-02 13:05（Asia/Hong_Kong）

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

## 当前核心问题

InfographicVQA 的 raw-attention action 在相同 entropy call set 上显著优于四个
deployable where 基线，但所有 call rate 的净 utility 仍为负。现有证据指向
因子化瓶颈：`where` 已有信号，`whether/when to call` 仍不足。

## 正在验证的假设

### H1：文献 attention 方法能进一步改善 where

- 候选：固定 ViCrop Qwen 相对 attention 与 LASER contrastive all-head bank。
- 最小实验：完整 official-train outcome-free 特征抽取、合并、审计；随后用冻结
  entropy call set 和 97.5% Bonferroni 区间评价。
- 任务：Slurm `203273`，两张 H800，四个 source-disjoint 分片，两轮执行。
- 决策：只有正 utility、校正区间和所有强基线条款同时通过才允许 calibration。

### H2：固定 raw-attention action 后，主要剩余 headroom 来自 stopping

- 最小实验：保持 raw action 不变，比较 entropy stop、attention max、attention
  margin 与 privileged realized-utility stop ceiling。
- 任务：Slurm `203290`，20,000 次 whole-source bootstrap。
- 目的：量化 fixed-action stop ceiling，并判断简单 outcome-free confidence 是否
  已能改善 entropy；该实验是 post-hoc 诊断，不可直接产生正式候选。
- 结果：诊断已完成。固定 raw action 的 unrestricted privileged stop ceiling
  为 `+0.021318`，95% CI `[0.018447, 0.024444]`；但 attention max/margin
  在所有注册调用率均差于 entropy。因此 stopping headroom 明显，简单
  attention confidence 无法利用它。

### H3：固定 raw action 的 signed net value 可在 source-held-out 条件下学习

- 候选：一个预先固定的 L2 logistic head，仅为 raw-attention 已选动作
  预测 `delta_success - 0.05 > 0`；样本权重按绝对 net utility，每个
  source 总权重相等。
- 固定设置：`C=0.01`，5 个 whole-source OOF folds，seed `20260918`，无
  特征/模型/正则化搜索；在 0.5/1/2/5/10% 相同 pooled call budget 比较
  entropy stopping，20,000 次 whole-source bootstrap。
- 解释：这是 opened official-train 上的 exploratory OOF 候选。只有在预先
  固定的决策条款下通过，才能冻结到独立 calibration；不可事后选择
  有利 call rate。
- 实现：commit `0683526`；真实输入 smoke 已通过，无模型拟合或策略
  指标计算。唯一主检验点为 2%（479 calls）。

## 决策树

1. Fixed-action privileged ceiling 已证实明显，max/margin 已证实无增益；
   因此进入低容量、whole-source OOF 的 fixed-action signed-value stop 模型。
   只预测“是否执行已固定动作”，不再同时学习四动作排序。
2. 若 H3 在所有注册预算仍不能优于 entropy 或不能保持正 utility：
   停止该 stop 学习路线，不在当前 train outcomes 上继续搜索模型。
3. 若简单 confidence 已稳定优于 entropy：优先冻结无训练或单调小模型，减少
   多重比较与过拟合风险。
4. 若 ViCrop/LASER where-only 通过：先做独立 calibration；stop 模型作为后续
   消融而非改变已通过的主候选。
5. 若所有 where-only 候选失败且 stop ceiling 明显：主路线切换为“显式分离
   stop 与 where 的反事实工具价值学习”；否则转向“强基线下的瓶颈审计与风险
   控制”包装，并评估其是否具备顶会新颖性。

## 紧接着的行动

1. H3 的冻结协议、实现和无 performance inspection 的真实输入 smoke 均已
   完成；提交完整 source-OOF signed-value stop 评估。
2. 持续监控 `203273` checkpoint、吞吐、磁盘与 8 小时时限；完成后自动合并审计。
3. 提交 literature Bonferroni evaluator，绑定 feature job 的旧 commit 与当前
   evaluator commit。
4. Literature 和 H3 的结果分别写入 `PROJECT_STATUS.md` 与 `EXPERIMENTS.md`，
   不在结果出来后选择有利阈值或模型族。
