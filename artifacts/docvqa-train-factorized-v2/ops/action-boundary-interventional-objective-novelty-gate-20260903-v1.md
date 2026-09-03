# Action-boundary interventional objective 新颖性与零支持 gate v1

状态：2026-09-03 14:24 HKT 完成。决定为
`action_boundary_candidate_reduces_to_existing_objective_families`。N0 在实现主方法或
提交 GPU 前关闭；本审计不改写 Job `206205`/`206227`，不打开 protected split。

## 候选的最强形式

令 `s` 为首个 answer/tool 边界前的完整多模态状态，`A(s)` 为 `ANSWER_NOW` 与所有
finite typed visual macro-actions。每个动作的序列分数为其 canonical token sequence 的
teacher-forced log probability，归一化得到 `p_theta(a|s)`。对每个动作执行 same-prefix
干预并固定 continuation protocol，得到 cost-adjusted utility `U(s,a)`。

候选原本希望同时做到：

1. 不依赖当前 policy 采到 parser-valid action；
2. 用具体 action/observation 的 signed intervention effect，而非 question-level necessity；
3. 只在 action boundary/action tokens 上更新，不污染 answer tokens；
4. 不是普通 SFT、DPO、listwise reward、utility router 或 Q regression。

## 穷尽目标后的形式化二分

### A. 直接最大化完整干预期望效用

`J(theta|s) = sum_a p_theta(a|s) U(s,a)`，其对 macro-action logit `z_a` 的精确梯度为：

`dJ/dz_a = p_theta(a|s) [U(s,a) - E_p U(s,.)]`。

这确实可在已经枚举全部干预效用后精确计算，不需要 Monte Carlo 采到该动作；但它仍由
当前动作概率乘权。对近零支持的有效工具动作，梯度随 `p_theta(a|s)` 同阶趋零；数值
underflow 为严格零时梯度也严格为零。它不能解决 Job `206205` 暴露的 support collapse。

### B. 用 utility 构造外部 target 绕过支持

令 `q_tau(a|s) = softmax(U(s,a)/tau)`，最小化
`CE(q_tau, p_theta)`，则梯度为 `p_theta-q_tau`。它在当前 `p_theta(a|s)=0` 时仍可对
高效用动作产生非零梯度，但原因正是：干预效用已经变成离策略监督标签。这是
utility-weighted/listwise policy projection；one-hot 极限是 best-action SFT，成对极限是
DPO/ranking，连续权重版本属于 advantage-weighted regression。把 loss 只落在 action
tokens 不改变目标族。

### C. 单独回归 action value 或边界 head

若训练 `Q_phi(s,a)≈U(s,a)` 后 `argmax_a Q_phi`，或新增一个 categorical macro-action
head 预测最佳动作，问题就成为 full-information contextual bandit/value regression。
在本项目语境中，它也重新落入此前失败的 action-value/router 路线，并与 GapSight 的
pre-crop gate、utility 与 box heads 直接重叠。它可作强 baseline，不能单独构成 N0。

因此不存在同时满足“零支持非零梯度”和“没有外部离策略/listwise/value监督”的当前
候选。强行声称无偏会混淆两个分布：干预可以识别 `U(s,a)`，但 policy-gradient 仍受
`p_theta(a|s)` 权重；移除该权重就是换成监督投影，而不是同一无偏 on-policy gradient。

## Dependency-free 数值 gate

三动作环境固定为：

- actions：`answer_now / beneficial_tool / harmful_tool`；
- utilities：`0 / +0.95 / -1.05`；
- near-zero logits：`0 / -20 / -20`；
- exact-underflow logits：`0 / -1000 / -1000`；
- utility-target temperature：`0.25`。

结果：10/10 checks 全真。

- beneficial tool 概率：`2.0611536e-9`；直接期望效用梯度：`1.9580959e-9`。
- exact-underflow 下 beneficial probability 与期望效用梯度均为 `0`。
- Boltzmann target 给 beneficial action `0.9777979` 概率；其 listwise CE logit gradient
  为 `-0.9777979`，在 exact-underflow 下仍同量级非零。
- 两个解析梯度均通过 central finite-difference 检查；最大绝对误差分别为
  `2.34e-18` 与 `1.51e-9`。

机器报告 SHA-256：
`c1bfd08a571cab4dc8d5f017e681434e6fd7caf364808e7e5ffbb85dc474e4f1`。
实现/runner/test SHA-256：
`a4bbbf785d717aa560d647549eded09816132522f524247bf8ad717e585c3a2d` /
`186ebfed8cc9727872e26300120213fe37daaf9f7075f36bfcda42f3b4a04d53` /
`9e8f17c80cdb36f6defdeced22d30152a7a0e53e3bc1c3aa609314db0c76416a`。

## 一手文献碰撞

1. [ToolVision](https://arxiv.org/abs/2608.08907) 已在 SFT 搜索中用跨模型 committee 的
   stepwise evidence gain 排序/剪枝，并在 RL 前用 frozen learner 的 paired
   with/without-tool rollouts 构造 MUT 权重。其 full text 明确把 SFT 的作用描述为先把
   useful behaviors 放入 practical support，再让 RL 放大它们。N0 不能声称首次用
   evidence benefit 解决 support collapse。
2. [The Illusion of Visual Tool-Use](https://arxiv.org/abs/2608.06270) 在固定 prefix 与
   action 下替换 observation，并以 real/counterfactual answer-margin 增量之差定义
   Visual Evidence Gain。Same-prefix observation intervention 本身已不是新 estimand。
3. [GapSight](https://arxiv.org/abs/2608.21762) 离线执行 candidate crop bank，用 target
   answer NLL/option-margin gap 产生 preserve/review、utility 与 box supervision，再从
   pre-crop global state 学 gate/router。把 task score 换成 signed utility 或换成 typed
   action bank仍属于高度相邻的 utility-router 家族。
4. [LIRE](https://arxiv.org/abs/2405.13516) 直接把多 response 的 offline rewards 纳入
   listwise optimization；[LiPO](https://arxiv.org/abs/2402.01878) 把多 response 排序统一
   为 learning-to-rank，pairwise DPO 是其特例；
   [ToolPrefer-LLaMA](https://arxiv.org/abs/2406.07115) 已从 inference tree 的成功/失败
   分支构造 step-wise tool preferences。候选 B 的 target 来源更 causal，但优化族不新。
5. [Advantage-Weighted Regression](https://arxiv.org/abs/1910.00177) 已用 value regression
   加 advantage-weighted target-action maximum likelihood 利用 off-policy 数据。
6. [Tool-RL collapse](https://arxiv.org/abs/2606.26027) 已系统研究 off-policy、hint、错误
   轨迹与 interleaved SFT 等 supervisory signals 对 tool structure collapse 的修复。
   用 forced/canonical typed actions 注入支持是必要 baseline，而非 N0 新颖性。

上述工作没有一篇与本项目全部数据合同完全相同，但顶会方法新颖性不能只建立在
“相同通用目标、不同视觉工具和 reward label”上。

## 决定与边界

- N0 正式关闭，不实现大模型训练版，不提交 GPU，不把 token-local mask 工程包装为方法。
- 保留 G0/G1 的完整负结果；same-prefix credit 仍可作为诊断量或 future comparator。
- V3 concrete typed prompt/structured decoding只允许承担 reliable baseline，不改变本决定。
- 下一信息价值最高的候选改为 N1 benchmark/estimand gate：审计现有完整 sibling bank
  是否足以定义并规模化 **stop regret / action-selection regret / evidence-use regret** 三项
  可识别分解。它必须明确区别于 The Illusion 只干预 policy 实际选择动作后的 observation，
  也区别于 GapSight 把 bank 折叠成 proxy-best crop。
- N1 在任何新生成前先盘点已有数据集、backbone、action family、source 数、完整 sibling
  覆盖与强 baseline；若达不到至少多数据集、多 backbone 和公开可复现规模，继续关闭，
  不降低 ECCV/ICCV/CVPR 目标。
