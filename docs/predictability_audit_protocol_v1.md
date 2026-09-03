# Pre-action 工具效用可预测性审计协议 v1

冻结时间：2026-09-03 17:20（Asia/Hong_Kong）

## 唯一研究问题

本轮不再寻找新的 crop 选择器，也不学习 `where`。唯一问题是：在视觉工具执行前，
VLM 的状态能否跨图像、跨 source 稳定预测一个固定视觉工具调用的实际效用？决策只有
`ANSWER_NOW` 与 `USE_VISUAL_TOOL`。

固定工具为现有 `EntropySearchPolicy`：执行四个冻结 UG-grid crop，支付四次调用成本，
再按 post-action entropy 最低者返回答案；并列时按 `action_id` 升序。post-action
entropy 只属于工具内部与标签构造，任何 deployable predictor 都不可读取它。

每个状态保存成本无关的 `Y0`、`Ytool`、`gain=Ytool-Y0`、`rescue=(gain>0)` 与
`harm=(gain<0)`。`lambda=0.05` 仅在 validation 阈值选择和 policy utility 评估时使用，
不进入预测标签。

## 冻结实验矩阵

正式矩阵严格为 3 个 benchmark × 4 个 predictor level × 3 个 target，共 36 个 cell：

- benchmark：ChartQA、DocVQA、HRBench；
- L0：entropy、max probability、top1-top2 margin；
- L1：L0 加浅层 question/state 特征，固定 logistic 与小 MLP；
- L2：question embedding、global visual embedding、逐元素交互与 uncertainty，固定小 MLP；
- L3：冻结 Qwen language、pooled visual、fused multimodal state，固定 linear 与两层 MLP；
- target：direct gain、rescue/harm 双头、error/rescue/harm factorization；
- seed：17、29、47，最多三个，不得以换 seed 搜索结果。

Factorized target 冻结为 `error=(Y0<1)` 与 `correct=(Y0==1)`。训练样本按每个 source 的
question 数倒数加权并归一到均值 1。每个 raw score 只在 validation 上用 logistic 分别
校准 positive-net、rescue 与 harm 概率；这一步不改变成本无关的训练 target。

另只允许一个 post-action oracle probe：固定两层 MLP 预测 direct gain。它仅用于诊断
“信息是否只在工具执行后才出现”，永远不能作为 deployable 方法或主结果。

## 数据与泄漏边界

每个 benchmark 必须具有 train/validation/test 三个角色，按 `source_id` 与解码后 RGB
SHA-256 的联合连通分量隔离；任意角色间 source 或像素内容重合都必须为零。validation
只可用于模型变体、阈值与 calibration；test 只允许在所有选择冻结后运行一次。

历史上已经打开 outcome 的 ChartQA test、V*Bench bank、DocVQA development/formal
数据只能用于 retrospective smoke，不能进入最终 test。旧 `.pt` feature bundle 即使同时
保存 outcome，也必须先经过 typed allowlist adapter；runner 不得把原始 decision dict
交给 predictor。任何 `_after`、gain、rescue、harm、label 或 outcome 派生字段进入输入
时立即失败。

## 指标与强基线

预测指标固定为 AUROC、AUPRC、Brier、calibration error、rescue AUPRC 与 harm AUPRC。
policy 指标固定为 accuracy、cost、call rate、utility、rescue precision、harm rate 与
marginal gain/call，并画 accuracy-cost-call-rate 曲线。主 endpoint 是 source-balanced
incremental utility；不确定性使用 20,000 次 paired source bootstrap 的 95% CI。

强基线包含 answer-now、entropy gate、random gate、matched-gate fixed crop、uniform
random crop expectation，以及按四次成本收费的 exhaustive UG entropy search。阈值只在
validation 上按 utility 最大、调用更少、阈值升序的顺序确定。

## 当前证据与缺口

现有 ChartQA、DocVQA、V*Bench sibling bank 足以验证固定工具的标签构造和 runner，但不
构成正式矩阵：旧 rollout 没有 max probability/top1-top2 margin，缺 L3 完整状态，部分
旧 split 也不满足 RGB/source 双重隔离，且 HRBench 尚未生成。故在新数据和特征完成前，
矩阵完成度必须报告为 `0/36`，不得从旧负结果直接宣布该 research question 失败。

## 唯一终局分类

只有 36 个 cell、强基线、paired bootstrap 与一次性 test 全部完成后，才能生成
`PREDICTABILITY_AUDIT.md` 并给出一个结论：

- `STOP`：至少两个 benchmark 的 privileged binary oracle utility 不超过 `0.005`；
- `GO`：至少两个 benchmark 上 best deployable 相对 strongest baseline 的 paired 95%
  lower CI 大于零，并通过已注册的 Pareto/rescue/harm 条件；
- `REPRESENTATION`：至少两个 benchmark 的 L3 in-domain improvement lower CI 大于零，
  但至少两个 benchmark 的 image-disjoint 或 cross-domain upper CI 不大于零；
- `PIVOT`：至少两个 benchmark 有大于 `0.005` 的 oracle utility，post-action probe 的
  utility lower CI 大于零，而全部 deployable predictor 均失败。

判断优先级冻结为 `STOP → GO → REPRESENTATION → PIVOT`。若完成后仍不满足任何规则，
runner 必须报 inconclusive 并拒绝生成终局分类，而不是临时改门槛。
