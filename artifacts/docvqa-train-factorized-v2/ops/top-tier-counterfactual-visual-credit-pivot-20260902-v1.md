# Top-tier-only counterfactual visual-action credit pivot v1

状态：2026-09-02 路线选择审计。本文不授权训练、下载数据、读取封存 outcomes 或
提交 Slurm；它只冻结下一阶段必须先回答的 novelty/feasibility 问题。目标保持
ECCV/ICCV/CVPR main conference，不设置降低投稿档位的完成出口。

## 为什么必须硬转向

InfographicVQA 的 attention where、literature attention、simple confidence 与
low-capacity signed stop 均未得到正且显著的 deployable utility。随后唯一优先的
answer-conditioned evidence 候选虽然工程可行，却与 ContextualLens、LRP、VRP、
V-Loop 等工作直接碰撞。继续在 pre-call classifier 上换表征、模型容量、domain
weight 或阈值，既违反既有止损规则，也不足以形成新颖方法。

## 唯一优先研究对象

下一阶段只审计 **same-prefix counterfactual visual-action credit**：

1. 对一个已经产生的视觉 action，在完全相同的 agent prefix 上构造 factual
   observation 与预先冻结的 counterfactual observation（至少 no-op；可扩展为固定
   alternative action）。
2. 用二者对最终任务分数的 signed difference 减去显式 action cost，定义
   `A_visual`；保留 rescue、harm 和 zero-effect，不把它压成“是否需要工具”的
   question-level 标签。
3. 保留 outcome-only advantage 给 reasoning/final-answer tokens；`A_visual` 只施加
   于对应 tool/action tokens，observation tokens 不训练。
4. 以匹配数据、rollouts、optimizer steps、GPU-hours 和 tool-call budget 的
   outcome-only VTool-R1 分支作为主对照。

这与当前失败的 deployable gate 是不同命题：这里不要求一个小 classifier 在新
domain 直接预测 tail benefit，而是测试 causal local credit 是否能让 end-to-end
policy 学到更少伤害、更有效的 visual action。

## 与最近工作的区别及碰撞风险

- [VTool-R1](https://github.com/VTool-R1/training-v2) 使用 final outcome reward；
  pinned `training-v2` loop 当前把第一段 assistant tool tokens 全部设为
  `response_mask=0`。因此 action-token credit 需要新的 mask/advantage 数据通路，
  不是 reward-manager 参数改名。
- [ToolVision](https://arxiv.org/abs/2608.08907) 已有 committee-based stepwise
  evidence gain，以及 frozen SFT policy 的 question-level must-use-tool reward。
  我们不能声称“首个 stepwise evidence reward”或“首个 counterfactual tool
  benefit”。剩余可能区别仅是同 prefix 的具体 action/observation signed effect、
  cost、harm 与 token-local advantage 的联合。
- [AdaTooler-V](https://arxiv.org/abs/2512.16918) 已用 query-level Tool Benefit
  Score 缩放 GRPO reward；它排除了 generic benefit-weighted RL 新颖性。
- [AdaptVision](https://arxiv.org/abs/2512.03794) 已把 tool learning 与 answer
  accuracy advantage 解耦；它排除了 generic decoupled visual RL 新颖性。
- MED 与 The Illusion of Visual Tool-Use 已做 intervention-based gain/harm 诊断；
  如果同 prefix contrast 只用于分析而没有带来 matched-budget training improvement，
  也不足以成为主方法。
- GapSight 已用离线候选 crop 的 loss gap 训练 when/where router；仅把 task score
  换成 loss 或把 fixed crop 换成 free-form crop 不能构成区别。

因此该路线仍是高风险候选，而不是已经成立的新颖性结论。

## 代码与算力可行性

现有仓库已经有 VTool identity audit、gate metadata adapter、完整 sibling outcome
schema，以及对 pinned upstream commit
`d2aa28353ec10c7f91b39f502925003a81d6982d` 的 mask 审计。上游 README 明确提供
Qwen2.5-VL-3B 的单节点 4×H100 recipe；集群 H800/H100 分区存在。

2026-09-02 17:42--17:44 HKT 的实时资源快照：

- GPU quota：222,000 分钟；已用 42,348；剩余 179,652，即 2,994.2 GPU-hours；
  使用率 19.08%。
- CPU quota：2,664,000 分钟；已用 200,618；剩余 2,463,382 分钟。
- association limit：最多 4 GPU、4 H800、48 CPU；当前使用均为 0。
- `squeue -u yihangc` 为空；H800/H100 节点处于 mixed/allocated 状态，没有稳定的
  立即启动保证。

算力已不再是一次有界 3B matched-control 研究的主要阻塞点。主要风险是新颖性、
upstream dependency 兼容、same-prefix counterfactual 的额外 rollout 成本，以及
单个四卡账户无法同时运行独立 GPU judge。

### 17:53 HKT upstream 静态 gate 结果

已将 `training-v2` 浅克隆到只读 reference 目录并固定在
`d2aa28353ec10c7f91b39f502925003a81d6982d`。静态审计确认：第一段 tool/action
tokens 当前在解析成功后被改为 `response_mask=0`，observation tokens 也为 0；GRPO
则把单一 outcome scalar 广播到 response mask。因此 `A_visual` 若不新增
`action_mask`、`answer_mask` 与自定义 token-local advantage 通路，梯度必然为零。

工程 gate 判为
`upstream_static_feasibility_supported_with_dependency_and_credit_path_blockers`：修改面
有限且原则上可行，但 upstream 同时出现 vLLM `0.17.0` Docker、`<=0.12.0` setup
extra 与 `0.8.4` requirements 注释，必须先冻结权威 image/digest。Checkpoint/resume
入口存在，但 paired RNG 与 sampler 的精确恢复还未验证。详细证据见
`vtool-counterfactual-credit-upstream-feasibility-audit-20260902-v1.md`。

## 前置 gate 与止损

在任何训练前依次完成：

1. 浅克隆、固定 upstream 并完成代码/依赖/license 静态审计。已完成；带 dependency
   与 credit-path blockers，不等于训练授权。
2. 写 protocol，唯一 primary 比较 `outcome_only` 与
   `outcome_plus_signed_action_credit`；随机符号或 shuffled credit 作为负对照。
3. 用 synthetic trajectory 证明 action mask 只覆盖 action tokens，且同 prefix
   factual/counterfactual scorer 在交换两臂时严格变号。
4. 用极小真实 batch 做 4×H800 smoke，检查显存、吞吐、judge、checkpoint/resume
   和每种 branch 的实际 GPU-hours；所有 Slurm 状态邮件使用
   `yihangc@connect.hku.hk`。
5. 只有 method branch 在预注册的短程学习曲线同时改善 task score、cost-adjusted
   utility 与 harmful-call rate，且超过 outcome-only/shuffled control，才扩完整训练。
6. 完整结果必须在至少两个 benchmark 或一个 benchmark 加第二 backbone 复现；
   单一 ChartQA/VTool overlap split 不能支持论文结论。

若 novelty full-text audit 或最小 smoke/learning-curve gate 失败，关闭该路线并选择
新的实质方法/benchmark contribution；不能通过降低投稿档位把失败路线记为完成。
