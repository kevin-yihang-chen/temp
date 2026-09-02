# 项目状态

更新时间：2026-09-02 21:46（Asia/Hong_Kong）

## 总体判断

项目仍然存活，但原始“训练一个 deployable pre-call value/gate 即得到顶会正结果”
路线现在应视为高概率失败，尚未形成 ECCV/ICCV/CVPR 可投稿主结果。最新
literature-attention 实验否定了 ViCrop/LASER；随后 answer-conditioned hidden-state
候选又因直接文献碰撞在实验前关闭。继续局部调 attention、hidden-state probe、
classifier 或 threshold 没有科学价值。

这不是工程失败，也不是证明研究问题不存在。完整 sibling outcomes 显示有大量
可获益状态，固定 raw action 的 privileged stopping utility 上界也显著为正；
失败点是 deployable pre-action prediction 无法跨 source 稳定识别稀疏正收益尾部。
当前最有价值的产出是严格的 stop/where 因子化与负结果证据，而不是一个成功策略。
下一步已转为机制不同的视觉工具 RL action-local credit feasibility，而不是给旧
gate 换特征。该方向现在只完成 protocol 与纯 synthetic G0，尚无训练趋势或正式
方法结果，不能据此提高项目成功概率。

2026-09-02 已做范围纠偏：VTool 只保留为 Apache-2.0 的可运行 RL 骨架和
outcome-only comparator，不再把 pixel、thought 或内部实现与 VTool 的一致性当作研究
问题。此前继续做等价性审计属于范围漂移；它既不验证 H5，也不构成论文贡献。唯一
核心 empirical question 是：same-prefix signed action credit 能否在冻结的强基线与
matched controls 下改善任务分数、cost-adjusted utility 和 harmful-call rate。

当前 pre-GPU 工程 gate 已前进：官方 Apache-2.0 ReFocus train 已固定，token-local
adapter 的真实 autograd、隔离 runtime import/version/worktree、单行真实 Qwen processor
和 paired agent fake-server contract 均已通过。尚未完成 H800 vLLM 模型加载或任何
optimizer step。因此当前不是“新方法已成功/失败”，而是“关键性能实验尚未开始”。

## 已完成的证据链

1. InfographicVQA official-train 已完成 23,946 decisions、2,204 sources、4,406
   images 的完整 sibling bank 与 source-level audit；validation/test/reserve 未开。
2. DECAR nested OOF（Job `203049`）正式 `decar_not_advanced`；0.5% utility
   `-0.000273`，95% CI `[-0.000460, -0.000103]`，主要失败为 stopping enrichment。
3. Entropy-when / oracle-where 诊断（Job `203078`）得到
   `where_bottleneck_supported`，证明 entropy 所选状态中存在可由更好 action
   selection 释放的正收益，但该 oracle 不可部署。
4. Relative-where OOF（Job `203237`）失败，teacher-action agreement 仅
   17.4%--22.8%，暴露 source-generalization 问题。
5. Raw-attention 特征与评估（Jobs `203257`/`203276`）完成。Raw action 在 5%/10%
   显著优于 fixed、random、old-DECAR 与 relative where，但所有净 utility 为负；
   5% 为 `-0.000410`，95% CI `[-0.002438, 0.001681]`。
6. Fixed raw-action stop diagnostic（Job `203290`）显示 unrestricted privileged
   stop ceiling `+0.021318`，95% CI `[0.018447, 0.024444]`；attention max/margin
   均比 entropy 更差。
7. Fixed-action signed-value OOF（Job `203330`）在唯一 2% primary 失败：utility
   `-0.000063`，95% CI `[-0.000739, 0.000655]`；相对 entropy 的 paired interval
   跨零。该线性模型族已关闭。
8. Literature attention 特征（Job `203273`）完整抽取并通过无泄漏审计；评估
   Job `203340` 得到 `literature_attention_where_train_not_supported`。ViCrop 与
   LASER 在 0.5/1/2/5/10% 的所有 utility 点估计均为负，也未显著优于 raw
   attention。完整结果见
   `infographicvqa-literature-attention-where-result-job-203340-v1.md`。
9. Answer-conditioned evidence outcome-free feasibility audit 完成。代码层面可从同
   一 generation 返回 answer hidden states，但 ContextualLens、LRP、VRP、V-Loop
   等已直接覆盖核心 representation/probe/grounding 组合；决定码为
   `answer_conditioned_evidence_candidate_rejected_before_experiment`。没有模型拟合、
   新 outcome 或 GPU job。
10. VTool-R1 `training-v2` 已只读浅克隆并固定到
    `d2aa28353ec10c7f91b39f502925003a81d6982d`。静态 gate 确认 upstream 默认把
    tool/action tokens mask 掉；本项目已通过最小 patch 和独立 adapter 新增
    token-local credit 通路。VTool 到此只承担底座/对照职责，不再继续等价性审计。
11. Same-prefix action-credit protocol v1 与 dependency-free G0 core 已在 commit
    `56b990c767973a8a23060d63293db8657254b35d` 冻结。17 项新测试覆盖 exhaustive
    token roles、arm-specific cost、swap antisymmetry、完整 provenance、序列化防
    篡改、token-local advantage 与 shuffled no-self-donor；完整仓库共收集 509 项，
    479 passed、30 项依赖/资源相关预期 skip。该证据只支持实现定义一致，不支持
    方法性能。
12. 官方 `ReFocus/ReFocus_Data` train 的 Apache-2.0 license、revision 与三个 shard
    SHA-256 已固定；旧 derivative 的 metadata 对照只作为一次性转换证据。无需也不会
    再证明它与 VTool 的 pixel/thought 一致。正式训练输入只允许 official train。
13. 审计期间曾误读 Refocus test 的非图像 metadata/ground truth，并随后一次性枚举
    original ChartQA val/test PNG path IDs。事件已透明登记和隔离；这些 split 永久不能
    再作为本项目 sealed formal evidence。没有拟合、方法选择、GPU 或结果使用。
14. 新增 upstream-shaped adapter 和 paired agent overlay；PyTorch autograd smoke
    证明 signed credit 对 action tokens 产生非零梯度，zero control 为零，observation/
    padding 梯度为零。隔离环境通过所需 imports、版本、pinned commit 与最小 patch
    检查。该证据仍不代表真实 rollout 或性能通过。
15. Official-train converter 重新验证三个 shard 的完整 SHA-256，只读取 train；policy
    输入排除 answer、thoughts、edited image 和 teacher focus boxes。冻结 64 个 train
    structural groups（72 rows）与 32 个 curve-eval groups（33 rows），交集为 0。
    Paired 与 outcome-only Parquet 除 `agent_name` 外逐字段相同。
16. 单行真实 Qwen processor smoke 通过：1 张原图、966 prompt tokens、无 focus-area
    泄漏。Paired fake-server contract 通过：rescue `+0.95`、harm `-1.05`、失败/无收益
    `-0.05`、direct `0`；shared prefix/seed、image-only delta、union mask 和 fail-closed
    均验证。四组 G1 配置已在 `configs/vtool_action_credit_g1_v1.json` 冻结。

## 当前最佳结果与解释边界

- 最强 deployable `where`：raw-attention action；它显著超过四个旧 where 基线，
  但在现有 stopping 下仍为负 utility，不能进入 calibration。
- 最清晰机制证据：固定 raw action 后，privileged stop 上界 `+0.021318`；这证明
  “值得调用的状态存在”，但不证明它们可由当前特征预测。
- 最新 OOF stop 候选有轻微 precision 改善，但自身 utility 与 paired lower
  endpoint 均未过门槛。
- ViCrop/LASER 是有效的 literature strong-baseline negative，而不是新方法成功。
- 目前没有可宣称正结果的 deployable candidate，validation/test/reserve 必须继续
  封存。

## 距离顶会目标

| 里程碑 | 当前状态 |
| --- | --- |
| 严格数据/无泄漏/强基线基础设施 | 既有数据完成；Refocus train audit 通过，test 已污染并隔离 |
| Action-credit protocol、synthetic G0 与 upstream adapter | pre-GPU contract 已完成；尚无 optimizer step |
| 官方 train license/identity 与可执行 runtime | 已通过；不要求 VTool pixel 等价 |
| 真实 converter 与 paired fake-server | 已通过 |
| 单卡 H800 vLLM model-load/generation preflight | 待运行；G1 暂不授权 |
| 可部署方法在 source-OOF train gate 取得正且显著 utility | 未完成 |
| 独立 calibration 通过 | 未开始；无候选获授权 |
| Sealed formal 一次性通过 | 未开始 |
| 第二数据集/骨干与外部方法比较 | 未完成 |
| 完整论文主张 | 仅有诊断与负结果骨架 |

所以当前离“可以承诺顶会结果”仍隔着至少 method gate、calibration、formal、
generalization 四个实质台阶。现在不能承诺日期。

## 正在运行

2026-09-02 20:41 HKT 的实时 `squeue -u yihangc` 为空；当前没有运行或排队的
Slurm job。Jobs `203273` 与 `203340` 均已正常完成，计算状态邮件已按全状态合同
配置。当前修改只保留在本地，未 push GitHub。

同日 20:41 HKT 的 live quota 为 GPU 222,000 分钟总额、42,330 已用、
179,670 剩余（2,994.5 GPU-hours，19.07% 已用）；association 上限为 4 GPU、
4 H800、48 CPU。算力足以支持一次有界 3B matched-control RL 研究，但科学 gate、
依赖与训练稳定性仍是主要风险。

## 已关闭的路线

- 当前 fixed four-box bank 上的 DECAR v1、relative-where、raw/literature pure
  attention where-only gate；
- attention layer/head/ratio、max/margin、entropy threshold 与 call-rate sweep；
- 当前 80 维特征上的线性 signed-value stop family，包括更换 C、权重、seed 或
  classifier family 的事后搜索；
- answer hidden-state/contextual embedding/grounding reliability probe 作为独立新
  方法；generic group-DRO、IRM 或 conformal threshold 的局部替代；
- 用已打开 train outcomes 选择有利 operating point，或用 privileged oracle
  冒充部署结果。

## 主要风险

- Positive-net calls 稀疏且跨 source 异质，现有表征只能得到弱排序信号。
- 固定四格 action bank 可能限制 proposal quality；但更丰富 proposer 会增加成本，
  并与 GapSight、CropVLM、AdaptVision 等工作产生更强新颖性碰撞。
- Generic pre-call classifier、necessity/harm learning、attention crop、hidden-state
  reliability probe 与 continuous crop routing 均已有直接相关工作；仅换模型不足
  以构成贡献。
- 即使新的 train OOF 候选通过，仍需独立 calibration、sealed formal 和至少一个
  generalization axis，时间不只由单次 GPU runtime 决定。
- Same-prefix action credit 与 ToolVision 的 stepwise evidence gain 存在强碰撞；
  若不能证明具体 action 的 signed causal contrast 和 token-local training 带来
  matched-budget improvement，该硬转向也会失败。
- Upstream FP8 3B script 默认把 `test.parquet` 同时作为 train/val；直接运行会造成
  明确测试泄漏。必须只用 official train 派生开发 split，并关闭 train-time test。
- 旧 Refocus_Chart derivative 不再是训练数据；正式输入改为已固定 revision/hash 的
  Apache-2.0 official ReFocus train。Refocus test 与 original ChartQA test 已因 metadata
  暴露失去 sealed 资格，formal 必须使用从未打开的独立 benchmark/split。
- 隔离 VTool 环境、真实 processor 和 fake paired generation 已通过；但 vLLM 模型
  权重加载、显存、真实 generation、checkpoint/resume 与吞吐仍未经过 GPU smoke，
  不能把 CPU/fake-server 成功等同于训练可行。
- 既有观测的 1%--3% tool-call rate 可能让 action pairs 过稀；若真实 G1 smoke 低于
  1%，不能靠事后改 prompt/temperature 制造正结果，必须重新审计 exploration 假设。

## 下一步最优行动

不再做 VTool 等价性审计。Converter、fake-server 与 matched-control config 已完成；
下一步是已绑定一条 official-train 样本、pinned model/runtime 和全状态邮件的 1×H800
vLLM model-load/generation smoke。只有它通过，才允许最多 2-step 的 4×H800 G1。G1
只回答实现与早期学习信号是否成立；若 signed 不能在 matched controls 下改善冻结
指标，就关闭 H5，而不是继续追 VTool 一致性。
