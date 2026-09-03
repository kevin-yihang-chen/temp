# 项目状态

更新时间：2026-09-03 09:51（Asia/Hong_Kong）

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

当前工程 gate 已前进：官方 Apache-2.0 ReFocus train、token-local adapter 的真实
autograd、隔离 runtime、单行及完整 72 行 Qwen processor、paired fake-server、单卡
H800 vLLM 模型加载/真实首轮 generation，以及最终 Hydra resolved-config gate 均已
通过。Job `205902` 随后获得 4×H800，但在模型加载前因 worker 的 jq 对象全真断言
语法错误同秒 fail closed；没有训练输出目录、rollout、optimizer step 或 checkpoint。
因此当前不是“新方法已成功/失败”，而是“关键性能实验尚未真正开始，工程断言已定位
并修复，待新 revision 重提”。修复 commit `8c0f6c0` 的 clean-revision Hydra v13
已经以 59/59 resolved-config checks 全真通过，科学配置与 E-20260902-16 相同。

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
17. 单卡 H800 model-load/generation Job `205784` 以 `COMPLETED`、`ExitCode=0:0`、零
    restart 结束。精确 model/runtime/data/config binding 通过；vLLM `0.12.0` 加载 3B
    BF16 权重约占 7.16 GiB，完整 engine load `64.05s`，966-token 视觉 prompt 生成
    12 tokens 用时 `19.21s`。报告 SHA-256 `1a67365b...009a`。首轮为 direct answer，
    因此本项不验证真实 tool/paired branch，也不是性能证据。
18. G1 最终 preflight 已通过。完整 72 行 paired train 经冻结 `RLHFDataset` 与 Qwen
    processor 后全部满足契约：64 个 structural groups，prompt tokens 为 min 438、
    median 931、p95 1,657、max 1,914，小于 4,096；无 protected split、无权重加载，
    报告 SHA-256 `e468c367...c1ed`。Shuffled control 在 batch 少于两个 valid tool
    pairs 时 action loss 置零并计数；PPO mini-batch 已按 upstream prompt-count 语义从
    32 修正为 8。Hydra signed dry-run v9 的 59 项冻结配置检查全部为真，launch
    manifest SHA-256 `dff12612...3da`，resolved config SHA-256
    `5a5d354d...7b3b`。这只授权首次 4×H800、2-step signed G1，不是性能结果。
19. 首次正式提交 Job `205870` 后，在其仍为 `PENDING (Resources)` 时发现 upstream
    rollout writer 只导出 `reward_extra_info`，而当时其中只有 `acc`；若直接运行将无法
    保存 counterfactual score、signed credit、pair provenance、tool success 与
    harmful-call 证据。作业在启动前取消，最终 `CANCELLED`、`RunTime=00:00:00`、
    `Restarts=0`，未消耗 GPU。现已把全部核心字段封装为 JSON-safe audit payload，新增
    自动 analyzer 逐行重建 pair、验证 score/credit/trajectory，并输出 task score、
    cost-adjusted utility、harmful/rescue/tool-call rate。Fake-server v2 与 Hydra v11
    均通过；这是核心可观测性修复，不是 VTool 等价性审计。
20. 修复后提交的 Job `205902` 于 2026-09-03 00:56 HKT 获得资源，但 worker 在
    runtime dataset audit 断言处同秒退出，status `exit_code=2`、
    `scientific_decision=not_available`。原 jq 表达式
    `.checks | all(.[] == true)` 在对象输入上会对布尔值再次执行 `.[]`，以 exit 5
    报 `Cannot iterate over boolean`。相同 jq 与冻结 72 行 report 已稳定复现；改为
    `.checks | all(.[]; . == true)` 后，全真输入通过、含假输入拒绝。训练输出目录
    不存在，未发生模型加载、rollout、optimizer 或 checkpoint；本项不是 H5 结果。

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
| 单卡 H800 vLLM model-load/generation preflight | Job 205784 通过；仅授权有界 G1 |
| 真实 paired rollout 与最多 2-step optimizer smoke | Job 205902 在模型加载前因 jq 断言秒退；已修复，仍待新 revision 重提 |
| 可部署方法在 source-OOF train gate 取得正且显著 utility | 未完成 |
| 独立 calibration 通过 | 未开始；无候选获授权 |
| Sealed formal 一次性通过 | 未开始 |
| 第二数据集/骨干与外部方法比较 | 未完成 |
| 完整论文主张 | 仅有诊断与负结果骨架 |

所以当前离“可以承诺顶会结果”仍隔着至少 method gate、calibration、formal、
generalization 四个实质台阶。现在不能承诺日期。

## 正在运行

截至 2026-09-03 09:40 HKT，Job `205902` 已终止且不再出现在活动队列；当前没有
正在运行/排队的 G1。worker status 为 `failed`、exit 2、开始与结束 epoch 均为
`1788368160`，科学 decision 不可用。日志只包含 jq 的
`Cannot iterate over boolean (true)` 与 fail-closed 提示，训练输出目录不存在。
任务配置了 `--mail-user=yihangc@connect.hku.hk --mail-type=ALL`，因此开始与失败均在
Slurm 状态通知范围内。当前修复只保留在本地，未 push GitHub。

修复 commit `8c0f6c0` 的 Hydra v13 decision 为
`vtool_action_credit_g1_hydra_dry_run_passed`；launch manifest SHA-256
`29e24dabdbf4986c981a737297227d4f6b2ee365ce45c6e9a9b9aff7aca28fe8`，resolved config
SHA-256 `cda71307a9c6ebc7fd634114bf5cac76664723d0e124c83351f3d8a12680ac43`。
59 项检查全部为真，manifest 的 worktree status 为空、protected split 未访问。

09:40 HKT 的 live quota 为 GPU 222,000 分钟总额、42,242 已用、179,758 剩余
（2,995.97 GPU-hours，19.03% 已用）；association 上限为 4 GPU、4 H800、48 CPU。
accounting storage 已禁用，无法从 `sacct` 获得更细粒度计费，但同秒退出且没有模型
加载。算力仍足以支持一次有界 3B matched-control RL 研究。

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
- 隔离环境、完整数据 processor、fake paired generation、单卡 vLLM load/generation
  与 Hydra 配置解析已通过；但首轮仍是直接回答，没有验证真实 tool/paired branch。
  JSON-safe pair/utility export 已由 fake-server 与 synthetic analyzer 验证；Optimizer、
  Ray 多进程、checkpoint/resume 与真实 action-credit metrics 仍未经过 GPU smoke。
- Job `205902` 暴露出 shell jq object predicate 未被旧静态字符串测试实际执行；两处
  同类表达式现已修正并加入真实 jq 正/负回归；全仓测试与静态检查已通过。重新提交前
  仍需在 clean commit 上执行完整 worker/Hydra 前置合同，避免再次用排队换取可在登录
  节点发现的语法错误。
- 既有观测的 1%--3% tool-call rate 可能让 action pairs 过稀；若真实 G1 smoke 低于
  1%，不能靠事后改 prompt/temperature 制造正结果，必须重新审计 exploration 假设。

## 下一步最优行动

不再做 VTool 等价性审计。Job `205902` 的失败已定位为 jq object-values 断言语法，
不属于科学负结果。全量回归已通过；下一步本地 commit 后完成真实 jq/worker 前置
合同与最终 Hydra gate。修复 commit 的 Hydra v13 已通过；文档 commit 后在最终 HEAD
复跑相同 gate，实时复核 quota/queue/disk 后以新 revision 重新提交唯一 4×H800、
最多 2-step paired-signed G1。
若真实 G1 的 tool-call rate、pair validity 或训练稳定性触发冻结 stop rule，就关闭或
只修复可明确定位的工程问题；只有它们通过，才以同一 revision 运行
zero/shuffled/outcome-only controls，不事后改 prompt、seed、temperature 或指标。
