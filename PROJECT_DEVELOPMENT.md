# 项目发展记录

更新时间：2026-09-03 17:47（Asia/Hong_Kong）

## 项目要解决的问题

本项目研究视觉语言模型在回答问题时，何时应获取额外视觉信息、应查看哪里，以及一次
视觉工具调用是否真的改善最终答案。核心量是具体动作的任务收益减去执行成本，而不是
把预测熵下降、attention 强度或工具调用次数本身当成成功。

项目目标始终是形成达到 ECCV/ICCV/CVPR 标准的新颖方法与可靠证据链。当前代码、数据
合同和评测基础设施已经比较完整，但尚无可投稿的 deployable 正主结果。以下记录按研究
问题的演化整理，而不是只列代码提交。

## 第一阶段：建立 counterfactual sibling 基础设施

最初路线把同一问题的 `ANSWER_NOW` 与多个 `ZOOM` 动作组成完整 sibling bank，直接测量
每个动作是否救回、伤害或不改变答案。工程上完成了 image/source 级拆分、不可变数据
哈希、成本敏感 utility、entropy/fixed/random/exhaustive 基线、source bootstrap 和
Qwen2.5-VL 执行链。四个主要开发 bank 共包含 59,949 decisions、299,745 action rows。

这一阶段建立了一个稳定事实：存在可由正确视觉动作释放的显著 privileged headroom，
但“有 oracle headroom”不等于能在动作前识别它。

## 第二阶段：从 entropy gate 走向 where/when 因子化

ChartQA 的 factorized stopping gate 曾在独立高功效 replication 上得到正 utility，说明
稀疏的 when-to-call 信号并非完全不存在。然而 DocVQA、TextVQA 和 InfographicVQA 的后续
证据显示，跨 source 的 action selection 与 tail precision 更困难。

InfographicVQA 上，raw attention 的 crop action 优于旧 where 基线，但净 utility 仍为负；
ViCrop 与 LASER 两个预先冻结的文献 attention 候选也没有显著改善。固定 raw action 后，
privileged stopping ceiling 为 `+0.021318`，但 attention confidence 与低容量 signed-value
router 都无法稳定利用。由此关闭了 attention layer/head、阈值、call-rate 和相近线性
classifier 的局部搜索。

## 第三阶段：same-prefix action credit 与真实 RL 链路

项目随后尝试把同一 agent prefix 下 factual visual observation 与 no-op/counterfactual
observation 的 signed、cost-aware 差值，只分配给工具动作 token。Synthetic mask、符号、
pair provenance 与 token-local gradient 均通过，VTool-R1 只作为训练骨架和 outcome-only
对照，不再作为需要逐像素复现的研究目标。

真实四卡链路先后暴露并修复了 attention backend、`DataProto.chunk()` 类型合同和
checkpoint 空间 gate。最终 Job `206205` 完成两步 optimizer、64 条 rollout 和唯一
`global_step_2` checkpoint，但 parser-valid 工具调用为 `0/64`，因此方法特有的 credit
通路从未激活，并按预注册 `<1%` 规则停止。事后诊断找到 16 条裸工具意图，但没有一条
符合真实 API 签名；typed-action V2 的独立 H800 smoke 同样为 0/16 参数合法。该路线没有
通过改 prompt、seed 或阈值追结果。

## 第四阶段：N0--N3 零成本候选筛选

- N0 证明直接 expected-utility gradient 在零支持处消失；能恢复非零梯度的替代会退化为
  已有 listwise、AWR、SFT 或 value-router 家族，因此在 GPU 前关闭。
- N1 证明现有 sibling bank 可识别 stop 与 action-selection regret，却缺少识别
  evidence-use regret 所需的 fixed-prefix matched observation/continuation，也缺少主证据中的
  多动作族、随机重复和同数据集多 backbone 因子。
- N2 证明 stop/selection regret 可以严格相加，但 prefix/evidence 项是 signed causal
  effects，不是非负 regret；真正的 evidence-use regret 依赖未观测 ideal continuation，
  且核心主张与同期工作碰撞。
- N3 确认公开 VTool 3B/7B checkpoint 可获得，但 exact prompt/parser/execution provenance
  不完整；更重要的是 signed tool value、token responsibility、action counterfactual 与
  with/without-tool supervision 已被相邻工作覆盖，所以没有为换 initializer 消耗 GPU。

这些负结论缩小了可行主张空间，也避免了用更多算力重复一个在新颖性或可识别性上已经
失败的方向。

## 第五阶段：N4--N5 selector information boundary

N4 将 selector 动作前可见的信息集显式纳入评测合同，并要求只有在相同 visibility、
action bank 与完整成本定义下才能给方法主排名。形式化 toy gate 14/14 通过，并能检测
跨信息集排序反转；但 aliasing 理论和 cost ledger 已有直接或部分文献覆盖，所以必须由
真实数据上的实质效应决定是否继续。

N5 在读取逐 decision 配对结果前冻结了回顾性否证协议。DocVQA 1,608 decisions、400
sources 的共同 5% 调用预算下，source-balanced 结果为：

- context-geometry utility：`-0.00274318`；
- semantic-context utility：`-0.00296742`；
- higher-minus-lower：`-0.00022424`；
- paired 97.5% CI：`[-0.00528886, 0.00430109]`。

8 项科学晋级条件全部失败。Question-weighted 差为 `+0.00113272`，但 source-balanced 后
反号，说明表面改善集中在 QA 数较多的 sources，不能当作稳健跨 source 效应。ScreenQA
OOF 差也只有 `0.00004824`，约比 `0.001` 门槛小 20.7 倍。项目因此没有打开 ScreenQA
risk-calibration 的 49,755 条 action records，也没有提交 GPU 或生成新 checkpoint。

同预算 privileged oracle 为 `+0.03117658`，再次表明正确动作确实有价值；当前失败的是
outcome-free、可部署选择器对这部分价值的跨 source 提取能力。N4/N5 当前候选已关闭。

## 当前状态与下一步原则

项目仍然存活，工程和评测资产可复用，但科学状态是“尚无可投稿正主结果”。已经关闭的
路线不会靠换随机种子、调阈值、增加相似特征或扩大模型容量重开。ScreenQA calibration、
formal-test 与 reserve 继续封存。

下一步重新回到零成本 problem selection。新候选必须同时满足：

1. 能解释“privileged oracle 很高、现有 deployable router 跨 source 失败”的残差结构；
2. 相对近期视觉获取、工具价值学习和 counterfactual credit 工作有不可约新颖性；
3. estimand 可识别，并能在现有开发资产上给出可被推翻的最小预测；
4. 先通过强基线、成本与泄漏审计，再获准打开 calibration 或使用 GPU；
5. 保留负结果，并同时报告 question-weighted 与 source-balanced 推断。

详细数字、哈希和复现命令见 `EXPERIMENTS.md` 及对应 `artifacts/**/ops/` 审计文件。

## 第六阶段：收敛为固定工具的 pre-action predictability audit

此前路线不断在 `where`、工具 RL、representation 与 benchmark framing 之间切换，虽然
积累了大量负证据，却没有直接回答最初最关键的问题：在动作前可见的信息里，到底有没有
稳定的工具效用信号。项目因此停止开放式 N6 候选生成，改成一次封顶的 36-cell 审计。

新的 binary task 不学习 crop：`USE_VISUAL_TOOL` 固定执行四个 UG-grid crops，并用已有
entropy-search 规则返回一个答案。标签只包含成本无关的 `Y0/Ytool/gain/rescue/harm`，
`lambda` 仅在 policy time 使用。四级 predictor 从 entropy/maxprob/margin，逐步增加浅层
问题状态、全局语义与冻结 Qwen 多模态状态；三个 target 则区分直接 gain、rescue/harm 与
factorized error-rescue-harm。这使“信号不存在”“只存在于更深 representation”“只在调用后
出现”成为可以被同一矩阵区分的假设，而不是靠继续换 feature 猜测。

第一版协议、typed leakage boundary、固定工具 outcome collapse、36-cell completeness
checker 与唯一终局 verdict 规则已经实现并通过单测。对旧 bank 的只读 smoke 证明
ChartQA、DocVQA 与 V* proxy 都有正 binary-oracle headroom，但 always-call 在四次成本下
显著为负。旧 bank 只能验证管线：它们缺完整 L0/L3 特征、HRBench 与 untouched test，故
正式矩阵仍为 `0/36`。下一阶段的工作是构造无像素/source 重叠的新 split 和统一 feature
export，然后一次性跑完冻结矩阵；矩阵完成前不再把任何局部结果解释成项目成败。

随后 evaluator 和固定训练器也已贯通：三个 target family 使用 source-balanced sample
weight，模型 variant 与 calibration/threshold 只在 validation 选择；test 只做冻结推断，
同时生成 AUROC/AUPRC/Brier/calibration、rescue/harm、policy curve 和 paired source
bootstrap。Synthetic 三 benchmark 的 36 cells、三个 seeds 共 108 次训练/评估已经完整
通过，证明 hard matrix 可以端到端执行；报告明确为非科学 smoke，不增加正式 `0/36` 计数。
