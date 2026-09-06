# 项目发展记录

更新时间：2026-09-05 17:34（Asia/Hong_Kong）

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

## 截至 N5 的状态与下一步原则

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

## 第八阶段：从单步 action ranking 转向共享前缀 stopping

Utility-SFT 的负结果说明“从原图一次性给多个 crop 排名”没有形成稳定跨域贡献，但它没有
回答已经取得部分视觉证据后是否能判断继续观察。新阶段因此把动作空间收缩为共享 prefix
上的 `STOP/CONTINUE`：已有 crop 与下一 crop 都由 outcome-blind 几何规则固定，模型只学
when，不再同时承担 where。

代码审计发现旧 sibling bank 的 baseline 是原图，ZOOM branch 是原图加一个 crop；它没有
原图加已有 crop的 STOP 分支，更没有与其配对的第二次 acquisition。因此旧 labels 不会被
重包装使用。新 collector 直接构造 2-image STOP 和 3-image CONTINUE，保留相同 seed 与完整
成本 ledger。critic 使用冻结 Qwen 的 question/global/ROI/current-prefix state，并通过 typed
allowlist 隔离所有 post-action outcome。

截至 2026-09-06 11:16 HKT，dependency-free schema、collector、risk/gain critic、stopping
policy、指标/bootstrap、三域配置、Slurm smoke 及测试均已实现。下一关不是训练模型，而是
先在真实 Qwen 小样本上确认第二次观察同时存在足够 beneficial 与 harmful support；若没有，
该路线按协议在 critic 前停止。

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

数据层随后冻结了 source 与 decoded-RGB 双重隔离的 ChartQA、DocVQA、HRBench
train/validation/test，共 22,027 个 states。分配只使用数据身份和图像内容，不读取模型
outcome；9,651 个实际图片路径全部存在，HRBench 151 个唯一图片又独立全量复核。冻结
规模分别为 ChartQA `3600/900/1000`、DocVQA `10861/2719/2147`、HRBench
`480/160/160`。约 1.2GB 的本地数据保持 git ignored，只提交分配报告和代码。

真实 Qwen2.5-VL-3B 执行从一次可复现的接口失败推进到三域通过。Job `206627` 在已经写出
baseline + 四 crop 后，因 Transformers 对 system content 的结构化输入要求而 fail-closed；
修复前后 prompt 文本哈希完全一致。Job `206628` 随后通过同一单行；Job `206629` 又在
67 秒内完成 ChartQA 32 states。最后，DocVQA Job `206630` 和 HRBench Job `206631`
分别在 39 秒和 84 秒内完成各 8 states。三域全部保持每 state 五条 sibling rollout、
固定工具四次成本以及 `3/22/6147/6147` 的 L0--L3 维度，特征均为有限值。

因此项目现在跨过了“协议是否可执行、数据能否无泄漏冻结、真实三域特征能否导出”三个
工程风险，但还没有跨过科学风险。异构强基线随后也已在 commit `daa43c1` 操作化：六个
基线的阈值、固定 crop 与 strongest 只由 validation 冻结，one-call 与 four-call 策略保留
独立 outcome/cost/call ledger，再做 source-paired bootstrap。完整六基线的 synthetic
36-cell × 3-seed runner 和全仓 656 tests 均通过。

随后 commit `19631c8` 完成了最后两个 pre-formal 代码 gate。唯一 post-action probe 被固定
为 direct-gain 两层 MLP，它只读取固定工具执行后的四分支 confidence trace、选中 crop
geometry 和原图加该 crop 的冻结 Qwen prompt state；它与 deployable input view 类型隔离，
只作 privileged diagnostic。feature format v2 支持 deterministic shard、resume、逐 shard
provenance/label/coverage 校验和原子 merge。含 probe 的完整 synthetic matrix 与 662 tests
通过，真实 ChartQA 单行 Job `206664` 又确认 `6167` 维 post-action vector 可实际导出。

随后又关闭了一个会破坏一次性 test 可信度的接口缺口。原 runner 在一个函数里同时持有
train/validation/test，即使选择逻辑正确，也无法留下“test 在 freeze 前不可达”的强证据。
commit `e90299a` 把它拆成 development-only fit/freeze 与 held-out-only evaluate 两个类型化
阶段；freeze 保存全部 estimator、calibrator、threshold、baseline、validation inventory
和 development source/RGB identity digest，并要求哈希匹配后才能加载。formal 一键调用被
禁止。进一步自审发现，如果 test rollout/features 已经生成，再由 evaluator 写 ledger，
时间顺序仍不可信。因此当前实现改为 ledger-first 的单体事务：Phase A freeze 后先建立不
读取 test 的 hash-bound plan，唯一 H800 job 在所有非-test preflight 后首先写不可覆盖
ledger，再读取 allocation/test manifest、生成三域 test、检查 development-vs-test
source/RGB 零重叠并执行冻结推断；ledger 后发生任何中断都禁止自动重提。

同一轮还纠正了一个概念泄漏：早期 calibrator 把 `gain-lambda*cost>0` 当概率标签，与本阶段
“只预测 cost-independent quantity”冲突。现在模型与 calibrator 只学习
`gain>0/rescue/harm`，lambda 仅在 validation threshold 和 policy utility 使用。每 seed 的
deployable/L3 cell 选择与三-seed 多数投票都在 validation 冻结，post-action probe 也使用
相同 seed aggregation；终局 report 明确保存 oracle、完整 ladder、rescue/harm、三条 curve、
paired bootstrap、GO/PIVOT/REPRESENTATION/STOP 证据和唯一下一阶段建议。

64-state 跨域吞吐 gate 随后全部完成。ChartQA/DocVQA/HRBench Job
`206665/206666/206668` 的 elapsed 为 `119/189/623` 秒，三者各生成 64 states、320
rollouts 与 64 v2 features，并各自通过 16/16 独立检查。由此冻结的完整 development
点估计/1.5 倍保守预算为 `15.19/22.79 H800-hours`，最终文件约 `1.63GB`；六个 role
顺序执行，每 256 states 保存到同一个可恢复 rollout/feature 文件，持久 resume 文件共
12 个且 learned-backbone checkpoint 为 0。

一次性 test 的额外冻结预算为 ChartQA/DocVQA/HRBench `1000/2147/160` states，按同一
吞吐估计合计 `2.71 H800-hours`，1.5 倍保守为 `4.07`；三个 test roles 共 28 次原子保存、
6 个滚动 resume 文件。test 尚未打开；这些文件只会在完整 development freeze 存在后由
ledger-first transaction 生成。

这一阶段的协议、两阶段 artifact API、ledger-first test transaction、冻结选择规则、
终局 renderer、Slurm 执行脚本及回归测试已统一绑定到 implementation commit
`cf70e2245bfbd629a156978861a8e96bf6fd5384`。发布快照通过 17 个相关 Python 文件的 mypy、
Shell/JSON/hash 检查、关键定向测试与全仓 676 tests；该提交只包含代码、配置、测试和文字
记录，不包含本地数据集、模型权重、checkpoint 或大型实验 artifact。

正式矩阵仍为 `0/36`，test 未打开。下一步生成完整 train/validation outcomes；最终
`PREDICTABILITY_AUDIT.md` 仍只能在完整真实矩阵后给出
`GO/PIVOT/REPRESENTATION/STOP`。

## 第六阶段终局：完整矩阵与一次性 test

2026-09-04 至 2026-09-05，六个 formal development roles 按冻结顺序完成：ChartQA
train/validation `3600/900` states，DocVQA `10861/2719` states，HRBench `480/160`
states。全部 role 都绑定 clean revision `2151b82e44bee0bcd48c30aebc7bc02e1da418a7`
和协议 SHA-256
`699073b149c957022b203e71dc0ae9e7c7733515efb125f26a86713021a3c6e1`。随后冻结 36 个
科学 cells、108 个 seed-specific fits；独立 audit 确认三个 development split、六个强基线、
deployable/L3 selection 与 post-action probes 完整，并确认 freeze 时没有 test data。

唯一 test transaction Job `208184` 在 ledger 落盘后顺序完成 ChartQA/DocVQA/HRBench
`1000/2147/160` 个 held-out states、冻结推断和机器报告。最终 report 为
`formal_claim_eligible=true`、`frozen_before_test=true`、matrix `36/36`，三域 split audit
全部通过。Job 的非零 exit 只发生在最后一步：四个预注册 verdict 都未命中，renderer 按
fail-closed 合同拒绝制造一个类别；此前所有 test artifacts 和 evaluator 均已完成。

科学上，fixed tool 在三域都有显著 oracle headroom，但 primary deployable policy 的相对
改进 lower CI 在三域均不大于零，post-action diagnostic 也没有正 lower CI。用户明确选择
保留这一冻结组合为最终 **INCONCLUSIVE** 结果。它关闭了当前 static pre-action router
主张，但不证明新的 sequential evidence-acquisition 机制必然失败。由于 test 已消费，任何
后续路线都必须重新预注册并使用新的 held-out test。
