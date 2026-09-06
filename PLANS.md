# 研究计划

## 9月13日最终方法冲刺：Factorized Potential Outcomes（2026-09-06 19:35 HKT）

用户给出的硬约束是：从当前全部结果出发，最迟在 **2026-09-13 23:59 HKT** 前形成最终
方法，目标为 CVPR/ICCV/ECCV。
本周不再并行发散多个 idea。唯一候选冻结为 **Factorized Potential-Outcome Visual
Acquisition**：端到端预测当前答案错误风险、错误条件下的 crop rescue 概率和当前正确
条件下的 crop harm 概率，并用
`P(error)P(rescue|error)-P(correct)P(harm|correct)` 排序视觉 acquisition。

对二值 correctness，该式就是概率分解；对 DocVQA 连续 ANLS，使用严格推广：
`e*=1-Y0`、`r*=max(Y1-Y0,0)/(1-Y0)`、`h*=max(Y0-Y1,0)/Y0`，并以
`1-Y0`/`Y0` 加权两个条件 loss。因而
`Y1-Y0=(1-Y0)r*-Y0h*` 对所有 `[0,1]` reward 精确成立。

该方向来自已有结果中的一个明确不对称：Job `209090` 的 Outcome-only 在两个域都优于
直接 Counterfactual Utility，ChartQA 的 top-25% 实际选中 8/16 个 rescue 且 0 个 harm；
它主要学到了“当前答案可能错误”，但尚未把 baseline-wrong 中的 rescue 与仍然失败区分
开。直接 gain 则只从 512 个 train pairs 中的 63 个非中性 pair 得到梯度并发生全
CONTINUE collapse。新方法使用每个 pair 训练 risk，并使用总权重为一的 rescue/harm
条件监督；不是继续给旧 gain loss 调 class weight。

协议已写入 `docs/factorized_potential_outcome_protocol_v1.md`。Job `209134` 证明三头训练
链路和无泄漏 gate 可运行，但随后在首个 Phase-B 日志、产生 validation 结果之前发现旧
loss 错把 DocVQA 连续 ANLS 按 `0.5` 二分。Job `209157` 已于 17:08:37 HKT 主动取消；其
部分训练不作结果。连续 reward 精确分解已实现并由 21 个定向测试覆盖。修正后的 Phase A
Job `209158` 已在 3×RTX 4090 上 `COMPLETED/0:0`，三臂全部工程 gate 为真，machine
decision 为 `PHASE_A_PASS`。修正 Phase B Job `209159` 随后完成并达到冻结晋级规则：
ChartQA 25% call rate accuracy `53.125%`，相对 strongest uncertainty `+3.906pp`、
Outcome-only `+0.781pp`；DocVQA `91.527%`，相对 strongest uncertainty `-0.163pp`、
Outcome-only `-0.746pp`。两域相对 Outcome-only 的平均差仅 `+0.017pp`，所以这是很脆弱
但合规的 **development GO**，不是论文结论。

现在进入 Phase C：冻结更大 paired train banks、全新 held-out allocation、3 seeds 和
ChartQA/DocVQA/HRBench。正式方法只有在至少两个域方向为正、至少一个域相对 Outcome-only
的 source-bootstrap 95% CI 下界大于 0，并通过 image/question/region 语义消融时才成立。
Phase B 的晋级规则原为
同时达到“一个域相对 strongest uncertainty `>+1pp`、另一域 `>-0.5pp`、两域相对
Outcome-only 的平均差为正”；它已经满足，但不能替代上述 Phase C 条件。

一周节奏：Day 1 完成实现、smoke 与 bounded pilot；Day 2--3（仅 GO）扩 paired banks 和
冻结新 held-out；Day 4--5 做三 seed/三域主结果及强基线；Day 6 做语义消融、frontier、
bootstrap 与 error analysis；Day 7 冻结方法、表图和论文 method/experiment 骨架。这个
计划承诺的是在一周内给出有证据约束的最终方法判定，不承诺顶会录用或正结果。

Phase-C 数据冻结采用 outcome-blind 规则：ChartQA 从固定 raw-train revision 排除所有历史
manifest 后抽 512 个平衡 held-out states；DocVQA 从固定 official-validation revision 排除
所有历史 source/RGB 后抽 128 个完整 document groups；两者都是真正新 source。训练侧分别
从旧 development-train 冻结 1024 ChartQA states 和 256 DocVQA document groups。HRBench
4K/8K 是同一 800 问题的两种分辨率，不能把 4K 当新独立样本；因此从原 train role 中仅
选取没有任何历史 sequential outcome 且与训练 image-disjoint 的 20 个 8K image groups
作为 held-out，并在论文中明确它不具备前两域同等级的全新 source 保证。Job `209165` 已
成功冻结 allocation：ChartQA 为 `1024 train / 512 held-out states`，DocVQA 为
`1012 train states from 256 documents / 522 held-out states from 128 documents`，HRBench
为 `388 train / 92 held-out states`（69/20 image groups）；三域 train/held-out 的
state/source/image overlap 均为零，选择未使用模型 outcome，held-out sequential outcomes
仍未打开。现在只生成三域 train paired banks；每域一个 4×RTX 4090 job 做确定性 state
sharding，再用 manifest/code/completion/record hashes 和 exact decision coverage 合并。

三个 train-bank jobs 已全部成功：ChartQA Job `209177`（3分45秒，rollout SHA
`e40f832f...0bbd`）、DocVQA Job `209178`（5分29秒，`b657a0b9...0a37`）、HRBench Job
`209179`（7分01秒，`2726a514...972f`）。对应 beneficial/harmful/neutral 为
`84/20/920`、`34/23/955`、`28/21/339`。下一步已冻结为三域联合训练：每个 seed 做
3072 steps（每域恰好 1024 draws），seeds `17/29/47`；Outcome-only、direct gain 和
Factorized 三臂使用完全相同的 outcome-independent schedule 并占三张 GPU 并行。旧
validation 只作为 previously-seen monitor，不能选方法、seed 或阈值；正式 held-out
不出现在训练 matrix 中。

截至 2026-09-06 19:35 HKT，seed `17/29/47` 分别绑定 Job
`209187/209188/209189`。Job `209187` 正在 3×RTX 4090 上运行，日志已推进到约
`2011/3072` steps；其余两个 job 因 `AssocGrpGRES` 等待同一账户 GPU 额度释放。这是运行
健康度快照，不是模型选择或效果结果。正式 held-out 仍未打开。

正式 Phase-C one-shot 评测实现已冻结在 commit `5e2f77d`：包含 ledger-first transaction、
四分片真实 rollout、三 seed 独立 selector scoring、Answer-only/random/uncertainty/
Outcome/direct-CF/Factorized/oracle 基线、20,000 次 source-cluster bootstrap、语义消融、
accuracy-cost frontier 和 `GO_NO_GO.md` renderer。执行前还必须完成一个只读已开放
validation 的真实 runtime smoke，并把 smoke report、三个 selector checkpoint、配置、
代码和 held-out manifest hashes 一并绑定到最终 plan；不得在三个训练 job 结束前运行该
smoke，也不得在 plan 冻结前读取 held-out bytes。

## Counterfactual Visual Utility Post-training 已关闭：Phase B NO-GO（2026-09-06）

新的独立路线已预注册在 `docs/cv_counterfactual_method_protocol_v1.md`。它不重写此前
Sequential frozen-critic 的 NO-GO，而是直接对 Qwen2.5-VL-3B 做轻量 post-training，比较
绝对 final-reward 的 Outcome-only 对照和显式 `G=R_continue-R_stop` 的 Counterfactual
Utility preference。两臂复用完全相同的 ChartQA/DocVQA paired partial-prefix bank、架构、
优化器、state schedule、步数和 seed；proposed crop 在决策前不会执行。

Phase A Job `209085` 的工程 smoke 全部通过。随后 Phase B Job `209090` 按冻结设置完成：
2×RTX 4090、每域完整 256 train / 128 validation、512 steps、seed 17、精确 25% matched
call rate。Counterfactual 相对 Outcome-only 在 ChartQA/DocVQA 分别为 `-3.906pp/-0.521pp`，
两个域均不为正；相对最强 uncertainty baseline 为 `-0.781pp/+0.062pp`，也未满足
`>+1pp` 与 `>-1pp` 的 Phase C 转移规则。

因此本路线按预注册停止条件正式 **NO-GO**。不运行三 seed Phase C，不创建新 test
transaction，不做 class reweighting/loss 搜索，不进入 7B、RL 或 multi-turn。终局证据见
`CV_METHOD_GO_NO_GO.md`。

## Sequential Visual Acquisition 已关闭：NO-GO（2026-09-06）

当前目标已完成到预注册的否证终点。三域 counterfactual headroom 均存在，但 18,461 维
初版 critic 和唯一允许的 90 维 relational correction 都没有在至少两个域以正的 paired
CI 下界超过 matched entropy/confidence/margin；跨域迁移也失败。按 STOP-2/3/4/5，禁止
继续换表示、加 seed、调 threshold、扩 backbone、打开 sequential test 或进入 RL。

若未来重新立项，必须是新的 estimand 与新的 held-out allocation，不能复用本路线已经看过
的 validation 或 pre-existing test identities。当前没有后续训练/排队任务；终局证据见
`SEQUENTIAL_GO_NO_GO.md`，历史 PENDING 计划保留在下方。

## 当前用户授权目标：Sequential Visual Acquisition（2026-09-06）

Utility-SFT 已按原停止规则以 NO-GO 关闭；新的唯一目标是验证 partial visual evidence 后的
`STOP` 与“一次固定 additional acquisition”。本阶段冻结 Qwen2.5-VL-3B backbone，只允许
一个已有 observation、一个 outcome-blind fixed next crop、binary stopping、linear/小 MLP
risk/gain critic；不学习 where，不做 exhaustive candidates、RL、7B 或两步以上 agent。

协议已预注册在 `docs/sequential_acquisition_protocol_v1.md`。旧 sibling bank 只有“原图 vs
原图+单 crop”，不能伪装成共享 partial-prefix 数据，因此必须新生成原图+已有 crop 的 STOP
和原图+已有 crop+固定新 crop 的 CONTINUE paired branches。现已实现 typed schema、真实
rollout、严格 pre-action feature allowlist、Critic A/B、policy/metrics、10,000 次 source
bootstrap、三域配置和 test authorization 骨架；相关定向测试 `20 passed, 2 skipped`。

三域 train `32` / validation `16` 的真实 smoke 已全部完成，证明第二次观察存在稀疏的
beneficial 和 harmful support；ChartQA tiny critic/evaluator 也已端到端通过，但没有优于
entropy 的证据。下一步按协议扩大到 train `256` / validation `128` 的有界 diagnostic，
分别训练 linear/MLP critic；只有至少两个域显示可学习且 matched-rate CI 方向合理，才生成
完整 development bank。test 继续封存；smoke/pilot 均不能作为 GO。

更新时间：2026-09-06 11:43（Asia/Hong_Kong）

## 当前用户授权目标：Counterfactual Utility SFT（2026-09-05）

用户已明确指定新的 supervised-only 假设：对 Qwen2.5-VL-3B 做 spatial utility
post-training，比较 Format/Support、Best-Action 与 soft Utility-SFT。本轮不做 RL、
7B、continuous bbox 或 multi-turn。下方旧路线及“下一步”保留为历史，不再作为当前
执行指令；旧 INCONCLUSIVE 结论和已消费 test 保持不变。

执行合同与 Phase 0 字段/复用审计：`docs/utility_sft_phase0.md`。完整目标仍包括
三套训练、八基线、至少两个域的独立证据、空间语义消融、一次性新 test freeze、两张图
和四问 `GO_NO_GO.md`；不能以模块单测或单域 overfit 代替完成。当前先实现数据隔离、
离散动作和可反传的原图 ROI head，再进行有界真实输入 sanity check。

真实 TRAIN overfit gate、三个 matched development arms、冻结 VOI、八策略 validation
evaluation、两张图与三类语义消融均已完成。首轮 pilot 未通过 Go 后，预注册唯一允许的
coverage correction Job `208822` 也已完成：三臂统一 1024 steps、全 source pool、
outcome-independent source-cycle，其余训练和评估设定不变。

修正后 ChartQA/DocVQA 的 Utility-SFT 在 primary policy 上均退化为全选 ANSWER，与
Best-Action 持平；HRBench 相对 Best/Frozen 虽为正方向，但两个 paired 95% CI 都跨零。
语义消融也没有显示跨域一致的 image-question-region 依赖。因此 Stop 1/2/3 已触发，
`GO_NO_GO.md` 判定为 **NO-GO**。本路线到此结束：不打开新 test，不进入 RL、7B、额外
loss/threshold/seed 搜索。详见 E-20260906-31/32。

更新时间：2026-09-06 03:53（Asia/Hong_Kong）

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

## 已完成路线：pre-action predictability audit

2026-09-05 终局更新：六个 development roles、36 个科学 cells、三个固定 seeds、唯一
post-action probe 和 ledger-first held-out test 已全部完成。机器报告通过 formal、freeze、
split、coverage 与 20,000 次 source bootstrap 合同。结果未命中预注册的
`GO/PIVOT/REPRESENTATION/STOP` 任一完整分支；用户明确接受 fail-closed 结果作为终局，
因此最终报告标注为 **INCONCLUSIVE**。当前 test 已消耗，禁止用它继续选模型、阈值、
feature、seed 或判定规则。

2026-09-03 起停止继续做开放式 N6 problem selection。当前唯一研究问题固定为：在不学习
`where` 的前提下，pre-action VLM state 能否稳定预测一个固定视觉工具的实际效用。决策
只有 `ANSWER_NOW` 与 `USE_VISUAL_TOOL`；固定工具执行全部四个 UG-grid crops，按
post-action entropy 选结果并支付四次成本。deployable predictor 永远不能读取任何
post-action 字段。

机器协议为 `configs/predictability_audit_v1.json`，文字协议为
`docs/predictability_audit_protocol_v1.md`。正式实验严格为 ChartQA、DocVQA、HRBench ×
L0/L1/L2/L3 × direct-gain/rescue-harm/factorized，共 36 cells，最多三个固定 seeds。
train/validation/test 必须按 source 与解码 RGB hash 双重隔离；test 不参与选择。终局文件
必须是 `PREDICTABILITY_AUDIT.md`，且只能输出 `GO/PIVOT/REPRESENTATION/STOP` 之一。

第一轮 dependency-free 合同与 retrospective asset audit 已完成。现有 opened ChartQA、
DocVQA、V*Bench bank 只验证固定工具标签构造与 headroom；由于尚无新 max probability、
margin、完整 L3 state 与真实 paired outcomes，正式矩阵完成度仍为 `0/36`。

image/source-disjoint 数据冻结现已完成：ChartQA 为 `3600/900/1000` states，DocVQA 为
`10861/2719/2147` states、对应 `2800/700/500` documents，HRBench 为
`480/160/160` states。三者 train/validation/test 的 source 与 decoded-RGB overlap 均为
零；test 在任何新 rollout 前冻结，分配未读取模型 outcome。真实 Qwen rollout + L0--L3
feature path 已在三套数据的 opened train role 通过，test 继续封存。

首个 smoke Job `206627` 证明单行 baseline + 四 crop rollout 可运行，但 L3 extractor
在第二次模型加载后因 Transformers 要求结构化 system text block 而 fail-closed。plain
与 structured 表示经真实 processor 验证生成完全相同的模板文本与 tokenization；修复仅
改变输入容器类型，后续同一个工程 gate 已通过，未改变 prompt、模型、数据、指标或协议。

修复后的 Job `206628` 已完整通过单行 gate；ChartQA 32-state Job `206629` 又验证了
吞吐、显存与 artifact growth。随后 DocVQA 8-state Job `206630` 和 HRBench 8-state Job
`206631` 分别在 39 秒和 84 秒内完成。三者均生成每 state 一条 baseline 和四条 crop
rollout，固定工具恰收四次成本，L0/L1/L2/L3 维度稳定为 `3/22/6147/6147` 且全部有限。
这些 smoke 不用于选择模型、阈值或 endpoint。

32-state ChartQA Job `206629` 在 67 秒完成，折合约 `1719 states/H800-hour`；最终
rollout + feature 约 `61KB/state`。此前按 18,720 个 train/validation states 线性外推的
`10.9 H800-hours` 与 1.5 倍保守预算 `16.4 H800-hours` 只来自 ChartQA，不能直接当成
跨域最终预算。DocVQA/HRBench 的 8-state gross throughput 约为 `738/343 states/hour`，
但样本过小且包含两次模型加载，主要用于结构验证。正式 shard size 和总预算要在分片
实现后按各域较大 opened-train shard 再冻结一次。

当前 runner 已包含三 target 固定训练、source weighting、validation score calibration、
threshold、全套 prediction/policy metrics、call-rate curves 与 paired source bootstrap。
完整 synthetic 36-cell × 3-seed smoke 已通过；这是 108 个 seed-runs 的工程证据，明确标记
`formal_claim_eligible=false`。真实 manifest 与三域 feature path 阻塞已解除。

异构强基线 gate 已在 commit `daa43c148dc1f3a1e2fe5e1603ea1ae464ab7ed6` 完成：六个基线
均只在 validation 冻结，one-crop 与 four-crop 方法保留各自逐样本 `Ytool`、cost 和 call
mask，learned-vs-baseline 的 paired source bootstrap 不再共用错误 ledger。含完整六基线
的 synthetic 36-cell × 3-seed 再次通过，656 个当时的全仓测试也通过。

其余两个代码 gate 已在 commit `19631c853504981ba97617dfab44dc228e8baf4b` 完成。唯一
post-action diagnostic probe 固定为 direct-gain `[128,32]` MLP；输入与 deployable view
分离，只含四 crop 执行后的 confidence trace、entropy-selected crop geometry，以及原图加
选中 crop 的冻结 Qwen prompt state。feature format v2 同时支持 deterministic shard、断点
续跑、逐 shard hash/coverage 验证与 atomic merge。含 probe 的 synthetic 36/36 × 3-seed
矩阵、662 个全仓测试和 merger smoke 均通过。

真实 Job `206664` 又在 opened ChartQA train 单行上于 25 秒内通过：pre-action 维度仍为
`3/22/6147/6147`，post-action probe 为 `6167` 维且全部有限，固定工具四次成本与
entropy-selected action 完全一致。

正式 test 隔离不再只靠约定。commit `e90299a645e528dc937c7c346cd5978e8016a599`
把 matrix 拆为不接受 test 参数的 train/validation fit-freeze API，以及只接受持久化 freeze
和 held-out data 的 evaluate API；formal one-shot wrapper 会 fail closed。冻结 bundle、
development source/RGB identity index、全部 variant/threshold/calibrator/baseline 和 JSON
inventory 均有 SHA-256。后续自审把 transaction 顺序进一步收紧：不允许先生成 test 再由
evaluator 建账；独立 starter 必须先 exclusive-create ledger，同一个不可自动重提的 H800
job 才能读取 allocation/test manifest、顺序生成三域 test artifacts、应用冻结 matrix 并
渲染终局报告。两阶段 artifact loader/CLI 和 test transaction worker 已实现，并已通过
全量回归、绑定到 clean implementation commit `cf70e2245bfbd629a156978861a8e96bf6fd5384`。

冻结前还修复了 lambda 泄漏：probability calibrator 现在只学习成本无关的
`gain>0/rescue/harm`，lambda 仅用于 validation threshold 与 policy utility。每个 seed 在
validation 上唯一选择 primary cell，三个 seed 多数投票；L3 和 post-action probe 同样冻结
聚合。GO 的 Pareto/rescue/harm 与 PIVOT 的“所有 individual cell-seed 和 primary 的 lower
CI 均不为正”已有可执行证据字段，终局 renderer 会在 matrix、one-shot ledger 或 split
证据不完整时拒绝生成 `PREDICTABILITY_AUDIT.md`。

64-state 较大吞吐 gate 已全部完成：ChartQA/DocVQA/HRBench Job
`206665/206666/206668` 分别运行 `119/189/623` 秒，折合
`1936.13/1219.05/369.82 states/H800-hour`；三个 run 均为 64 states、320 sibling
rollouts、64 条 v2 feature，并各自通过 16/16 独立检查。完整 train+validation 的线性点估计
为 `15.19 H800-hours`，1.5 倍冻结保守预算为 `22.79 H800-hours`，最终 rollout+feature
约 `1.63GB`。正式执行固定为六个顺序 role jobs、每 256 states 原子 checkpoint；不训练
VLM、持久 resume 文件共 12 个，最后只保存一个 matrix bundle。正式矩阵仍为 `0/36`，
test 继续封存。test transaction 预注册为 3307 states、`2.71/4.07` raw/保守 H800-hours、
28 次原子保存和 6 个滚动 resume 文件。matrix round-trip 新回归 `7/7` 通过，全仓 676 tests
回归 `ExitCode=0`；当前未提交正式作业，下一步只提交 ChartQA train。

## 当前核心判断

完整 formal audit 的核心结论是：三域 fixed-tool oracle utility 均显著为正，但已测试的
pre-action predictor ladder 没有在任何 benchmark 上取得相对最强基线为正的 95% CI lower
endpoint；post-action diagnostic 也没有正 lower endpoint。因此当前 static-router 主张没有
得到支持，但证据不足以断言另行预注册的 active/sequential acquisition 必然失败。

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
pre-GPU contract 已通过。此后 Job `206205` 已完成真实 rollout、两步普通 outcome
GRPO update 与完整 checkpoint，但 64 条 trajectory 中没有 parser-valid、可执行的
工具动作，因此 H5 特有的 action-local credit 通路从未激活，不能据此获得方法性能证据。

G0 后已做范围纠偏：VTool 只作为 Apache-2.0 的可运行 RL 骨架和 outcome-only
comparator，不再审计 thought、pixel 或内部实现是否与 VTool 等价；该问题与 H5 的
成败无直接关系。训练数据改为固定 revision/hash 的 Apache-2.0 official ReFocus train。
token-local autograd、隔离 runtime import、official-train converter/processor、paired
agent fake-server contract、单卡 H800 vLLM model-load/真实首轮 generation、72 行完整
运行时数据审计、JSON-safe counterfactual rollout export、自动 stop-rule analyzer 与
最终 Hydra resolved-config gate 均已通过。四卡 Job `206179` 暴露并修复 DP batch
chunk 类型合同；Job `206184` 暴露并修复 checkpoint 空间 gate；最终 Job `206205`
完成两步运行。其正式结果为 tool call `0/64`、rate `0.0`，冻结分析器给出
`paired_signed_g1_stop_rule_triggered`。这关闭当前 sampled on-policy H5 路线，不进入
matched controls 或 G2。

独立 typed-action V2 的 CPU/真实 executor gate 随后通过，但唯一 H800 generation
smoke（Job `206227`）得到 11/16 tool intent、7/16 完整且 syntax-valid Python fence，
参数合同、strict parser 与 execution 均为 0/16。11/11 有意图输出都逐字复制 prompt
里的无效元变量函数名 `focus_on_x_values_with_MODE`，机械决定为
`typed_action_b0_malformed_tool_intent`。因此 V2 baseline 自身关闭，不改 prompt/seed 后
重跑；这个结果不改变 G1 stop，也不能充当 N0 方法证据。

N0 action-boundary interventional objective 的形式化与 dependency-free 零支持 gate 也
已完成。完整干预期望效用的精确梯度仍乘当前宏动作概率，在零/近零 support 下消失；
改用 utility-induced target 可恢复梯度，但就成为 LIRE/LiPO/AWR 类 listwise/off-policy
投影；单独回归 `Q(s,a)` 则回到 GapSight/既有 action-value router 家族。数值 gate
10/10 checks 全真，一手文献还确认 ToolVision 已用 evidence gain/paired MUT 建立 support，
The Illusion 已定义 fixed-prefix observation intervention。N0 因
`action_boundary_candidate_reduces_to_existing_objective_families` 在 GPU 前关闭。

N1 现有 sibling-bank 盘点也已完成。四个主开发 bank 跨 InfographicVQA、ScreenQA、
DocVQA、TextVQA，共 `59,949` decisions、`299,745` rows；完整 sibling、source IDs 与
不可变 provenance 均通过。因此 stop regret 和已注册四个 UG-grid 候选内的
action-selection regret 可识别。但 `239,796` 条主 ZOOM 记录中没有一条保存 fixed action
prefix、匹配 factual/counterfactual observation 与受控 continuation，evidence-use regret
不可识别。主证据还只有一个 ZOOM/UG-grid 动作族、每状态一个 replicate，且 3B/7B 与
数据集混杂；唯一同数据集多 backbone 只有 ScreenQA 的 512-state opened-development
diagnostic。N1 以 `n1_existing_assets_insufficient_for_top_tier_regret_benchmark` 关闭，
不把规模当成完整因果 benchmark。

N2 随后验证了“严格可加三/四段 causal regret”能否补救。Stop regret 与 action-selection
regret 可以严格分成两个非负项；固定 action 时 prefix effect 与 real-vs-counterfactual
visual-evidence effect 也严格可加，但它们是可正可负的 causal effects，不是 regret。
真正的 evidence-use regret 需要未观测的 ideal continuation；相同观测可对应 `0` 或 `0.4`
的不同 regret，而 best-of-k ceiling 又从 k=1 的 `0.6` 随 k=8 机械增到 `0.9993`。
The Illusion 已直接覆盖 action-shortcut/observation-mediated 分解，GapSight 已覆盖 stop/action
utility。N2 因 `n2_additive_causal_regret_candidate_not_identified_and_not_novel` 在 GPU 前关闭。

N3 已完成公开 checkpoint 与独立新颖性联合 gate。VTool 3B/7B 权重公开、ungated、MIT，
且均有可固定 full revision；若科学上获授权，8,143,089,840-byte 的 3B 是唯一优先候选。
但是当前 model card/代码没有把精确 checkpoint、prompt/parser contract 与 parser-valid
execution trace 绑定，baseline gate 只通过 4/7。独立新颖性又因 TACO 已覆盖 signed
tool value 与 token responsibility routing、TAPO 覆盖 action-level counterfactual credit、
The Illusion 覆盖 fixed-prefix observation intervention、ToolVision 覆盖 with/without-tool
benefit supervision 而为 0/6。最终决定
`n3_public_initializer_exists_but_joint_gate_failed_before_download`；没有下载、GPU 或新
checkpoint，当前 H5 不能靠换初始化重开。

N4 的零成本 problem-selection gate 已完成。最初的 prospective crop-completion/VOI 机制
因 VOILA、Learning to Look Around、AdaptVision 等直接邻近工作而关闭。替代候选把
selector 动作前可见的信息集作为评测合同：同可见性、同 action bank、同净效用定义内做
方法比较，并测试跨信息集的排名反转。形式化与机器实现 14/14 checks 通过，但其中
aliasing-regret decomposition 已被 Self-Certification 直接覆盖，完整成本账本也被 VQABench
部分覆盖。N4 当前只以 selector-input ledger、matched-visibility comparison 和
cross-information-set rank reversal 三项联合协议进入 N5；这是待证伪的 benchmark/evaluation
候选，不是方法或主结果。

N5 已在读取逐 decision 配对结果前冻结回顾性否证协议，并在同一 DocVQA bank、同一
5% 调用预算与 source-balanced 20,000 次配对 bootstrap 下完成。较低信息
`context-geometry` utility 为 `-0.00274318`，较高信息 `semantic-context` 为
`-0.00296742`，higher-minus-lower 为 `-0.00022424`，97.5% paired CI
`[-0.00528886, 0.00430109]`。8 项科学条件全部失败；ScreenQA OOF 增量也只有
`0.00004824`，约比 `0.001` 门槛小 20.7 倍。N4/N5 当前候选因此关闭，不打开
ScreenQA risk-calibration，不提交 GPU。Privileged oracle 仍为 `+0.03117658`，说明
action headroom 存在，但当前 outcome-free router 不能跨 source 稳健利用。

2026-09-03 00:56 HKT，重提的 paired-signed Job `205902` 获得资源后在 worker
前置检查中同秒退出。原因是 shell 中的 jq 对象全真断言误写为
`.checks | all(.[] == true)`，当前 jq 会把单个布尔值再次送入 `.[]` 并以 exit 5
失败。没有创建训练输出目录，也没有 rollout、模型加载、optimizer step 或
checkpoint；因此这不是 H5 的正/负结果。两处同类断言已改为二参数形式
`.checks | all(.[]; . == true)`，并加入真实 jq 正/负回归。下一次重提前必须完成
全量回归、重新生成最终 Hydra gate 并以新 commit 绑定 worker 哈希。全量回归现已
通过；修复 commit `8c0f6c010a4dfeb1bf01d955054da2287691896e` 上的 Hydra v13
也以 59/59 checks 全真通过，launch manifest/resolved config SHA-256 分别为
`29e24dab...8fe8` / `cda71307...ac43`。当时计划是在最终 HEAD 再跑一次同一 gate，
然后实时复核资源并只重提 signed arm；该步骤后来已由 Job `206205` 完成。

2026-09-03 12:31 HKT，paired-signed Job `206205` 在 clean revision `9c6bdc4` 上
正常完成。两步 task score 为 `0.5625` / `0.53125`，总体为 `0.546875`；正式
parser-valid tool call `0/64`。两步 action-credit tool trajectory count 与 applied
credit 均为 0；结构审计 10/10 全真，pair mismatch/judge failure 为 0，唯一
`global_step_2` checkpoint 完整且全部文件已做 SHA-256 绑定。由于 `<1%` 是结果前冻结
的停止规则，当前路线必须停止，不能通过改 prompt、seed、temperature 或阈值追结果。

2026-09-03 13:11 HKT 的只读 raw-response 诊断纠正了“全部 direct”的过强描述：
48/64 以 `FINAL ANSWER:` 开头，16/64 以 `focus_on_*` 开头；后者 step 1/2 分别为
12/32 与 4/32。15 条可解析为单个 focus expression，但 0 条符合真实三参数 API，另
1 条混入最终答案而语法无效，所以只补 Python 围栏可恢复的调用仍为 0。正式 G1 stop
不变；新信息是 baseline 同时存在 prompt/API 格式支持缺口，不能再等同“无 latent tool
intent”。机器报告与路线审计见 `vtool-g1-intent-format-posthoc-job-206205-v1.json`
和 `vtool-g1-format-contract-and-next-route-audit-20260903-v1.md`。

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

1. Exact typed-action V2 baseline 的 CPU/runtime gate 与唯一 H800 generation smoke 已
   完成。V2 因 11/11 tool intents 复制无效 `MODE` 元变量、0/16 参数合法而关闭；不在
   已打开 row/seed 上改 prompt 重跑。若做 V3，只能以全部 concrete、parser-valid 的六个
   函数模板、独立 structural group 和新种子预注册为一次新的 baseline correction；
2. N0 action-boundary interventional objective 已因零支持/目标族 gate 关闭：不提交 GPU，
   不把 token-local mask 或 same-prefix effect 包装成新方法；
3. N1 现有资产路线已关闭：前两项 regret 可识别，但 evidence-use、同数据集多 backbone、
   多动作族与随机重复四项 gate 失败；
4. N2 已关闭：两类 effects 虽可加但不是非负 regret，ideal continuation 不可识别，且
   causal 路径与一手文献直接碰撞；不做 augmentation 成本估计或数据生成；
5. N3 已关闭：公开 tool-capable initializer 存在，但 artifact-level prompt/parser/support
   provenance 不完整，且当前 H5 的核心训练主张已被 TACO/TAPO/The Illusion/ToolVision
   覆盖；不下载模型、不提交 GPU；
6. N4 formal gate 已通过：14/14 机器检查验证信息边界、toy rank reversal 与完整成本排序，
   但不把已碰撞的 aliasing 理论作为贡献，也不把 toy 结果当现实证据；
7. N5 已完成并关闭 N4：同预算 source-balanced 高信息 router 相对低信息 router 的差为
   `-0.00022424`，paired 97.5% CI 跨零；8/8 科学条件失败。ScreenQA calibration、
   formal-test 与 reserve 未打开，GPU/checkpoint 均为 0；
8. 下一轮重新回到零成本 problem selection。新候选必须解释“privileged oracle 很高、
   deployable router 跨 source 失败”的残差结构，并先通过一手文献碰撞、可识别性和已有
   数据上的最小可证伪 gate。不得把 source weighting、fixed-crop 偶然正点、更多特征、
   阈值或模型容量变化包装为新贡献。

## 止损规则

- 不再运行 attention layer/head/ratio、entropy threshold、call-rate、随机种子或
  线性 classifier-family sweep。
- 不用 validation/test 帮助选路线，不把 privileged oracle 当部署结果。
- 下一候选必须先写 protocol，再实现，再 smoke；没有能区分科学假设的新信息时
  不提交 GPU job。
- 不再对 answer hidden-state/grounding probe、generic group-DRO/IRM 或 conformal
  threshold 进行局部变体搜索。
- 当前 sampled on-policy action-credit 路线已因 Job `206205` 的零 parser-valid 工具调用
  正式关闭；
  不运行无法区分 credit 效果的 zero/shuffled/outcome-only controls。
- 新候选若在文献审计或最小 gate 失败，继续选择实质方法/benchmark contribution；
  不以降低投稿目标作为完成条件。

## 下一阶段计划（覆盖下方历史执行清单）

1. 将本次 formal test 视为已消费、只读证据，不重跑、不事后调参。
2. 停止当前 fixed-tool static gate 路线；`INCONCLUSIVE` 是终局记录，不改写为四个注册
   verdict 中的任意一个。
3. 如继续研究，先提出 active/sequential evidence acquisition 的新 estimand、机制和强基线，
   完成一手文献碰撞与可识别性审计。
4. 新路线必须使用新协议和新 held-out test；先做 CPU/小规模真实输入 gate，再按信息价值、
   排队时间、运行时间和 GPU-hours 决定是否提交多 GPU 作业。
5. 在新协议冻结前不启动正式 GPU 实验。

## 历史执行清单（已由上节覆盖）

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
6. 单卡 H800 model-load/单条 generation smoke（Job `205784`）已通过；72 行 frozen
   train 已全部经过真实 `RLHFDataset` 与 Qwen processor，prompt 最大 1,914 tokens，
   row/data/provenance 全部匹配。最终 Hydra dry-run v9 的 59 项 scientific/resource
   contract 全部通过。
7. 首次提交的 Job `205870` 在排队期间发现 rollout JSONL 只会保存 `acc`、无法导出
   counterfactual utility/harmful-call 证据，因此在启动前取消；`RunTime=00:00:00`，
   没有消耗 GPU。现已加入稳定 audit JSON、pair/score/utility analyzer 与 worker
   自动 gate，科学配置未改变。
8. 重提的 Job `205902` 在 2026-09-03 00:56 HKT 获得资源，但因 worker 的 jq
   object-values 断言语法错误同秒 fail closed；没有创建输出目录或产生科学结果。
   根因已由相同 jq/相同 72 行 audit report 稳定复现，并已修正两处同类断言。
9. jq 修复后的 Job `206170` 已真正进入四个 FSDP actor 初始化，但 pinned runtime
   默认强制 `flash_attention_2`，隔离环境没有 `flash_attn`，四个 rank 在权重加载前
   以相同 ImportError 退出。没有 rollout、optimizer step、checkpoint 或 H5 指标。
   进一步源码诊断发现，只覆盖为 SDPA 仍不充分：`use_remove_padding=true` 会把 Qwen
   attention forward 再替换回自定义 FlashAttention 路径。因此所有实验臂共同冻结为
   `attn_implementation=sdpa` 且 `use_remove_padding=false`；这属于首次科学结果前的
   对称运行时修复，不改变方法或比较定义。
10. commit `22b89a5ca872356c621203f7bb724042c846a091` 已加入 fail-closed 配置审计和
   单 H800 真实图片 actor-load/forward smoke。Job `206174` 随后以 `COMPLETED`、
   `ExitCode=0:0`、50 秒、零 restart 通过：完整 Qwen 权重与 verl 多模态 patch 在
   SDPA/no-remove-padding 下对 966-token 真实图片输入完成前向，6/6 checks 全真，
   峰值已分配显存约 7.42 GiB。报告已由 commit
   `0122689050a4bfc91df0a55dd154dc52d7fce83d` 绑定进 G1 launcher。下一步在最终
   clean revision 复跑 gate，再重新提交唯一 4×H800、最多 2 optimizer-step 的
   paired-signed G1。只有实际
   tool-call rate、pair validity 与训练稳定性通过冻结 stop rules，才以同一 revision
   顺序运行 zero/shuffled/outcome-only controls；失败则按预注册规则停止，不调整
   prompt、seed、temperature 或阈值追结果。
11. 四卡 Job `206179` 已越过此前 backend 阻塞，到达首个 `_update_actor`，随后因
   action-credit adapter 把 donor trajectory IDs 作为 Python list 注入而在 verl
   `DataProto.chunk()` 失败。该 runtime 要求全部 `non_tensor_batch` 值为 ndarray；
   同一 runtime 的最小复现已从断言失败变为 4-way chunk 7/7 checks 全真。commit
    `ea489f7c520880ab087af761a620b03f357b18e0` 修复类型并把 exact CPU smoke 加入
    submitter/worker。科学配置和 stop rules 未变；最终 clean Hydra gate 后允许一次重提。
12. 重提的 Job `206184` 已完成 32 行 step-1 rollout 并进入最终 checkpoint 保存，
    但原 32 GiB 空间 gate 低于实测至少 40.39 GiB 的 checkpoint shards，持久盘写满后
    无法写出 step-2 rollout、完整 checkpoint 或 analyzer。step-1 结构审计 10/10
    checks 全真、task score `0.53125`，但 parser-valid 工具调用为 0；事后 raw 文本为
    19 条 final answer 与 13 条裸 focus intent。这只提高正式 stop 风险，不能
    替代两步判定。不可恢复 checkpoint 与可重建 Arrow cache 已按用户授权清理，当前
    空间约 77.1 GiB；资源合同提高为 submitter/worker 双重 64 GiB fail-closed gate。
    当时计划是验证并 clean commit 后只重提 signed arm；该重提已完成。
13. Job `206205` 已完成两步、64 行 rollout、正式 analyzer 与约 42 GiB 的唯一
    `global_step_2` checkpoint。两步 score 为 `0.5625` / `0.53125`，但工具调用为
    `0/64`，正式 decision 为 `paired_signed_g1_stop_rule_triggered`。checkpoint 和核心
    小产物均已做内容哈希审计。
14. 按冻结 G1 规则关闭当前 on-policy H5：不运行三组 controls，不改 seed/prompt/
    temperature/threshold。下一步先完成零支持条件下新算法的一手文献碰撞与可行性审计，
    审计前不提交 GPU。
15. 事后格式诊断已证明 Job `206205` 有 16/64 裸工具意图，但 0/16 可仅补围栏执行；
    进一步发现 V1 prompt 声称可用的 `x_values_bbox/y_values_bbox` 不在实际 execution
    context；这不改变 G1 stop，却要求新路线先建立 typed-action V2 reliable baseline。
16. Prompt/SFT/forced-call、representation steering、logit bias、ordinary listwise reward
    和 crop loss-gap router 均已有直接文献覆盖，只能作 baseline。唯一暂存主方法候选是
    action-boundary interventional objective；若形式化后退化为上述目标，则在 CPU gate
    关闭。
17. Typed-action B0 完整 CPU/真实 runtime gate 已在 commit `47fde37` 通过。独立 V2
    official-train 单行 Parquet SHA-256 为 `2c6a6c9b...e8184c`，与 V1 使用同一 row/image
    以隔离 prompt 变化；旧 V1 Parquet 仍字节级复现为 `0de5b142...66199`。真实 Qwen
    processor 得到 975 prompt tokens，26/26 checks 全真；renderer-owned action 经 strict
    parser 后在 pinned runtime 显示一个像素发生变化的 PIL image。没有模型权重、optimizer、
    checkpoint 或 protected split。
18. 唯一 V2 H800 smoke（Job `206227`）已完成，0 optimizer、0 checkpoint。16 次生成中
    intent/fence/syntax/argument/parser/execution 分别为 11/7/7/0/0/0；11/11 intents
    都复制无效 `_with_MODE`。V2 不再重跑，完整负结果和 raw outputs 原样保留。
19. N0 形式化/数值 gate 已完成并关闭：直接 expected-utility gradient 在零支持处为零；
    非零替代退化为 listwise/AWR/value-router。没有提交 GPU 或打开新 outcome。
20. N1 流式机器盘点已完成：四主 bank `59,949` decisions / `299,745` rows，完整性与
    provenance 通过；stop/selection 可识别，evidence-use 不可识别，且动作族、replicate、
    同数据集主 backbone 因子不足。现有-assets benchmark 路线关闭。
21. N2 已完成并关闭：stop/selection 非负分解成立，prefix/evidence signed-effect 分解也
    成立，但 ideal evidence-use regret 不可识别，且与 The Illusion/GapSight 直接碰撞；
    `authorized_new_gpu_jobs/checkpoints=0/0`。
22. N3 已完成：VTool 3B/7B 权重公开、许可/revision/model family 通过；精确 code mapping、
    prompt/parser contract 与 exact-artifact execution trace 失败，baseline 4/7。TACO、TAPO、
    The Illusion、ToolVision 已覆盖五个候选 core claims，novelty 0/6；联合 gate 失败，
    `downloaded_checkpoint_bytes=0`、新 GPU/checkpoint `0/0`。
23. N4 information-boundary audit 已完成：14/14 checks 全真，RICO 35,352/35,352 required
    images 可 decode；19 个 dimension mismatch 保留为 QC 风险。没有读取 action outcome、
    GPU 或 checkpoint。
24. V3 如有必要只承担 concrete-template 强 baseline，不作为新颖贡献，且必须使用独立
    row/seed 预注册。其他 validation/test/reserve 继续封存；本地修改不 push GitHub。
25. N5 在 outcome 前预注册 question-only/统一 preview/full-resolution privileged 三种信息
    集、固定四格 action bank、全部成本、强基线、source bootstrap 和最小实际效应；只用
    已打开的 ranker-training 资产做回顾性否证。结果 10/10 artifact checks 通过但 8/8
    scientific conditions 失败：DocVQA source-balanced higher-minus-lower 为
    `-0.00022424`，paired 97.5% CI `[-0.00528886, 0.00430109]`；ScreenQA OOF 差仅
    `0.00004824`。因此未打开 risk-calibration，formal-test/reserve 继续封存。
