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

### Post-action probe 操作化修订（2026-09-03 21:05 HKT）

该 probe 在正式 validation/test outcome 生成前固定为唯一一个 hidden sizes 为
`[128,32]` 的两层 MLP，不做 linear/architecture/feature 变体搜索。输入严格来自固定工具
已经执行后的可观察信息：baseline entropy/max probability/margin；按 action ID 排列的四个
crop 各自 entropy/max probability/margin；entropy 选中 action 的 one-hot 与 normalized
bbox；以及在相同 Qwen 上用“原图、选中 crop、原 model prompt”顺序重建的 generation 前
final-layer pooled language/visual/fused prompt states。它不读取 ground truth、
`correct_before/correct_after`、生成答案文本或 target 派生量。

训练 target 仅为 direct gain；source weighting、validation score calibration、threshold、
三个固定 seeds 和 test 一次性评估规则与 deployable probes 相同。其 call mask 是拥有
post-action 信息后的 retrospective privileged policy，因此不具有物理可部署性，也不能与
pre-action 方法混称为新 router；只用于判断 PIVOT 条件中“额外视觉证据出现后是否才有可用
utility signal”。feature bundle 因此升级为 format v2，pre-action typed view 仍完全忽略并
拒绝吸收这个独立 namespace。

## 数据与泄漏边界

每个 benchmark 必须具有 train/validation/test 三个角色，按 `source_id` 与解码后 RGB
SHA-256 的联合连通分量隔离；任意角色间 source 或像素内容重合都必须为零。validation
只可用于模型变体、阈值与 calibration；test 只允许在所有选择冻结后运行一次。

### 两阶段执行修订（2026-09-03 21:27 HKT）

正式执行不能调用同时接收 train/validation/test 的一键 runner。第一阶段的类型与 CLI
只能接收 train、validation feature 和 sibling rollouts；拟合全部 36 cells、三个 seeds、
三个 post-action diagnostic probes 与六个强基线后，必须先原子保存模型 bundle 和严格
JSON freeze inventory。inventory 记录 development source/RGB identity digest、全部
validation-selected variant/threshold/calibrator/baseline、protocol/input/code hashes，并
明确 `test_data_present=false`。模型 bundle 只允许在提供完全匹配 SHA-256 时从可信本地
路径加载。

第二阶段的独立 CLI 只接收已经持久化的 freeze 与 test artifacts，禁止任何 fit、variant/
threshold/baseline selection。它先验证 freeze report/model/protocol/code 一致，再在读取或
哈希任何 test manifest、rollout 或 feature 前，以 exclusive-create 写入不可自动覆盖的
test-access ledger；若后续中断，ledger 保留且程序拒绝自动重跑。加载后还必须将 test 的
source 与 decoded-RGB hashes 和 bundle 内的 development identity index 比较，任一重叠即
在计算指标前 fail closed。旧的一键接口只保留给 `formal_claim_eligible=false` 的 synthetic
smoke。

同一修订同时把正式 feature execution contract 固定为真实 smoke 已验证的配置：
Qwen2.5-VL-3B-Instruct revision `66285546d2b821cf421d4f5eb2576359d3770cd3`，generation
seed `0`、`max_new_tokens=16`、UG-grid ratio `2.0`、每 crop cost `1.0`、bf16、SDPA、
`min/max_pixels=200704/602112`、offline-only、强制 prompt hash。feature format 为 v2，
正式维度必须逐 role 稳定为 pre-action `3/22/6147/6147`、post-action `6167`；任一 input
spec、feature metadata 或代码 revision 不一致都 fail closed。

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

### 强基线操作化修订（2026-09-03 20:27 HKT）

这次修订发生在正式 validation/test outcome 生成前，只消除上述名称的实现歧义，不改变
研究问题、模型、数据角色、endpoint 或终局门槛：

- 四个 crop action ID 固定为 `ug-grid-00/01/02/03`；entropy gate 以
  `entropy_before` 为分数，高于等于 validation 阈值时执行固定四-crop 工具并收四次成本；
- random gate 以 seed `20260903`、`state_id` 和 `replicate_id` 的 SHA-256 前 64 bits
  构造 `[0,1)` 分数，按同一 validation-only 规则选阈值，调用时也执行固定四-crop 工具；
- matched-gate fixed crop 完全复用 entropy gate 的阈值与 call mask，只在 validation 上
  选择一个全局固定 action ID，调用时只收一次对应 crop 的成本；
- uniform-random crop 复用相同 gate，评测时对四个已注册 crop outcome 取精确均值以消除
  Monte Carlo seed 方差，不用 crop outcome 做选择，调用成本为一个 crop；
- exhaustive UG 总是执行四个 crops，按 post-action entropy 和 action ID 稳定选答案，
  收取四次成本；answer-now 永不调用；
- strongest baseline 只按 validation 的 source-balanced utility 选择；并列时依次选择成本
  更低、call rate 更低、名称字典序更小者。冻结后的 threshold、random seed、fixed action
  和 strongest baseline 原样应用到 test，test outcome 不参与任何选择。

比较 learned candidate 与 strongest baseline 时，两者保留各自逐样本 `Ytool`、tool cost
和 call mask，再在相同 source 上做 paired bootstrap；禁止把 one-crop baseline 错套到
four-crop outcome/cost ledger。

## 当前证据与缺口

现有 ChartQA、DocVQA、V*Bench sibling bank 足以验证固定工具的标签构造和 runner，但不
构成正式矩阵：旧 rollout 没有 max probability/top1-top2 margin，缺 L3 完整状态，部分
旧 split 也不满足 RGB/source 双重隔离，且 HRBench 尚未生成。故在新数据和特征完成前，
矩阵完成度必须报告为 `0/36`，不得从旧负结果直接宣布该 research question 失败。

## 唯一终局分类

### 冻结选模、种子聚合与 cost-independent 校准修订（2026-09-03 22:30 HKT）

本修订发生在任何正式 train/validation/test outcome 生成前。每个 seed 只按 validation
source-balanced utility 在全部已注册 deployable cells 中选一个 policy；并列时依次选择调用
更少、predictor level 冻结顺序更早、target 冻结顺序更早的 cell。三个 seed 的 call mask
做严格多数投票，构成每个 benchmark 唯一 primary deployable。L3 representation diagnostic
在 L3 内按相同规则选 target，post-action probe 也按三个 seed 多数投票。非正式的两-seed
smoke 若平票一律不调用。

概率校准标签固定为 `Ytool-Y0 > 0`、rescue 与 harm，全部与 cost 无关；`lambda=0.05`
只允许进入 validation threshold 和 policy utility，禁止进入模型或 calibrator 标签。GO 的
Pareto 条件在上述冻结 operating point 上检查：accuracy 不低、visual cost 不高且至少一项
严格改善；rescue precision 必须严格更高，harm/call 不得更高。零调用导致的 undefined
rescue/harm rate 按预注册约定比较为 `0.0`。PIVOT 的“全部 deployable 失败”要求每个
cell-seed policy 和 primary ensemble 相对 strongest baseline 的 paired 95% lower CI 最大值
仍不大于零，不能只看 seed 平均。

### 一次性 test 事务修订（2026-09-03 22:30 HKT）

test 不再预先生成再由 evaluator 写 ledger。Phase A frozen bundle 完成后先建立只包含冻结
模型、协议、代码版本、预注册 allocation digest 与 test 路径的 transaction plan。单个 H800
job 在检查 GPU、离线权重和所有非-test 依赖后，首先 exclusive-create 永久 access ledger；
此前不得 stat、hash、计数或加载 test allocation/manifest。ledger 成功落盘后，同一 job 才
依次生成 ChartQA、DocVQA、HRBench 的 test rollouts/features，封存 hashes，应用 frozen
matrix，进行 20,000 次 paired whole-source bootstrap，并在终局规则有唯一答案时生成
`PREDICTABILITY_AUDIT.md`。任何中断均保留 ledger，禁止自动重提；只能人工审计后做精确
恢复。

只有 36 个 cell、强基线、paired bootstrap 与一次性 test 全部完成后，才能生成
`PREDICTABILITY_AUDIT.md` 并给出一个结论：

- `STOP`：至少两个 benchmark 的 privileged binary oracle utility 不超过 `0.005`；
- `GO`：至少两个 benchmark 上冻结的 validation-selected、三-seed-majority primary
  deployable 相对 strongest baseline 的 paired 95% lower CI 大于零，并通过已注册的
  operating-point Pareto/rescue/harm 条件；
- `REPRESENTATION`：至少两个 benchmark 的 L3 in-domain improvement lower CI 大于零，
  但至少两个 benchmark 的 image-disjoint 或 cross-domain upper CI 不大于零；
- `PIVOT`：至少两个 benchmark 有大于 `0.005` 的 oracle utility，post-action majority
  probe 相对 answer-now 的 utility lower CI 大于零，而且全部 individual cell-seed 与
  primary deployable 的 lower CI 在三个 benchmark 上均不大于零。

判断优先级冻结为 `STOP → GO → REPRESENTATION → PIVOT`。若完成后仍不满足任何规则，
runner 必须报 inconclusive 并拒绝生成终局分类，而不是临时改门槛。
