# N5 信息集效应回顾性否证协议

时间：2026-09-03 15:52（Asia/Hong_Kong）

## 目的与证据等级

N4 只在 toy construction 上证明了“选择器信息边界可能改变方法排序”，尚没有现实效应。
直接打开 ScreenQA `risk_calibration` 将涉及 9,951 个 QA decision、49,755 条
`ANSWER_NOW + 4 ZOOM` action 记录。N5 先复用**此前已经打开**、同一 DocVQA sibling
bank 上的两个冻结模型，检验增加 full-resolution semantic selector 信息是否至少满足进入
新 calibration 的必要条件。

这是回顾性路线止损，不是预注册 confirmatory/formal 结果。写本协议时，两个旧 evaluation
的 question-weighted aggregate 已知：context 模型 utility 为 `-0.00535664`，semantic 模型
为 `-0.00572917`。尚未计算本协议定义的逐 source 配对差、共同 5% 调用预算、统一强基线
排名或额外信息成本敏感性。所有结果必须如实标为 retrospective，不得重新解释旧 formal
split，也不得用本项选择阈值、特征或模型。

ScreenQA ranker-training 的旧 OOF aggregate 也已知：context 与 semantic utility 分别为
`0.00006547` 与 `0.00011371`，两者的 tail-risk audit 都是
`no_non_degenerate_safe_threshold`。本项只把这些已知点估计按预先写明的最小实际效应和
安全阈值条件纳入机械止损，不把它们包装成新盲测。

## 假设

若 N4 剩余的 selector-information contribution 在现实数据上值得继续，则在完全相同的
1,608 个 decision、400 个 source、四个 UG-grid action、`lambda=0.05` 和 outcome scorer
下，加入 full-resolution semantic 信息的冻结模型应当在共同 5% 调用预算上：

1. source-balanced utility 为正，97.5% whole-source bootstrap 下界大于 0；
2. 相对较低信息的 context 模型至少提升 `0.001` utility，且配对 97.5% 下界大于 0；
3. 即使把额外 full-resolution feature acquisition 成本乐观地设为 0，上述条件仍成立；
4. ScreenQA ranker-training OOF 上 higher-minus-lower 的既有 question-weighted utility
   差也至少为 `0.001`，且 higher-information 路线存在非退化安全阈值。

第 3 条是对高信息模型最有利的上界：其 semantic features 每个 decision 都需额外读取原图
并做视觉/多模态计算，任何非负真实成本只会进一步降低净效用。若零成本上界都失败，无需
估算具体 GPU 秒或 token 成本来挽救候选。配置中的额外成本敏感性单位是“已经乘过成本权重
的每 decision utility”，直接从 source-balanced utility 扣除。

## 信息集账本

较低信息模型 `context-geometry` 可见：规范化问题表面、baseline answer 表面、baseline
token entropy 统计和候选框几何。它不是 question-only selector，因为已经复用了 baseline
VLM 的回答及不确定性。

较高信息模型 `semantic-context` 在上述之外还可见：原图条件下的 question embedding、
原图 global visual embedding、四个候选 ROI embedding 和 question-to-region attention。
这些特征不含 answer correctness、delta-success 或其他 action outcome；label-free audit
必须重新验证。

现有资产**没有** N4 原先要求的 question-only、统一低分辨率 preview，亦没有同一算法在
每个信息集下重新拟合的完整 factorial。因此本项不能识别严格的 cross-information-set
rank reversal；它只检验当前可用 nested-information candidate 的必要现实效应。模型容量
也并非严格匹配，所以失败只能关闭“现有候选足以支持 N4”的路线，不能证明视觉信息本身
永远无用。

## 输入与不可变绑定

配置 `configs/n5_information_set_retrospective_v1.json` 固定：

- N4 report SHA-256 `d34449b6...aa6ef5c`；
- DocVQA rollouts SHA-256 `a7f44c26...0b5aa3`；
- context model SHA-256 `33f2e0b1...8496b`；
- semantic model SHA-256 `1f8b6cf5...06ff3`；
- label-free semantic features SHA-256 `bc58c169...48ba7`；
- feature audit SHA-256 `c605952e...ec2e`；
- 两个既有 evaluation、两个 ScreenQA OOF report 与 ScreenQA allocation audit 的完整哈希。

两个 DocVQA 模型都必须是 `source_grouped_oof_v1`、5 folds、seed `20260828`、
`selected_alpha=10`、相同 domain 与 `lambda=0.05`。旧阈值仅用于复现原 frozen-policy
诊断；primary 使用共同的 5% score-top-k 调用预算，按 score 降序、decision key 升序破同分，
不读取 outcome 决定调用集合。

## 指标、强基线与统计

Primary 为 5% matched-budget 下每个 source 内先平均、再跨 source 平均的净 utility。
对 400 个 source 做 20,000 次 iid whole-source paired bootstrap，seed `20260903`，报告
97.5% percentile interval。一次 bootstrap 同时重采样两个模型和所有 baseline，以保持
paired comparison。

除两个 learned router 外，统一报告：

- `ANSWER_NOW`；
- entropy gate + expected random crop；
- entropy gate + 四个 fixed crop；
- random gate + expected random crop；
- post-action minimum-entropy 单执行的乐观 idealized comparator；
- 执行四个候选并扣四份成本的 exhaustive UG-style comparator；
- privileged oracle ceiling。

后两类明确使用 action outcome 或 post-action entropy，只能标作 idealized/privileged，
不能作为 deployable 结果。强基线结果是诊断，不允许从中事后改 primary rate 或模型。

## 机械停止规则

以下任一项失败，N5 决定为
`n5_current_information_boundary_candidate_not_supported_before_calibration`：

1. exact N4 所需的三种信息集和同方法 factorial 不可用；
2. 高信息模型 5% utility 不为正或 97.5% 下界不大于 0；
3. 高信息减低信息的 paired point effect 小于 `0.001`，或 97.5% 下界不大于 0；
4. ScreenQA opened-development OOF 的 higher-minus-lower utility 小于 `0.001`；
5. 输入哈希、同样本覆盖、label-free、模型训练协议、action bank 或成本合同不匹配。

停止后不得打开 ScreenQA `risk_calibration`，不得提交 GPU，也不得生成 checkpoint；关闭
的是当前 N4 benchmark/evaluation candidate。若全部通过，才允许另写一次性 ScreenQA
calibration 协议；即使通过，本项本身仍不是正式主结果。
