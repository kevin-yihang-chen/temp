# N4 选择器信息边界与排名反转形式化 gate

时间：2026-09-03 15:33（Asia/Hong_Kong）

## 结论

机器决定为 `n4_information_boundary_candidate_survives_formal_gate`，14/14 项检查
全真。N4 只保留为一个**待真实数据证伪的评测候选**：对每个视觉获取方法显式登记选择器
在动作前可见的信息，只有在相同可见信息、相同 action bank 和相同净效用定义下才比较
方法，并额外检验方法排序是否会随选择器信息集改变。

这不是方法成功，也不是论文主结果。当前正证据只来自精确 toy construction 和已有 RICO
图像的 outcome-free 可用性；没有读取现有 action outcome、没有训练模型、没有提交 GPU，
新增 checkpoint 为 0。N5 必须在结果读取前冻结真实数据协议，并观察到稳健且有实际意义的
matched-visibility 差异或排名反转，N4 才能继续；否则关闭。

## 为什么更换 N4 候选

最初考虑的是：从低带宽 preview 自监督预测未观察高分辨率 crop 的分布，再用预期任务风险
下降决定是否调用工具。碰撞审计表明，这个机制不是独立空白：VOILA 已做 pre-retrieval
value-of-information fidelity selection，Learning to Look Around 已做 label-free unseen-view
completion，AdaptVision 已做低分辨率到局部高分辨率的自适应获取，Starve to Perceive 也已
直接研究受限视觉带宽。继续实现只会成为现有 active perception/VOI 的组合变体。

因此 N4 转向一个更基础、且能利用当前负结果检验的问题：视觉工具/裁剪方法的 selector
究竟看到了什么？如果一个 crop selector 在决策前已读取完整高分辨率图像，而另一方法只看
问题或低分辨率 preview，那么它们的准确率、utility 与工具调用率不是同一信息条件下的公平
比较。候选主张不是“aliasing 是新的”，而是把 selector-visible information 作为视觉获取
评测的一等协议变量，并报告 matched-visibility 结果及跨信息集 rank reversal。

## 形式化边界

设真实世界为 `w`，选择器允许观察的信息状态为 `z(w)`，动作 `a` 的效用 `U(w,a)` 已扣除
视觉获取和 proposer 成本。定义：

- `V_full = E_w[max_a U(w,a)]`：知道完整世界后选动作的上界；
- `V_obs = sum_z max_a sum_{w:z(w)=z} p(w) U(w,a)`：只允许看 `z` 的 Bayes 上界；
- `V_pi = E_w[U(w, pi(z(w)))]`：实际可部署策略价值。

机器验证：

`V_full - V_pi = (V_full - V_obs) + (V_obs - V_pi)`。

第一项是信息表示造成的 aliasing regret，第二项是给定信息边界下的 policy estimation
regret。两个不同高分辨率 2×4 raster 经 2×2 block mean 后得到完全相同 preview；若两者
最优 crop 相反，则 `V_full=1.0`、`V_obs=0.5`、aliasing regret `0.5`。细化观察后该项变为
0；若同一 alias cell 的最优动作相同，该项也为 0。一个在相同 preview 下按隐藏世界选择
不同 crop 的 oracle 会被机器判为违反信息边界。

但这段分解不能当作新理论。Self-Certification of Representation Adequacy 已直接形式化
Bayes-risk representation adequacy、alias cell 内的 action conflict 和 representation
aliasing regret；经典 perceptual aliasing 与 VOI 也属于明确先验工作。代码中的分解只作为
评测合同的校验器。

## 候选贡献与碰撞边界

预注册的五项组成中，两项已被覆盖：

- `aliasing_vs_policy_regret_decomposition`：被 Self-Certification 直接覆盖；
- `joint_acquisition_and_proposer_cost_ledger`：VQABench 已把 client-side preprocessing
  纳入端到端 VQA 成本，至少部分覆盖；N4 只能要求完整执行，不能声称首次提出。

本轮检索后仍暂未发现直接覆盖以下三项联合协议的视觉工具/VLM 工作：

1. `selector_input_information_ledger`：逐方法登记动作前可见字段与其生成成本；
2. `matched_visibility_method_comparison`：只在相同信息集、action bank 和 utility 定义内
   给主排名；
3. `cross_information_set_rank_reversal_test`：显式报告方法相对排序是否因 selector
   获得额外图像信息而改变。

“未发现”仅是截至本轮的一手文献初筛，不是新颖性证明。真实实验若没有显示现有结论会因
信息边界而实质改变，这三项也不足以构成顶会贡献。

主要相邻来源：

- VOILA：https://arxiv.org/abs/2602.03007
- AdaptVision：https://openaccess.thecvf.com/content/CVPR2026/html/Lin_AdaptVision_Efficient_Vision-Language_Models_via_Adaptive_Visual_Acquisition_CVPR_2026_paper.html
- Starve to Perceive：https://arxiv.org/abs/2605.18603
- Self-Certification of Representation Adequacy：https://arxiv.org/abs/2608.02267
- The Illusion of Visual Tool-Use：https://arxiv.org/abs/2608.06270
- VQABench：https://arxiv.org/abs/2608.07861
- Learning to Look Around：https://openaccess.thecvf.com/content_cvpr_2018/html/Jayaraman_Learning_to_Look_CVPR_2018_paper.html

## 机器检查

除了上述 regret identity，新增的信息边界合同还验证：

- 同一比较内若 `information_set_id`、`selector_visible_fields`、`action_bank_id` 或
  `utility_definition_id` 不一致，则 fail closed；
- toy preview-only 条件下 conservative selector 的 `0.6` 高于 adaptive selector 的
  `0.5`；允许 selector 读取 full-resolution 信息后 adaptive 为 `1.0`，高于 conservative
  的 `0.6`，程序检测到严格 pairwise rank reversal；
- 一个 raw task utility 为 `0.70` 的 costly proposer，在同时扣除 `0.05` acquisition 与
  `0.06` proposer cost 后为 `0.59`，排名低于无额外成本、utility `0.65` 的 baseline；
- required uncovered claims 必须逐项仍未被 registry 文献覆盖，不能用任意三个弱 claim
  凑数过关。

这些数值只证明审计逻辑能检测所需现象，不是现实效应量。

## 真实数据可行性与风险

既有 RICO integrity report 的三个必要可用性 gate 全真：35,352/35,352 张 required image
可 decode，JPG/JSON 对齐且所有非坏图像被 allocation component 覆盖。19 个 annotation/image
dimension mismatch 原样保留为 QC 风险，不能隐藏或事后删除。现有 ScreenQA 分配有
6,007 个 `ranker_training`、4,001 个 `risk_calibration`、6,000 个 `formal_test`、1,004 个
`reserve` 和 11,348 个 `untouched` image；image/component 跨角色 overlap 均为 0。

N4 本轮只绑定 integrity report，没有读取任何 action outcome。RICO 的可用性只说明 N5
可以构造 label-free preview/geometry 信息，不证明 ScreenQA outcome、selector 排名或论文
结论成立。

## N5 预注册要求与停止规则

N5 在打开既有 sibling outcome 前必须冻结：

1. selector 信息集（至少 question-only、question+统一低分辨率 preview；full-resolution
   只能作为明确标注的 privileged/leakage diagnostic）；
2. 完全相同的 `ANSWER_NOW + 4 UG-grid ZOOM` action bank、source-disjoint role、task score
   与 acquisition/proposer/token/latency 成本；
3. entropy gate、random/fixed crop、exhaustive UG、现有 learned router 等强基线及统一拟合
   预算；
4. primary rank-reversal statistic、bootstrap unit、最小实际效应和多重比较规则；
5. 只用 `ranker_training` 拟合、`risk_calibration` 做一次性 N5 screen，继续封存
   `formal_test/reserve`。

只有在 matched-visibility 下至少一个既有主排序/优势结论发生预注册的稳健反转，或发现足以
改变方法结论的显著 utility gap，且结果在 source bootstrap 与成本敏感性下保持，才允许
把 N4 扩展为多数据集 benchmark。没有反转、效应很小、只在 privileged full-resolution
成立，或近期文献已直接覆盖三项联合协议，都立即关闭，不提交 GPU 追结果。

## 复现与哈希

- registry SHA-256：
  `25dd301a76e933d663ccb95ab9513bed2170e89b401159801e8c3e0bdeebe12f`；
- module/runner/test SHA-256：
  `eea7468ba45cf7237c322f345bd8a48c70ab61a691b5be610b628e6ca25d7a0d` /
  `5740d5d6cf1f75722d93602438d96dc79dd69fcb93d098314158abfe3e0e0fc6` /
  `8e2baa9e5f50b81fe54f0e6811eb1cf11d0e2aa1eebc3bd8e210813d22c925a0`；
- N4 report SHA-256：
  `d34449b6b10b8b49f7264bcabe4866e0b225ec586b1a7454743a2f6d8aa6ef5c`；
- RICO integrity report SHA-256：
  `ff239a93f12a0e0e173a7d13219f0f605d028807539c02e2d2a11a83de360a73`；
- 上游 N3 report SHA-256：
  `6d145cba4846ff608788b5dc8791d7fabcd0cdd1380ff9f2907bea5be3394f5c`。

复现命令：

`PYTHONPATH=.:src python scripts/audit_selector_information_boundary.py --registry configs/n4_selector_information_boundary_v1.json --n3-report artifacts/docvqa-train-factorized-v2/ops/n3-tool-checkpoint-novelty-audit-v1.json --expected-n3-sha256 6d145cba4846ff608788b5dc8791d7fabcd0cdd1380ff9f2907bea5be3394f5c --rico-integrity-report artifacts/screenqa-train-factorized-v1/image-integrity-audit-v1/report.json --output artifacts/docvqa-train-factorized-v2/ops/n4-selector-information-boundary-audit-v1.json`

最终验证已通过：12 个 N4 targeted tests、N4+N2 共 24 个 targeted tests 与全仓 pytest
均通过；三文件 mypy、compileall、Black in-process check、两次独立报告字节比较、N3 hash
负路径、JSON 断言、凭证扫描和 `git diff --check` 也全部通过。Black CLI 在当前 NFS 环境
等待超过 60 秒，终止后改用同版本 formatter 的无 cache in-process check；这是验证命令的
运行方式问题，不是格式失败。本轮产物只保留在本地，不 push。
