# InfographicVQA post-attention pivot audit v1

状态：2026-09-02 在 Job `203340` 给出
`literature_attention_where_train_not_supported` 后编写。本文只决定下一步信息
来源与止损边界，不是实验 protocol，不授权读取 validation/test/reserve、拟合模型
或提交计算任务。

## 已知约束

1. 工具 headroom 不是零：固定 raw action 的 privileged positive-net stop utility
   为 `+0.021318`，95% CI `[0.018447, 0.024444]`。
2. Raw attention 在相同 entropy calls 上显著优于四个旧 where baseline，但净
   utility 为负；ViCrop/LASER 没有显著改善 raw attention。
3. Attention max/margin、ENCORE early entropy 与现有 80 维特征上的线性
   signed-value OOF stop 都失败。
4. Positive-net states 稀疏且跨 source 异质；下一候选必须提升 tail precision，
   不能只改善平均 action agreement。
5. Generic pre-call gating、necessity/harm learning、attention cropping 与 continuous
   crop routing 均有强文献碰撞。单纯换模型或 action geometry 不是足够贡献。

## 候选矩阵

| 方向 | 新信息 | 是否直击 stopping | 在线成本 | 新颖性/泄漏风险 | 决定 |
| --- | --- | --- | --- | --- | --- |
| 当前 80 维特征上换 MLP/tree/C/seed | 无 | 可能仅拟合更强 | 低 | 已打开 outcomes 上 model-family search；高过拟合 | 关闭 |
| 继续改 attention layer/head/ratio/max/margin | 无实质新观测 | 既有证据为负 | 低 | 与 ViCrop/LASER/ENCORE 直接重叠 | 关闭 |
| 直接换 continuous crop proposer | 新 action geometry | 主要改善 where | 中到高 | GapSight/CropVLM/AdaptVision 碰撞强；不能解决 fixed-action stop 失败 | 暂不优先 |
| OCR/text-semantic retrieval proposer | 显式文本内容与位置 | 可提供“是否存在相关小字”信号 | OCR 前向需计费 | InfographicVQA 特化，跨自然图像泛化风险高 | 备选/消融 |
| 分辨率退化或多视图 disagreement | 对 fine-detail 依赖的反事实敏感性 | 是 | 若全量多一次 VLM prefill，成本过高 | 与 adaptive resolution/uncertainty 路线需额外审计 | 仅保留低成本实现 |
| Answer-conditioned evidence consistency | 已生成答案 token 的语义与区域 grounding，而非仅 question attention | 是；可区分高 entropy 与缺乏视觉证据 | 若复用 baseline generation 内部量则无额外 tool call；离线需重抽特征 | 仍有 attention/grounding collision，但输入语义与旧特征不同 | 唯一优先可行候选 |
| 完整 sibling-bank empirical audit | 不产生新 policy；整合跨域 gain/harm/where/stop 证据 | 解释失败而非解决 | 低 | 与 MED/Illusion 有碰撞，需突出 candidate-level prospective protocol | 方法候选失败后的主 fallback |

## 唯一优先候选的前置审计

在冻结任何结果实验前，只允许做 outcome-free feasibility audit：

1. 确认 baseline answer 在部署决策时已经生成，因此 answer token 属于合法
   pre-action state；不得使用 target/reference answer。
2. 确认 answer-token visual grounding、question-token grounding 与区域级一致性可从
   同一次 baseline generation 的缓存/attention 中得到；若在线需要额外完整 VLM
   forward，必须先证明其成本低于预期收益，否则直接淘汰。
3. 明确与旧 80 维 features 的差异：旧特征只有 answer 长度、类型和 token entropy，
   没有 answer 语义或 answer-to-region grounding。
4. 在不读 outcomes 的前提下检查 feature coverage、有限值、token/region alignment、
   显存与吞吐；不允许同时比较多层、多 head pooling、多种 score 或多种模型。
5. 补做针对 answer-conditioned grounding / evidence-consistency routing 的 primary-
   source 文献审计；如果直接先例已覆盖同一 estimand 与部署点，则不立新方法线。

只有上述五项通过，才允许写一个单候选 source-OOF protocol。该 protocol 必须固定
raw-attention action，仅改变 stopping information；唯一 primary 仍使用预先指定的
matched-call budget，并要求 candidate utility 与 candidate-minus-entropy 的
whole-source lower endpoint 同时大于零。所有额外 baseline 计算都必须计入成本。

## 终止条件

- Feasibility/cost/literature 任一前置审计失败：不提交 GPU job，直接进入 empirical
  audit 路线评估。
- 单一 answer-conditioned 候选在冻结 source-OOF primary 失败：关闭 InfographicVQA
  正方法开发，不再从当前 official-train outcomes 派生新 stop feature。
- 即使 train OOF 通过，也只授权单独 calibration protocol；不得立即打开
  validation/test 或宣称顶会结果。

这个设计把下一次实验定义为最后一次机制上不同、信息来源明确的 stop 候选，而不是
无限续命。它也保留了一个诚实的失败出口：若新信息仍不能识别稀疏正收益尾部，项目
应转为审计论文或降低投稿目标，而不是继续调整局部超参数。
