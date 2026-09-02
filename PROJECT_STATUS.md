# 项目状态

更新时间：2026-09-02 13:16（Asia/Hong_Kong）

## 总体判断

项目仍在推进，但尚未形成 ECCV/ICCV/CVPR 可投稿主结果。原始“仅用 uncertainty 决定
工具调用”的宽泛命题已被多次强基线削弱；当前最可信的新证据是 stop/where
瓶颈的可重复因子化，而不是一个已经成功的部署策略。

## 已完成的阶段与证据

1. 完成 InfographicVQA official-train 23,946 decisions、2,204 sources、4,406
   images 的 raw-attention outcome-free 特征抽取、合并与无泄漏审计。
2. Raw-attention train gate（Job `203276`）正式得到
   `attention_where_train_not_supported`，validation/test 保持封存。
3. 尽管总 gate 失败，5% 和 10% nominal call-rate 上 raw action 对 fixed、random、
   old-DECAR-where、relative-where 的 paired 95% lower endpoints 全部为正，证明
   `where` 有真实改进。
4. 所有 raw operating point 的 source-balanced utility 均为负；5% 点为
   `-0.000410`，95% CI `[-0.002438, 0.001681]`，helpful-call precision 仅
   `19.35%`。当前主要瓶颈是 stopping 与 residual harm。
5. Raw action 全状态 task-oracle crop agreement 为 `44.45%`，最高 attention-max
   decile 为 `50.65%`；相应 helpful-state rescue 从 `64.06%` 升至 `85.86%`。
6. Fixed-action stop-factorization（Job `203290`）完成。Raw action 共有
   1,023 个 positive-net states（483 sources）；其 privileged positive-net stop
   ceiling 为 `+0.021318`，95% CI `[0.018447, 0.024444]`，证明主要
   剩余 headroom 确实在 stopping。Attention max/margin stopping 在所有注册
   call rates 均比 entropy 更差，不能作为候选。
7. 已在 protocol commit `e0f95d2` 中预先冻结单一 fixed-action
   signed-value stop 候选，并在 implementation commit `0683526` 完成实现。
   真实输入 smoke 确认 80 维特征全部有限，五折 source overlap 均为
   0；未拟合模型或打开 OOF 策略结果。
8. Fixed-action signed-value OOF（Job `203330`）在唯一 2% primary 上得到
   `fixed_action_signed_stop_train_not_supported`。其 positive-net precision
   从 entropy 的 `16.08%` 提高到 `18.79%`，但 utility 仍为
   `-0.000063`，95% CI `[-0.000739, 0.000655]`；candidate-minus-entropy
   为 `+0.000522`，paired CI `[-0.000304, 0.001444]`。只有 precision
   条款通过，不允许 calibration 或事后调参。

## 当前最佳结果

- 最强 deployable where 证据：raw-attention action，在 entropy 5%/10% call set
  上显著优于四个注册 where 基线，但净 utility 未过零，不能 calibration。
- 非部署上界：task-oracle where 在 entropy call set 上通过 where-bottleneck
  诊断，说明 action localization 的可学习 headroom 存在。
- 新的诊断上界：固定 raw action 后的 privileged stop utility 为
  `+0.021318`，远高于 entropy 5% 点的 `-0.000410`；这只是学习
  stopping 的充分动机，不是 deployable 结果。
- 最新 deployable OOF 证据：signed-value stop 在主点比 entropy 有正向但不显著
  的 utility 差异，且自身 utility 不能过零；该候选已正式失败。
- 尚无满足主张、校准、跨域与正式测试要求的最终方法。

## 正在运行

| Job | 内容 | 资源 | 状态（13:16 快照） | 关键产物 |
|---:|---|---|---|---|
| 203273 | ViCrop/LASER literature attention 完整抽取 | 2×H800，16 CPU，192 GiB | RUNNING，wave 1 两分片均至少 2,560 decisions | `literature-attention-where-v1/` |

Job `203330` 已于 13:13 正常完成，runtime `00:07:00`，exit `0:0`。
Job `203290` 已于 12:44 正常完成，runtime `00:18:09`。所有计算任务
均启用全状态邮件。当前改动均为本地提交，未 push GitHub。

## 失败、风险与解释边界

- Raw-attention where-only 已正式失败；不得通过更改 utility、cost、bootstrap 或
  事后阈值把它改写为成功。
- ViCrop/LASER 仍使用 entropy stop，因而即使 where 更强也可能继续负 utility。
- Stop diagnostic 使用已打开的 train outcomes，只有诊断价值；任何后续方法都
  必须另行冻结并 whole-source OOF，不能把 privileged ceiling 当部署结果。
- Simple attention confidence 已被证据否定；不应继续搜索 attention
  max/margin 变形。后续 stop 候选已按单候选、低容量且
  source-held-out 的规则完成评估。
- 单一低容量 source-OOF signed-value stop 也已失败。它有轻微 precision
  和 harm 改善，但不显著且 utility 不过零；禁止在当前已打开 outcomes
  上继续调该模型族。
- 当前论文新颖性仍不足。Generic action-value classifier 本身与 selective VQA、
  learning-to-defer 等工作重叠；论文级贡献必须来自反事实工具收益、stop/where
  因子化、风险控制与跨数据集证据的组合。
- Literature extraction 当前吞吐预测总耗时约 6.5--7 小时，接近但低于 8 小时
  限制；checkpoint/resume 已启用。输出峰值预计约 5 GiB，当前磁盘约 52 GiB
  可用，不构成阻塞。

## 下一步最优行动

已关闭 H3 的局部调参路线。等待 Job `203273` 的冻结 literature where
强基线；让其完成而不
改变其冻结设置。禁止继续调 raw attention 层、head、max/margin 或 entropy
threshold。
