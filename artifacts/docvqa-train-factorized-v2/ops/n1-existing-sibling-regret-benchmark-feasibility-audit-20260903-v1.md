# N1 现有 sibling-bank regret benchmark 可行性审计

时间：2026-09-03 14:38（Asia/Hong_Kong）

## 结论

机器决定为
`n1_existing_assets_insufficient_for_top_tier_regret_benchmark`。现有资产规模大、
sibling 完整且可复现，足以识别 answer-now 相对工具动作的 stop regret，以及已注册
四个 UG-grid 候选内部的 action-selection regret；但不能识别 evidence-use regret，
也不满足同数据集主证据多 backbone、多工具动作族和随机重复要求。因此不能把现有 bank
直接包装成顶会级三段 regret benchmark，也不允许先生成更多同构数据再寻找主张。

## 冻结问题与判定条件

N1 在读取汇总结果前固定三个 estimand：

1. `stop regret`：在 answer-now 与调用工具之间做错停止决定产生的损失；
2. `action-selection regret`：已决定调用时，所选 action 相对完整注册 action bank 最优
   action 的损失；
3. `evidence-use regret`：action 已固定后，agent 对真实视觉 observation 的使用相对匹配
   counterfactual/no-op observation continuation 的差异。

第三项要求同一个 action prefix、同一个具体 action、匹配的 factual/counterfactual
observation，以及受控 continuation。共享 `generation_seed` 的 answer-now/ZOOM 独立推理
不等价于共享 action prefix 的 continuation，因此不能用现有 `answer_after` 冒充第三项。

顶会 gate 同时要求：至少三个主数据集、完整 sibling、source-level 标识、不可变复现
元数据、同数据集多 backbone 主因子、多于一个工具动作族、每状态多个随机 replicate，
以及三个 estimand 都可识别。十项中六项通过、四项失败。

## 只读盘点结果

| 主 bank | 模型 | decisions | rows | sources | sibling |
|---|---|---:|---:|---:|---|
| InfographicVQA | Qwen2.5-VL-7B | 23,946 | 119,730 | 2,204 | answer + 4 UG-grid |
| ScreenQA | Qwen2.5-VL-3B | 14,511 | 72,555 | 1,510 | answer + 4 UG-grid |
| DocVQA | Qwen2.5-VL-3B | 13,580 | 67,900 | 3,500 | answer + 4 UG-grid |
| TextVQA | Qwen2.5-VL-3B | 7,912 | 39,560 | 5,000 | answer + 4 UG-grid |

四个主 bank 合计 `59,949` decisions、`299,745` rows、按数据集直接求和 `12,214`
sources。每个 decision 都是唯一 answer-now 加四个 ZOOM；schema、row-level 模型 revision、
manifest SHA-256 和 rollout SHA-256 全部通过。四个主 bank 的 `239,796` 条 ZOOM 中，
具有完整 evidence-use intervention contract 的记录为 `0`。

另外两项只作为诊断，不提升主 gate：

- ScreenQA Qwen2.5-VL-7B 有 `512` 个 opened-development states，与 3B bank 精确重叠
  `512` 个 states；它不是主结果，规模也不足以构成多数据集多 backbone 因子；
- ChartQA Qwen2.5-VL-3B 有 `4,500` decisions，但 provenance 明确标为 diagnostic，
  不能事后改作 benchmark claim。

UG-grid bbox 会随图像长宽比变化，不能称为四个固定坐标框；但 proposer 和工具语义仍只有
一个 `ZOOM` family。所有主 bank 每个 state 只有一个 `replicate-000`，无法估计 decoding
随机性对 regret 分解的稳定性。模型覆盖也与数据集混杂：主 bank 中 InfographicVQA 只用
7B，其余三个只用 3B。

## 与最近工作的边界

- [The Illusion of Visual Tool-Use](https://arxiv.org/abs/2608.06270) 已在固定 prefix/action
  下用真实与反事实 observation 定义 visual evidence gain；缺少同类 matched intervention
  时，N1 不能声称测量 evidence use，补齐后又必须证明三段分解提供独立于该工作的结论。
- [GapSight](https://arxiv.org/abs/2608.21762) 已用 candidate crop loss gaps 训练
  gate/utility/box router；现有 complete sibling action bank 的 selection oracle 或 action
  value 统计本身不足以形成方法新颖性。

因此，单独发布前两项 regret 是不完整贡献；把第三项从 `answer_after` 推断出来则是错误
识别；仅增加行数、随机种子或另一批 UG-grid crop 也不会修复新颖性和 action-family 缺口。

## 可复现性

- 基线/实现 commit：`f7c944948942b28e4c4c2030b21138fe2930d436`；
- 机器报告：`n1-existing-sibling-bank-inventory-v1.json`，SHA-256
  `d17bb8eec9bf0f5cce89105d43c0a676b134ce0779b9f799a4c02903ae3d62c7`；
- module/runner/test SHA-256：
  `60b5fa33adaa81bf1232f798e473601a53a8987cf159e7e72f447c3b513bcd60` /
  `1daab181d989b844e15cb2b36aa3ff5f53950cd146f49973ec62d60a02c9fd5a` /
  `b72515bfab1b7a2c823502decab0a4e92ce0afcd7cc50943731f768c87f66f5b`；
- 复现命令：
  `PYTHONPATH=.:src python scripts/audit_sibling_bank_inventory.py --repo-root . --output artifacts/docvqa-train-factorized-v2/ops/n1-existing-sibling-bank-inventory-v1.json`；
- 五个 targeted tests、三文件 mypy、compileall、Black in-process check、第二次独立输出
  byte comparison、JSON 决策断言、凭证扫描与 `git diff --check` 均通过。

本项只流式读取已有 JSONL/provenance；没有加载模型、使用 GPU、提交 Slurm、执行 optimizer、
产生 checkpoint 或读取 validation/test/reserve。

## 决定与下一步

关闭“直接用现有 assets 做完整 N1 benchmark”的路线，不降低投稿目标。下一步 N2 仍先做
CPU/纸面 gate：给出一个严格可加的 stop/selection/prefix/evidence 分解，审计它相对 The
Illusion/GapSight 是否真有不可约命题，并对最小 factorial augmentation 做 sample、GPU-hour
和存储上界。只有新颖性、可识别性、同数据集多 backbone、多动作族和统计功效能同时冻结，
才允许生成新 intervention data；否则 N2 也在 GPU 前关闭。
