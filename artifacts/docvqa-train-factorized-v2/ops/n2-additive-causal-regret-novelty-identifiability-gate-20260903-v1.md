# N2 严格可加 causal regret 的新颖性与可识别性 gate

时间：2026-09-03 14:47（Asia/Hong_Kong）

## 结论

决定为 `n2_additive_causal_regret_candidate_not_identified_and_not_novel`。N2 的前半段
可以严格成立：stop regret 与 action-selection regret 是两个非负且可加的项；固定 action
后，action-prefix effect 与 visual-evidence effect 也可严格相加。但后一对是可正可负的
causal effects，不是 regret。要定义非负的 evidence-use regret，必须额外指定一个理想
continuation 反事实，而它不能从 direct、fixed-prefix counterfactual 和 fixed-prefix real
三个已观测结果中识别。即使补齐 intervention 数据，这一 causal effect 分解也与近期一手
工作直接碰撞。因此在数据生成、算力估计和 GPU 前关闭。

## 形式化

对一个状态，记 direct utility 为 `d`，各候选 action 的真实 utility 为 `r_a`，policy 的
call decision 为 `z`，调用时选中 `a_pi`。令 `r_star=max_a r_a`，oracle utility 为
`max(d,r_star)`。定义：

- stop regret：oracle 与“保留 policy 的 stop/call 决定、但调用时给最优 action”的差；
- selection regret：仅在调用时为 `r_star-r_(a_pi)`。

两项都非负，并严格满足
`total regret = stop regret + selection regret`。三组数值例覆盖：正确调用但选错 action、
本应停止却调用且选错 action、本应调用却停止；残差均为 0。

固定某一 action prefix 后，记 counterfactual/no-op observation utility 为 `c_a`、真实
observation utility 为 `r_a`，则

`r_a-d = (c_a-d) + (r_a-c_a)`。

第一项是 action-prefix effect，第二项是 visual-evidence effect。数值例
`d=0.8,c=0.9,r=0.6` 得到 `+0.1` 与 `-0.3`，总 effect 为 `-0.2`。因此两项虽严格可加，
但可为负，不能同时称作非负 regret components。

## 不可识别性

真正的 evidence-use regret 需要定义 `u_ideal(a)-r_a`。在完全相同的已观测三元组
`(d,c,r)=(0.8,0.9,0.6)` 下，两个潜在世界可分别令 `u_ideal=0.6` 或 `1.0`；二者与现有
观测均相容，却给出 `0` 与 `0.4` 两个不同 regret。因此理想 continuation 不是现有
intervention contract 的可识别量。

把同一 prompt 多采样并取 best-of-k 也不能定义稳定 oracle。若单次正确率为 `0.6`，
best-of-`1/2/4/8` 的期望分别为 `0.6/0.84/0.9744/0.99934464`，estimand 会随研究者选择
的重复数机械上升。除非预先固定 k 并把它明确称为 privileged best-of-k ceiling，否则
不能把它解释为模型“应当如何利用证据”的内在 regret。

## 一手文献碰撞

- [The Illusion of Visual Tool-Use](https://arxiv.org/abs/2608.06270) 已给出
  `(I,Q,T,O,Y)` causal graph，显式分离 observation-mediated path 与 action-induced
  shortcut，并在固定 prefix/action 下用 real/counterfactual observation 的差定义 Visual
  Evidence Gain；还在 policy、trajectory、step 三个层级做 intervention 和 policy-gain
  decomposition。N2 的 prefix/evidence effect 不是独立命题。
- [GapSight](https://arxiv.org/abs/2608.21762) 已从 global-only 与 candidate crop-augmented
  answer loss/margin 的差构造 preserve/review、utility 与 crop targets，并在三个 backbone、
  六个 benchmark 上评估。N2 的 stop/action utility bank 也不能单独构成新方法。
- [ToolVision](https://arxiv.org/abs/2608.08907) 已用 student-scale stepwise evidence gain
  筛 SFT trajectory，并在 RL 前比较 frozen learner 的 with/without-tool benefit；把上述
  effects 用作训练信号也需要进一步证明不同于现有 capability-aligned supervision。

The Illusion 明确把受控 credit-assignment training study 列为后续方向，但这不等于任何
credit loss 都自动新颖；ToolVision 与其他 process-aware work 已覆盖相邻空间。

## 机器 gate 与复现

- 基线/实现 commit：`0cc20ab7235b1be4c0af4fbe6d264c854f8cecaf`；
- N1 输入 SHA-256：
  `d17bb8eec9bf0f5cce89105d43c0a676b134ce0779b9f799a4c02903ae3d62c7`；
- N2 机器报告 SHA-256：
  `60b398454f6a495c4fbcb337a0c1eae075cc1536ea09f2f78b2f0a2c2ac99404`；
- module/runner/test SHA-256：
  `d5b94e5dcdabe432cd8ac96fd580bc418c22c99f5e7608ad109814ab3b4eacd2` /
  `27ecaf3ee0992784dcfc830070abac31189eddda72622936e3c7f4cc364285fa` /
  `e89a71ad6eb7ad67be9ac7cd46ba6febcea27ce4e014787543a31c960cef3295`；
- 七项 gate 中四项通过；失败项为 ideal continuation 可识别、best-of-k 对重复数不变、
  与注册一手文献不同；
- 12 个 targeted tests、三文件 mypy、compileall、Black check、deterministic byte compare、
  N1 hash 正/负 gate、JSON 断言、凭证扫描与 `git diff --check` 通过。

复现命令：

`PYTHONPATH=.:src python scripts/audit_causal_regret_decomposition.py --n1-report artifacts/docvqa-train-factorized-v2/ops/n1-existing-sibling-bank-inventory-v1.json --expected-n1-sha256 d17bb8eec9bf0f5cce89105d43c0a676b134ce0779b9f799a4c02903ae3d62c7 --output artifacts/docvqa-train-factorized-v2/ops/n2-causal-regret-decomposition-audit-v1.json`

本项 CPU-only；无模型加载、Slurm、GPU、optimizer、checkpoint 或 protected split。
报告显式冻结 `authorized_new_gpu_jobs=0`、`authorized_new_checkpoints=0`。由于科学 gate
已失败，没有继续做 augmentation GPU-hour/storage 估算。

## 下一步边界

关闭 causal-regret benchmark/decomposition 作为当前主贡献，不以改名、增加 K 或增加
UG-grid 行数重开。下一步只允许先审计一个具备公开权重、合法许可和真实 parser-valid
tool support 的强初始化是否存在，以判断 The Illusion 所提出的 controlled training study
在本项目是否技术可测；这只是 baseline gate，不自动恢复 H5 新颖性。任何训练前还必须
单独证明 signed same-prefix action credit 相对 ToolVision/TACO/CodeVision 的不可约区别。
