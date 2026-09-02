# 实验记录

更新时间：2026-09-02 12:49（Asia/Hong_Kong）

本文件记录当前决策链中的关键实验。更早的完整协议、哈希与结果保存在
`artifacts/docvqa-train-factorized-v2/ops/` 及各实验产物目录。

## E-20260902-01：Raw-attention outcome-free 特征抽取

- 假设：Qwen baseline forward 的 question-to-image attention 能提供比已有
  embedding ranker 更可靠的 crop localization。
- Commit：feature revision `2020b423f7daa6e8b9a942a02308137136bba548`。
- 数据：InfographicVQA official-train；rollouts SHA-256
  `9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e`；
  source features SHA-256
  `d0508726a50b4c1e54778392d08329b242a680fc13292cac1ebec8b42a175300`。
- 配置：Qwen2.5-VL-7B pinned revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`，final 4 layers，mean heads，
  question query tokens，四个 UG boxes，checkpoint 512。
- 命令：`scripts/submit_infographicvqa_attention_where_h800.sh`。
- 环境/资源：Job `203257`，2×H800，两轮 source-disjoint shards，离线模型缓存，
  凭证和 proxy 清除，全状态邮件。
- 结果：COMPLETED；23,946 decisions / 2,204 sources / 4,406 images；merged
  feature SHA-256
  `009cdbfa1761f9b53b791a50f70b7e98bdf275eec8743d8bcaf078a52ded8ce8`；
  audit SHA-256
  `27ba5df9d45f9837f685d64589e32740238de6ff0ce46ce54ce6a1ac21a1d471`。
- 日志：`slurm-infovqa-attention-where-203257.out` 及
  `attention-where-v1/feature-shards/*/attention.log`。
- 结论/下一步：特征完整且无 outcome；进入冻结 train gate。

## E-20260902-02：Raw-attention train gate 首次评估中断

- 假设与设置：同 E-20260902-01 的冻结 gate；20,000 whole-source bootstrap，
  `lambda=0.05`，call rates 0.5/1/2/5/10%。
- Commit：评估启动时基于当时 clean revision；Job `203262`。
- 资源：debug partition，1×RTX 4090 预留但 evaluator 隐藏 GPU，4 CPU，64 GiB。
- 结果：在科学 decision 产生前中断。Frozen comparator 的
  `action_selection_regret` 仅有约 `2e-19` 的末位浮点差异。
- 根因：不同 reduction/order 的 machine-scale float noise；不是数据、策略或
  指标变化。
- 修复：commit `5264d25`；有限 float 使用 `rel_tol=1e-12`、
  `abs_tol=1e-15`，结构、类型、整数、布尔仍严格一致，非有限值拒绝。
- 日志/协议：`slurm-infovqa-attention-where-eval-203262.out`；
  `infographicvqa-attention-where-evaluation-float-recovery-v1.md`。
- 下一步：原设置原输入恢复评估，不改变科学 gate。

## E-20260902-03：Raw-attention train gate 恢复评估

- 假设：raw attention action 在 entropy call set 上可达到正 utility，并优于
  fixed/random/old-DECAR/relative-where。
- Commit：`5264d25e5cbd176dbd6597a74ba10e475e35b77a`。
- 数据/配置/种子：同 E-20260902-02；bootstrap seed `20260917`，20,000
  resamples，2,204 sources。
- 命令：`scripts/submit_infographicvqa_attention_where_evaluation.sh`。
- 环境/资源：Job `203276`，CPU evaluator，RTX 4090 预留并隐藏，4 CPU，64 GiB；
  runtime `00:07:08`，exit `0:0`，全状态邮件。
- 结果：`attention_where_train_not_supported`。所有 utility 为负；1% 点最接近
  零（`-0.000041`），5% 点 `-0.000410`，CI
  `[-0.002438, 0.001681]`。5%/10% 对四个 deployable where baselines 的
  paired lower endpoints 全为正。
- 产物：evaluation SHA-256
  `5c8bced0fdad0a4f7c3ad0dca8bf8cf31d40be4c9d2318c6b42ea72d065366ee`；
  complete SHA-256
  `ea38fb7adb024a1c96a6ec160d921687affb3ac0222aecba3f5d422728a4cbf5`。
- 日志：`slurm-infovqa-attention-where-eval-203276.out`；结果审计
  `infographicvqa-attention-where-train-result-job-203276-v1.md`。
- 结论/下一步：where 有信号但 entropy stopping 失败；禁止 calibration，转向
  fixed-action stop 因子化诊断与最后的文献 where 强基线。

## E-20260902-04：ViCrop/LASER literature attention 抽取

- 假设：固定 ViCrop relative attention 或 LASER contrastive attention 比 raw
  final-layer pooling 提供更强的 where localization。
- Commit：feature revision `940ee8603f8b84bb7e107be4ecbd21cf9698d2b8`。
- 数据：同 E-20260902-01 的四个 source-disjoint shards 与固定哈希。
- 配置：ViCrop Qwen layer 22、final token、mean heads、query/generic ratio；
  LASER all-head query/no-query contrast、动态 layer；ENCORE layer 0/1 entropy
  仅描述；每 decision 三次 prefill；checkpoint 256；无 candidate execution。
- 随机性：确定性 extraction；无科学阈值搜索。
- 命令：`scripts/submit_infographicvqa_literature_attention_h800.sh`。
- 环境/资源：Job `203273`，2×H800，16 CPU，192 GiB，8 小时，两轮 shards，
  离线 cache，全状态邮件，支持 resume。
- 当前状态：RUNNING；12:47 快照为 wave 1，墙钟 52 分钟。
- 日志：`slurm-infovqa-lit-attn-203273.out` 与
  `literature-attention-where-v1/feature-shards/shard-{0,1}/attention.log`。
- 失败前例：Job `203270` 在模型加载前因 shard-2 输入哈希抄写错误失败，无科学
  输出；commit `940ee86` 修复并加入回归测试。
- 下一步：完成四 shards、merge/no-leak audit，再提交 97.5% Bonferroni gate。

## E-20260902-05：Fixed-action stop-factorization 诊断

- 假设：固定 raw-attention action 后仍有足够 positive-net stop ceiling；简单
  attention confidence 可能比 entropy 更能识别值得调用的状态。
- Commit：`91a0359a5bafdd086b22cba153077177438952d0`。
- 数据：E-20260902-03 已打开的 official-train outcomes 与完全相同的 raw feature、
  source order 和 bootstrap indices；所有输入哈希写入冻结诊断协议。
- 配置：固定 `argmax(question_region_attention)` action；比较 entropy stop、
  max-attention stop、top-two margin stop、at-most-budget privileged stop oracle；
  另报告 unrestricted fixed-action 和 task-action positive-net ceilings；call rates
  0.5/1/2/5/10%，`lambda=0.05`，20,000 bootstrap，seed 继承 frozen indices。
- Smoke/验证：synthetic core 单测、runner/Slurm contract、raw point reproduction；
  共 11 项相关测试通过，mypy 2 files 通过，shell `bash -n` 与
  `git diff --check` 通过。
- 命令：`scripts/submit_infographicvqa_attention_stop_factorization.sh`。
- 环境/资源：Job `203290`，RTX 4090 预留但隐藏，4 CPU，64 GiB，45 分钟，
  全状态邮件。
- 结果：COMPLETED；runtime `00:18:09`，exit `0:0`。Raw action 有 1,023 个
  positive-net states / 483 sources。Unrestricted fixed-action privileged ceiling 的
  source-balanced utility 为 `0.0213175`，95% CI
  `[0.0184472, 0.0244436]`；full task-action ceiling 为 `0.0338476`，95% CI
  `[0.0301742, 0.0377351]`。Attention max/margin 在所有注册 call rates 均不如
  entropy；5% utility 分别为 `-0.002330` / `-0.002906` / `-0.000410`。
- 产物：diagnostic SHA-256
  `f07eddb658444cd11ab67a62b53143c90ebf81a07026f00c7bba1411a3ad8e1a`；
  complete SHA-256
  `0160654dd9173192409b434728c3a654c76a275dd55220e6ecd6ab74d50ef068`；
  execution SHA-256
  `03cfc69868333a9613c4a0e65fd01d20cda4763b970e1e9ec7c9ce4627b584c9`。
- 日志：`slurm-infovqa-stop-diag-203290.out`。
- 解释边界：post-hoc / privileged ceiling，不可作正式选择或成功声明。
- 结论/下一步：fixed action 有大量 stopping headroom，但简单 attention confidence
  失败。冻结一个单独的 whole-source OOF，低容量 signed-value stop 候选；
  不继续搜索 max/margin 变体。
