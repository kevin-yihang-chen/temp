# 实验记录

更新时间：2026-09-02 22:00（Asia/Hong_Kong）

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
- 结果：COMPLETED；23,946 decisions / 2,204 sources / 4,406 images，四个
  source-disjoint shards 完整，所有特征 outcome-free，validation/test 未读。
  Merged feature SHA-256
  `ffec54e5c48ee9711bccde13a53f9ee4c9e6b85a2453eadcbe8ddde3236bec02`；
  feature audit SHA-256
  `53775f83b9a0231c0104b1b0fe69fedab6d1ffbeecaf0c28d5696c7bd0bfca9b`。
- 执行：2026-09-02 11:56:03--17:00:07 HKT；exit `0:0`；extraction
  18,121 秒、merge 78 秒、零 restart。
- 日志：`slurm-infovqa-lit-attn-203273.out` 与
  `literature-attention-where-v1/feature-shards/shard-{0,1,2,3}/attention.log`。
- 失败前例：Job `203270` 在模型加载前因 shard-2 输入哈希抄写错误失败，无科学
  输出；commit `940ee86` 修复并加入回归测试。
- 下一步：已提交冻结 97.5% Bonferroni evaluator（见 E-20260902-07）。

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

## E-20260902-06：Fixed-action signed-value stop OOF

- 假设：固定 raw-attention action 后，一个低容量 source-held-out signed-value
  classifier 能比 entropy 更好地识别 positive-net 调用尾部。
- 协议 commit：`e0f95d2ff0ed01b0343530989d1a2b52242ada3d`；协议 SHA-256
  `32e6080399153b91f1584d068590826fb105451c52c0a8944f0ce62ba6dac74a`。
- 实现 commit：`0683526db78c13cdced9eda237b833e9614f54c5`。
- 数据：同 E-20260902-05 的 official-train raw features/outcomes；绑定哈希见协议；
  validation/test/reserve 保持封存。
- 配置/种子：固定 raw attention argmax action；80 维 pre-action 特征；L2
  logistic `C=0.01`，无 class weight，按绝对 net utility 加权并使每个
  source 总权重相等；5 个 whole-source OOF folds，seed `20260918`。唯一
  primary 为 2% / 479 calls；20,000 次冻结 whole-source bootstrap。
- 验证９ 项相关单测通过；mypy 2 files 通过；Black check、shell
  `bash -n` 和 `git diff --check` 通过。
- 真实输入 smoke：23,946 decisions / 2,204 sources；80 维全部有限；
  1,023 positive-net / 22,923 negative-net states；五折 train/held-out source
  overlap 均为 0。`fit_performed=false`、`policy_metrics_computed=false`；
  elapsed `00:58.67`，峰值 RSS `3,678,368 KiB`。
- 命令：本地 runner `--smoke-only`；完整执行由
  `scripts/submit_infographicvqa_attention_signed_stop_oof.sh` 提交。
- 环境/资源：smoke 在 CPU 运行；完整任务预留 RTX 4090 但隐藏 GPU，
  4 CPU，64 GiB，45 分钟，全状态邮件。
- 执行：Slurm Job `203330` 于 13:06:16 开始、13:13:16 完成，
  runtime `00:07:00`，exit `0:0`，queue wait 11 秒。提交 commit
  `7b5f5ea2500cd49ad101c3dd11422f32d8e5bb98`，全状态邮件已由
  Slurm 合同确认。
- 结果：`fixed_action_signed_stop_train_not_supported`。2% primary 为 479
  calls；candidate utility `-0.0000626`，95% CI
  `[-0.0007393, 0.0006553]`；entropy utility `-0.0005847`。Candidate-minus-
  entropy `+0.0005221`，paired CI `[-0.0003039, 0.0014439]`。Positive-net
  calls 为 90 vs 77，precision `18.79%` vs `16.08%`；因此仅 precision
  条款通过，utility 过零与 paired improvement 两条失败。
- 次要诊断：0.5% 点 candidate utility 为 `+0.0001598`，但 CI
  `[-0.0001800, 0.0005725]` 跨零，且 paired lower endpoint 为
  `-0.0000158`；不允许它挽救 primary 失败。10% 点 candidate utility 显著
  为负。
- 审计：所有 folds 均 source overlap 0，模型 6--7 iterations 收敛，OOF
  coverage、matched calls、finite scores、输入哈希与无泄漏条款全部通过。
- 产物：report SHA-256
  `aa5de1fa1d9891d8425d192e7ed03782c003491d28c435dcf22abc69711e51ad`；
  model SHA-256
  `a053a47c5914d96423906abdd2d09500d3e2e193bb66826436a3149c0290be5e`；
  scores SHA-256
  `9bf4ad6a895864811427e9c37aeadf4844a8b5345165babb545db7fc9cc5f945`；
  complete SHA-256
  `47e5cf8eb5ae89ed3834042492122844ebf236058e9b093efd2b2b7fc7b1d62a`；
  execution SHA-256
  `aff57cd082b644a957ebd3a45442e636f463ecd94a4aaaca939234811924c7c4`。
- 日志：`slurm-infovqa-signed-stop-203330.out`。
- 解释/下一步：存在弱排序信号，但净 utility 和统计证据不足。按
  冻结协议停止此模型族；不事后改 C、特征、权重、seed、classifier
  family 或 primary call rate。等待 literature where 强基线决定下一主路线。

## E-20260902-07：ViCrop/LASER literature-attention where gate

- 假设：文献 ViCrop relative attention 或 LASER contrastive attention 能在相同
  entropy call set 上产生正 utility，并在 multiplicity correction 后至少不劣于
  raw attention 与其他注册 where comparator。
- 协议 SHA-256：
  `a86c4327a5e7ea8f5787b95883240149835e52a603266715900b5fddf8d682b1`；
  blind audit SHA-256：
  `1731fe8cf14568bb92ec8878477fa1f47dbb102f06953e84490afaa356cd7993`。
- Commit：evaluator revision `96508310366e5327c80094c3016a67561ec882c9`；
  feature revision `940ee8603f8b84bb7e107be4ecbd21cf9698d2b8`；model revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`。
- 数据：E-20260902-04 的完整 outcome-free features；复用 20,000 次 frozen
  whole-source bootstrap、2,204 sources、seed `20260917`；validation/test/reserve
  未打开。
- 配置：ViCrop 与 LASER 两个注册候选；call rates 0.5/1/2/5/10%；相同 entropy
  state identities；`lambda=0.05`；raw attention 作为第五个 deployable
  comparator；Bonferroni-corrected central 97.5% intervals。
- 命令：`scripts/submit_infographicvqa_literature_attention_evaluation.sh`。
- 环境/资源：Job `203340`；CPU evaluator，RTX 4090 仅预留并隐藏，4 CPU，
  64 GiB；runtime `00:09:45`，exit `0:0`，零 restart，全状态邮件。
- 结果：`literature_attention_where_train_not_supported`。LASER 在
  0.5/1/2/5/10% 的 utility 分别为 `-0.000039/-0.000082/-0.000476/
  -0.000373/-0.002808`；ViCrop 分别为 `-0.000067/-0.000194/-0.000653/
  -0.000833/-0.002981`。所有 corrected lower endpoints 不大于零；所有点
  `qualified=false`。LASER 在 5% 显著超过四个旧 where baseline，但对 raw
  attention 的 paired 97.5% CI `[-0.002145, 0.002075]` 跨零。
- Localization：LASER 全状态 exact NLL-teacher / task-oracle agreement / helpful
  rescue 为 `33.60% / 24.14% / 72.15%`；ViCrop 为
  `33.18% / 20.28% / 70.00%`。ENCORE layer-0/1 entropy 与 helpful-state
  Spearman 仅 `0.0177/-0.0144`。
- 产物：evaluation SHA-256
  `560b47edfa6cf2465d40e4138a4e5b5133898a2437af864a6bd32f1599d264ee`；
  decision SHA-256
  `0e6c092e05d7a26da61ba77a54cba76f674599ab4f5e9b486740f027b9d58b91`；
  complete SHA-256
  `10c1f24c69c4deef611f1b3a74dc46fe05d9e1c2497af01639235a8026a3d61f`；
  execution SHA-256
  `f1f4a222f11db0dfb61e7c7aaa4a3d9fb44ed7590249b45b805170d080dffc6d`。
- 日志/审计：`slurm-infovqa-lit-attn-eval-203340.out`；
  `infographicvqa-literature-attention-where-result-job-203340-v1.md`。
- 结论/下一步：关闭当前 attention-localization/simple-confidence family；不调
  layer、head、ratio、threshold、call rate 或旧 classifier family。下一步必须
  引入新的 pre-action 信息来源/action proposer，或转为论文级 empirical audit。

## E-20260902-08：Answer-conditioned evidence outcome-free feasibility gate

- 假设：复用 baseline answer generation 的 token hidden states，并与原图区域
  embeddings 计算 evidence consistency，可为 stopping 引入旧 80 维 features 没有
  的合法 pre-tool 信息，同时形成足够独立的新方法。
- 审计基线 commit：`c1f080a`；本项没有 feature/model/result commit。
- 数据：不读任何 sibling outcome、ground truth、validation/test/reserve；只审计
  代码接口、固定环境类型与 primary literature。
- 配置/检查：`Qwen25VLBackend.infer` 的 generation contract；Transformers
  `5.4.0` 的 `GenerateDecoderOnlyOutput` annotations；现有 semantic extractor 的
  question/global/region outputs；answer/hidden-state/grounding 相关一手文献。
- 命令：`rg`/`sed` 静态审计；
  `/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python -c ...` 只打印
  Transformers 版本与 generation output 字段。
- 环境/资源：CPU-only 静态检查；无模型加载、无 GPU、无 Slurm job、无新 artifact
  outcome；validation/test/reserve 继续封存。
- 结果：工程 feasibility 通过，但顶会 novelty gate 失败。ContextualLens、LRP、
  VRP、V-Loop 与 grounding-signature work 已覆盖 hidden-state/attention probe、
  answer-conditioned verification 与视觉 grounding 的核心组合。
- 决定：`answer_conditioned_evidence_candidate_rejected_before_experiment`。不实现、
  不拟合、不提交 GPU；不能把本项描述为实验负结果。
- 审计：
  `artifacts/docvqa-train-factorized-v2/ops/infographicvqa-answer-conditioned-evidence-feasibility-and-collision-audit-20260902-v1.md`。
- 下一步：只审计 same-prefix signed counterfactual visual-action credit 的
  novelty/upstream feasibility；generic hidden-state probe、group-DRO/IRM 或 threshold
  变体关闭。

## E-20260902-09：VTool counterfactual credit upstream static feasibility gate

- 假设：在 pinned VTool-R1 `training-v2` 上可以实现 action-local signed credit，
  且不需在静态阶段读取新 outcomes、安装 GPU 环境或提交 Slurm。
- Pinned upstream：`d2aa28353ec10c7f91b39f502925003a81d6982d`；Apache-2.0；
  本地只读浅克隆约 25 MiB，worktree clean。
- 数据/配置：无数据、模型或 outcome；只审计 agent loop、trajectory mask、GRPO
  advantage、checkpoint 与 3B recipe/dependency surface。
- 环境/资源：CPU-only `rg`/`sed`/git 静态检查；无 dependency install、无 GPU、
  无 Slurm job；validation/test/reserve 继续封存。
- 结果：`upstream_static_feasibility_supported_with_dependency_and_credit_path_blockers`。
  当前 tool/action chunk 在解析后整段 `response_mask=0`，observation 也为 0；GRPO
  仅把 outcome scalar 广播到单一 response mask，因此现状下 action credit 梯度为
  零。需要独立 `action_mask`/`answer_mask` 与 token-local advantage estimator。
- 环境 blocker：Docker 使用 PyTorch `2.10.0`、CUDA `12.9.1`、vLLM `0.17.0`；
  `setup.py` 却要求 `vllm<=0.12.0`，`requirements.txt` 还留有 `0.8.4` 注释。
  未确定权威 image/digest 前不安装环境。
- 恢复/成本：checkpoint/resume 入口存在，但 paired RNG/sampler 精确恢复和额外
  continuation cost 尚需 smoke 测量。
- 审计：
  `artifacts/docvqa-train-factorized-v2/ops/vtool-counterfactual-credit-upstream-feasibility-audit-20260902-v1.md`。
- 下一步：写唯一 matched-control protocol 和纯 synthetic sign/mask tests；在
  dependency、mask、antisymmetry、resume 与 novelty gate 全通过前不授权训练。

## E-20260902-10：Same-prefix counterfactual action-credit G0

- 假设：在不加载模型/数据和不修改 upstream 的条件下，可以把 signed net-utility
  pair、token roles、token-local advantage 与 shuffled control 写成可审计纯函数，
  并由 synthetic tests 排除 cost/arm-swap、mask 泄漏和 provenance 错配。
- Protocol：
  `artifacts/docvqa-train-factorized-v2/ops/vtool-counterfactual-action-credit-protocol-20260902-v1.md`；
  SHA-256 `5b85bc7381e8028f2ae07f423c2f7b165ebd3a8f28fafb4ceac1bf6043b5b1e0`。
- 实现 commit：`56b990c767973a8a23060d63293db8657254b35d`；core SHA-256
  `b5adcd4300d3fcf1df6efb7003ae6de0ae6179727e1b2e9cc0b5f2c1f2c55b5a`；
  tests SHA-256
  `7b651236ff6d59455f7d6242542cfc4d248e20a487b971d846a8413bcb063a7c`。
- 数据/种子：无数据、模型、outcome 或随机采样。Synthetic fixtures 固定 seed `17`
  字段，但没有 stochastic test；validation/test/reserve 未打开。
- 冻结定义：
  `A_visual=(Y_f-0.05*C_f)-(Y_c-0.05*C_c)`；primary `beta=1.0`，不 center/
  normalize；outcome 只给 answer tokens，action credit 只给 action tokens；
  observation/padding 为零。Pair 必须绑定 prefix/action/target/policy/decoding/scorer
  SHA-256 与 continuation seed。
- 对照修正：额外 counterfactual continuation 使“同 steps/rollouts/GPU-hours”无法由
  单一 baseline 同时满足。协议改为 paired zero/shuffled 的 exact-compute control，
  加 outcome-only 的 step/trajectory-matched 与独立 GPU-hour-matched 两种比较。
- 命令：`python -m pytest -q tests/test_counterfactual_action_credit.py
  tests/test_vtool_adapter.py`；`python -m pytest -q`；`python -m mypy
  src/beyond_entropy/counterfactual_action_credit.py
  tests/test_counterfactual_action_credit.py`；`python -m compileall -q ...`。
- 验证：17 项新 G0 tests 与 7 项 VTool adapter 回归通过；完整仓库收集 509 项，
  全量 pytest 为 479 passed、30 项依赖/资源相关预期 skip、exit 0；mypy 2 files 与
  compileall 通过，`git diff --check` 通过。
- 格式工具事件：Black `24.8.0` CLI 在输出文件检查结果后不退出，在 repo 与
  `/tmp` 均可复现；精确根因未证明。安全中断后改用同版本 in-process
  formatter/check，两个文件均 clean。未把 CLI 无输出误记为通过。
- 环境/资源：本地 CPU-only；无网络、无 dependency install、无 GPU、无 Slurm、
  无邮件事件。18:19 HKT 队列为空，GPU quota 剩余 179,656 分钟。
- 结果：`counterfactual_action_credit_g0_passed`。这只证明 schema/arithmetic 与协议
  不变量，不能证明 upstream 集成、训练稳定性、新颖性或性能。
- 下一步：只审计 public Refocus_Chart train metadata/source identity 和 vLLM 0.17
  权威 image/digest，冻结 train-only/data/environment amendment；明确禁止 FP8 script
  默认 `test.parquet`。G1 前不提交 GPU。

## E-20260902-11：Refocus train lineage、license 与 vLLM runtime gate

- 假设：在不下载 image columns、不加载模型且不碰 protected split 内容的条件下，
  可以确认 Refocus official train 的 schema/row lineage，并固定可复现 vLLM 0.17
  image identity，从而决定是否授权 G1。
- 实现 commit：`91a5cb438c9503dee5f0337d5bc118bcfef482bb`。Core/script/test
  SHA-256 分别为
  `71ad6776c0987ae79367b653b113b35f6435198f5180076c89eed6af6fbf55b9`、
  `353957c559b409a470527b015e863b33aecde368f01e7bae8a3708bfb59cda48`、
  `0bda624479a3f0cb9af9c07ade9c59e4782785f22c14c81f60a1df8f42fdc383`、
  `af4d71f16c61a56f3faa0853e51b62acba0faf8b385d3312ea55d6f7fc04a489`。
- 数据：`VTOOL/Refocus_Chart` revision
  `00f10ecc5b25d94fd66e14c3671af9fb0f088989`；train LFS digest
  `d7972ca232aa9c0646af387f7dffb987528b99b3d9693ccd58bbef0463f2d4e1`。
  Corrected runner 只做 train HTTP range read，并明确记录 LFS digest 未由全文件重新
  哈希；report `test_accessed=false`。
- Train 结果：14,344 rows/unique IDs，0 duplicate IDs；10,806 structural groups，
  3,538 duplicate-group rows，最大 group 59；exact question/Q+A/prompt duplicate rows
  分别为 141/23/141。Manifest SHA-256
  `a034f7fd1d3492950faa0d079a6b2da58e86742bee3ed3696a8c657b0c19677f`。
- Lineage：对 pinned original ChartQA root tree
  `044eabfc306abfe9340c5741f0093aefc5973d06` 只逐级遍历 train/png；18,317 original
  train PNG stems 中命中全部 14,344 Refocus IDs，missing=0。Lineage report SHA-256
  `15189ebb6128900c684ffc3cd7b07838a802a4eba88a42353b3ddb3b9dca0f6c`。
- Incident：初版 runner 曾访问 Refocus test 的非图像 metadata、question 与 ground
  truth；随后 one-off 命令还枚举 original ChartQA val/test PNG path IDs。没有读取
  test pixels、拟合、方法选择、GPU 或结果使用；报告已 quarantine，两个 test split
  不再作为 sealed formal。事件审计 SHA-256
  `51ce2bfe23dd09ca71c6184b9714330b348b6823dd611f836d92002a8382794a`。
- License：Refocus hosted dataset repository 无 card/license；VTool code 的 Apache-2.0
  不能自动覆盖 dataset，original ChartQA 的 GPL-3.0 也不能补写派生发布条款。因此
  full train download/training 未授权。
- Environment：官方 `verlai/verl:vllm017.latest` 已解析为 immutable linux/amd64
  digest `sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6`，
  compressed 14,356,458,058 bytes；但本地无 vLLM/Ray、容器工具或可执行 bundle，
  Slurm `JobContainerType=(null)`，model/judge path 不存在，home 仅余约 50 GiB。
- 命令：`scripts/audit_refocus_chart_metadata.py`；
  `scripts/audit_refocus_chart_lineage.py`；`python -m pytest -ra`；`python -m mypy ...`；
  `python -m compileall -q ...`；Black 24.8.0 in-process check；`git diff --check`。
- 验证：新 audit tests 10 项通过；完整仓库 519 项中 489 passed、30 个依赖/资源
  related expected skip；mypy 4 files、compileall、Black in-process 与 diff check
  全部通过。无随机 seed、模型 outcome、GPU、Slurm 或邮件事件。
- 结果：
  `g1_not_authorized_pending_dataset_license_pixel_identity_and_runtime`。这不是 H5 的
  性能负结果；它证明当前直接训练会缺少许可、pixel certificate 与可复现 runtime。
- 下一步：先许可或 original-data regeneration，再做 train pixel grouping；同时在
  有足够 scratch 的位置建立 pinned runtime，完成 import-only 后才冻结 G1 amendment。

## E-20260902-12：范围纠偏与 action-credit pre-GPU integration

- 范围纠偏：E-11 之后继续要求与 VTool 的 pixel、thought 或内部实现等价属于研究
  scope drift；这些条件不检验 H5，也不是论文贡献。VTool 此后只作为 Apache-2.0
  可运行 RL 骨架与 outcome-only comparator，不再进行 equivalence 审计。
- 数据：唯一训练来源改为 Apache-2.0 `ReFocus/ReFocus_Data` official train，revision
  `6af42739216fd58047121bb51dba683277cfdfe3`；三个 shard 的 SHA-256 固定在
  `configs/refocus_official_train_v1.json`。旧 derivative metadata 对照仅作一次性转换
  证据；不读取 ReFocus test、ChartQA validation/test 或 reserve。
- 实现：新增 upstream-shaped token-local adapter、paired agent overlay 与 pinned
  upstream 两文件最小 patch。原 outcome advantage 只进入 answer tokens，signed
  counterfactual credit 只进入 action tokens，observation/padding 为零；upstream
  answer-only mask、pair mismatch 或 scorer failure 均 fail closed。
- Scorer：冻结为本地确定性的 ChartQA relaxed match 与 answer extractor，取消外部
  LLM judge，避免 timeout/服务差异进入 credit。
- 环境：隔离 conda env `beyond-entropy-vtool-g1`；Python 3.10、torch `2.9.0+cu128`、
  transformers `4.57.6`、vLLM `0.12.0`、Ray `2.58.0`、TensorDict `0.10.0`；editable
  `verl` 指向 pinned runtime worktree。`pip check` 无 broken requirements。
- 验证：真实 PyTorch autograd smoke 中 signed arm 的 action-token gradients 非零，
  zero control 严格为零，observation/padding gradients 为零，answer gradients 在两组
  相同。environment audit 的 required imports、versions、pinned commit、最小 patch 和
  import origin 全部通过；reference clone 保持 clean。
- 资源：没有 Slurm job、GPU model load、rollout 或 optimizer step，因此没有计算状态
  邮件事件。修改均保留本地，未 push GitHub。
- 结果：`vtool_action_credit_g1_import_gate_passed`，但这只是 pre-GPU engineering
  evidence，不是方法成功。G1 仍需 official-train converter 单行 smoke、paired agent
  fake-server contract 和 model-load/显存 preflight。
- 下一步：只完成上述与核心假设直接相关的三个 preflight，再冻结 outcome-only、
  paired-zero、paired-shuffled、paired-signed 配置并决定是否提交最多 2-step G1；不再
  为 VTool 等价性投入时间或算力。

## E-20260902-13：Official-train converter 与 paired-agent contract

- 假设：在不读取 protected split、不使用 teacher thought/edited image/focus box、
  不加载模型权重的条件下，可以把 official train 转成真实 verl multimodal row，并
  证明 paired agent 的两臂只改变视觉 observation，正确导出 signed credit 与 token
  roles。
- 实现 commit：`b212dc844a25afff228c31eb47b16cd63007fc97`。该 commit 包含范围纠偏后的
  official-train converter、paired overlay、token-local adapter、四臂冻结配置与
  model-load preflight；没有 push GitHub。
- 数据：`ReFocus/ReFocus_Data` Apache-2.0 revision
  `6af42739216fd58047121bb51dba683277cfdfe3`；三个 official-train shard 的完整
  SHA-256 再次匹配 pin。Metadata pass 只读 structural/question/answer 列，selected
  image pass 只读原始 `image`；policy input 明确排除 answer、thoughts、edited image
  与 `focus_areas_bbox`。
- Split/配置：seed `refocus-official-g1-group-split-20260902-v1`；全 train 为 14,344
  rows/10,806 groups。冻结 64 train groups/72 rows 与 32 curve-eval groups/33 rows，
  group overlap=0。Paired/outcome-only 两份数据除 `agent_name` 外逐字段相同；train
  shared-content SHA-256 `06263c1e...566c27e`，curve-eval
  `15fd8ae6...23ffd93a`。四臂、seed、两步资源和 stop rules 固定在
  `configs/vtool_action_credit_g1_v1.json`。
- 数据产物：paired train/curve SHA-256 为 `3a5be076...ced46` /
  `28a2cd3f...b8a60`；outcome-only 为 `617924ee...44d1` /
  `cac6ae74...f6d67`；row manifests 在两 family 完全相同。
- 真实 processor smoke：单行 dataset SHA-256 `0de5b142...66199`；pinned
  Transformers `4.57.6` fast Qwen processor 解码 1 张原图、0 video、966 prompt
  tokens，pixel tensor shape `[2320,1176]`；所有 9 项合同通过，不加载权重。
- Paired fake-server：rescue/harm/tool-failure/direct credit 分别为
  `+0.95/-1.05/-0.05/0`；tool trajectories response mask 固定为
  `[1,1,0,0,1,1]`。两 continuation 的 prompt IDs、sampling config 与 seed 相同，
  factual success 只改变第二张图；observation token mismatch 与缺失 trajectory
  identity 均 fail closed。报告 SHA-256 `8dedebf8...72564`。
- 诊断事件：converter 首次直接调用因主包未在 `PYTHONPATH` 而在 import 前失败，
  以 `PYTHONPATH=src` 修正；无数据输出或科学选择变化。Fake smoke 首版在
  `asyncio.run` 关闭真实 thread-pool executor 时挂起；栈显示 case 已完成、worker
  线程空闲。改用只用于 fake decode 的 inline resolved Future 后稳定退出；不是 agent
  rollout 或 vLLM 死锁。随后一项 harm fixture 的预期 answer IDs 断言写错，修正测试
  fixture 后得到冻结的 `-1.05`。
- 验证：清除无关 equivalence tests 并加入 model/runtime/data/config binding contract
  后，完整仓库 `507 passed, 33 skipped`；skip 均为 base env 可选 Torch/资源项；
  mypy 11 files、compileall、Black in-process、JSON/shell
  syntax、`pip check` 与
  `git diff --check` 通过。隔离 runtime import gate 复验通过，环境报告 SHA-256
  `fb90ce60...94718`。
- 静态检查事件：扩大 mypy 覆盖后，首次发现 import origin、credit mode、PIL pixel
  与 dataclass kwargs 的局部类型歧义；逐项收紧类型后，目标测试与 10-file mypy
  通过。改动不改变 paired rollout 数值；复跑 fake-server/environment 报告后 SHA-256
  分别仍为 `8dedebf8...72564` 与 `fb90ce60...94718`。
- 环境/资源：CPU processor/fake-server；无模型权重、GPU、Slurm 或邮件事件。实时
  GPU quota 222,000 总分钟、42,327 已用、179,673 剩余（2,994.55 GPU-hours）；
  本轮实时查询时队列为空。
- 结果：`refocus_g1_dataset_and_paired_agent_contract_passed`。这是 pre-GPU correctness
  evidence，不是方法性能证据。G1 仍未授权。
- 下一步：提交已绑定 code/data/model/runtime、1×H800、30 分钟上限、
  `--mail-type=ALL` 的 vLLM model-load/单条 first-turn generation smoke；通过后才允许
  4×H800、最多 2-step G1。

## E-20260902-14：单卡 H800 vLLM model-load/真实首轮 generation gate

- 假设：冻结的 Qwen2.5-VL-3B、vLLM/verl runtime、official-train row 与视觉 prompt
  可以在本集群单卡 H800 上真实加载并完成一次非空、可解析的首轮 generation，且显存
  足以进入后续有界 G1。
- 代码 revision：`67822c14bd086929b64ec65803e7e8e9afa1d833`；核心实现 commit
  `b212dc844a25afff228c31eb47b16cd63007fc97`。没有 push GitHub。
- 数据/模型：单行 official-train Parquet SHA-256
  `0de5b1421c765724e77432f2d176e33c2af6d6bc27652ca4e9d5393306e66199`；模型 revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`，两块权重完整 SHA-256 与冻结配置匹配。
  Worker 同时验证 exact config、runtime commit 与 patch SHA-256；protected split 未访问。
- 资源/通知：Slurm Job `205784`，partition `q-h800`，1×NVIDIA H800、8 CPU、64 GiB、
  30 分钟上限、`--mail-type=ALL`。提交 `21:56:31`，开始 `21:56:32`，结束
  `21:58:46`，运行 `00:02:14`，零 restart，`COMPLETED`，`ExitCode=0:0`。
- 结果：`vtool_vllm_model_load_smoke_passed`。GPU 总显存 85,017,493,504 bytes；engine
  load `64.0501s`，模型权重加载报告 7.1562 GiB；生成前后剩余显存约 47.30/47.12 GB。
  966 prompt tokens 生成 12 completion tokens 用时 `19.2087s`，输出可由 parser 识别为
  `NOTOOL` direct answer。无 optimizer step。
- 产物：report SHA-256
  `1a67365b1fc1fbf76ad84c7954cdeeeb73f1fd1eeb52dd7d94971dd06469009a`；log SHA-256
  `abee46d422ed1b132bb90c736e16e7ba73929f48405bfb883dd8a06747e7be0d`。
- 诊断：日志仅有程序退出时未显式 `destroy_process_group()` 的 NCCL warning；作业
  正常退出且产物完整。该 warning 不影响本项 decision，但后续训练必须检查正常 shutdown。
- 解释边界：本项证明真实模型/视觉输入/显存/runtime 可执行，不证明 action credit
  有效，也没有覆盖真实 tool-call 与 paired continuation。G1 只获得工程授权，尚无
  方法性能证据。
- 下一步：核对 pinned upstream 的唯一训练入口，冻结 4×H800、最多 2-step 的
  paired-signed launch 并先做 Hydra dry-run；只有真实 call/pair/optimizer gate 通过，
  才运行同 revision 的 matched controls。
