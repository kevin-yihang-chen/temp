# 实验记录

更新时间：2026-09-03 20:07（Asia/Hong_Kong）

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

## E-20260902-15：Paired-signed G1 最终运行时与入口预检

- 假设：在不提交 GPU、不执行 optimizer step、不读取 protected split 的前提下，
  可以把完整 frozen train 输入、真实 Hydra 配置、训练资源和 action-credit 控制变量
  绑定为一个 fail-closed 入口，从而安全授权第一次 paired-signed G1。
- 范围：VTool 只作为 pinned RL 运行底座与 outcome-only comparator；没有继续审计
  pixel、thought 或内部实现等价性。核心问题仍是 signed action credit 相对
  paired-zero、paired-shuffled 与 outcome-only 的因果增益。
- 运行时数据审计：72 行 paired train 全部经过 frozen `RLHFDataset`、Qwen fast
  processor 与真实 image tensor 路径；64 个 structural groups，row ID 唯一，1 image/
  row、0 video，metadata 仅含 structural fields。Prompt tokens 为 min 438、median 931、
  p95 1,657、max 1,914，小于冻结上限 4,096；dataset 与 row manifest SHA-256 均精确
  匹配配置。报告 SHA-256
  `e468c36719ee3e303ff84a98f6feac1b88ebee9f0fb2731be2a129204671c1ed`。
- Action-credit 修正：协议规定 batch 少于两个 valid tool pairs 时 shuffled action loss
  全部置零并计入 `action_credit/shuffled_batch_skipped`；此前单 tool batch 会错误保留
  自身 credit，现已修复并由真实 autograd smoke 覆盖。Signed action gradient 非零，
  zero/shuffled-skip 为零，observation/padding 为零；报告 SHA-256
  `3fa21b9e5343348cf4619066c08b08049b63fe96fa11a9151d3723fd5238d99b`。
- Batch 修正：upstream 的 `actor_ppo_mini_batch_size` 单位是 prompt count，随后乘
  rollout `n` 再按 world size 切分。原值 32 会形成错误语义；现冻结为 train batch 8、
  `n=4`、4 GPUs、mini-batch 8，即每卡 8 trajectories 且每个 optimizer step 只有一个
  完整 local mini-batch。该修正发生在任何 G1 结果产生之前。
- Scorer/资源：四臂共用本地确定性 ChartQA relaxed scorer，不调用外部 LLM judge。
  Signed worker 仅允许 4×H800、48 CPU、384 GiB、2 小时、2 optimizer steps；step 2
  保存唯一 resumable checkpoint，rollout steps 1/2 必须同时存在。状态邮件冻结为
  `--mail-user=yihangc@connect.hku.hk --mail-type=ALL`。
- Hydra 事件：首个结构化 dry-run 在解析新字段 `rollout.limit_images` 时因 Hydra struct
  要求 `+` 前缀而失败；修正 override 后没有改变科学配置。最终 v9 对 59 个关键值逐项
  检查，全部为真，decision 为
  `vtool_action_credit_g1_hydra_dry_run_passed`。Launch manifest SHA-256
  `dff12612518927291f28ef356508961128b127657417b7b355b7394edf8263da`；resolved config
  SHA-256 `5a5d354d69058a5f931d1f0a6d15bbb5c459c7e1f024e791cf582cf0b5387b3b`。
- 安全边界：launcher 绑定 code revision/worktree、数据/模型/runtime/protocol/preflight
  hashes；execute 要求 clean repo、Slurm allocation 与恰好 4 张可见 H800。Worker 要求
  至少 32 GiB persistent free space，临时 Ray 数据写 `/dev/shm`；不会读取 validation/
  test/reserve，也不会把 credential 传入训练环境。
- 提交前诊断：首次调用 submit wrapper 在执行 `sbatch` 前退出，因为脚本收紧的 PATH
  不含 base-conda 的 `jq`；没有创建 Slurm job、没有 GPU 或邮件事件。Submit/worker
  现均固定并验证 `/userhome/cs3/yihangc/anaconda3/bin/jq`，避免节点环境继承差异。
- 验证：完整仓库 `514 passed, 34 skipped`；skip 均为 base env 缺少可选 Torch/资源
  项，隔离训练环境的真实 Torch gradient smoke 另行通过全部 8 项检查。10-file mypy
  （忽略第三方无 stub imports）、Python compileall、Black in-process、shell/JSON syntax、
  隔离环境 `pip check` 与 `git diff --check` 全部通过。
- 资源：本项只有 CPU validation/hash 与 Hydra config rendering；无 GPU、无 Slurm、
  无 optimizer step，因此没有新的计算状态邮件。磁盘清理未执行。
- 结果：`paired_signed_g1_ready_for_final_regression_and_submission`。这仍是工程授权，
  不是方法成功或性能证据。
- 下一步：完成全量回归、本地 commit（不 push GitHub）、实时复核 quota/queue/disk，
  然后只提交 paired-signed G1。真实 tool-call rate 低于 1%、pair mismatch 非零、judge
  failure 非零或训练/checkpoint 不稳定时按冻结规则停止；通过后才运行三组 controls。

## E-20260902-16：G1 排队取消与 counterfactual 结果可观测性修复

- 假设：正式 G1 不仅要完成训练，还必须在不可变产物中逐 trajectory 保存足以重建
  factual/no-op pair、signed credit、cost-adjusted utility、harmful/rescue rate 和冻结
  stop rules 的字段；否则即使 optimizer 成功也不能回答 H5。
- 发现：Job `205870` 提交后仍处于 `PENDING (Resources)`。只读检查 pinned upstream
  `_log_rollout_data` 发现 JSONL 只写 `reward_extra_infos_dict`；paired agent 当时的
  `reward_extra_info` 仅有 `acc`。完整 pair 虽存在 batch `non_tensor_batch` 并参与
  token-local training，但不会进入 `1.jsonl/2.jsonl`。这会使真实结果缺少
  counterfactual score、action credit、pair provenance、tool success 与 harmful-call
  证据，属于核心实验可观测性缺口。
- 作业状态：在任何 GPU 分配前执行取消。Slurm 最终确认 Job `205870` 为
  `CANCELLED`、`RunTime=00:00:00`、`Restarts=0`、`Reason=Resources`；没有模型加载、
  rollout、optimizer step 或 GPU-hours。任务配置 `--mail-type=ALL`，取消属于状态通知
  覆盖范围。没有重复提交。
- 实现：paired agent 现在把稳定 schema 的 audit payload 序列化为单个 JSON 字符串，
  作为 `reward_extra_info.vtool_action_credit_audit_json` 导出。使用字符串而非多个
  bool/int numpy scalars，是因为 upstream `json.dumps` 不能直接序列化 `np.bool_`/
  `np.int64`。Payload 对 direct/tool 两类都保持相同字段集合，含 factual 与
  counterfactual response、trajectory ID、三个 role token counts、signed credit、pair
  validity、完整 pair、tool success 和 counterfactual generation time。
- 自动分析：新增 `scripts/analyze_vtool_action_credit_g1.py`。它要求 step 1/2 各 32
  rows，逐行验证 JSON schema、唯一 trajectory、score/acc、direct/tool role contract，
  并用 `CounterfactualActionPair.from_dict` 重新验证 shared provenance 与序列化 credit。
  报告 task score、realized cost-adjusted utility、tool-call/tool-success、mean signed
  credit、harmful/rescue/no-effect rate 与 per-step summaries；tool-call `<1%` 产生正式
  stop decision，pair/scorer/产物 mismatch 则 fail closed。
- Worker：训练完成后必须同时存在两份 rollout dump 与 step-2 resumable checkpoint，
  随后自动生成 `rollout-analysis.json`。只有 artifact contract 完整，worker 才正常
  完成；科学上允许 `smoke_gate_passed` 或预注册的 `stop_rule_triggered` 两种终态，负
  结果不会被误报为基础设施成功，也不会隐藏。
- Fake-server v2：rescue `+0.95`、harm `-1.05`、tool failure `-0.05`、direct `0` 与
  全部原有 shared-prefix/mask/fail-closed 检查继续通过，并新增
  `rollout_audit_payload_exported=true`。报告 SHA-256
  `e24e48dfeee138d77bf6d50919043a5b685c108e7aa83c707b250b27b7df5ab5`；protected split
  未访问、模型权重未加载。
- Analyzer 合成回归：每 step 32 rows、每 step 1 个 valid rescue call 时，64 rows 的
  tool-call rate 为 3.125%，decision 为 `paired_signed_g1_smoke_gate_passed`，mean signed
  credit `0.95`、pair/judge failure 均为 0；全 direct 时 tool-call rate 为 0，按冻结
  阈值得到 `paired_signed_g1_stop_rule_triggered`，程序正常保存负 gate 结果。
- Hydra：更新后的 v11 仍有 59 项 scientific/resource resolved-config checks 全真，
  训练超参数与 E-15 相同；launch manifest SHA-256
  `cd2674577f56bcd3db7f6301c07171fa1019c6b4e80eb955296c0d80cca65d9a`，resolved config
  SHA-256 `70089aca355c1f0952dc454044d95fbeaa2f62b7eb3bed65bb481a4b8f4cfaf1`。
- 验证：完整仓库 `515 passed, 34 skipped`；skip 均为 base env 缺少可选 Torch/资源
  项。12-file mypy、Python compileall、Black in-process、shell/JSON syntax、隔离环境
  `pip check` 与 `git diff --check` 全部通过。隔离 runtime 的 rescue/harm/tool-failure/
  direct 四种 fake-server 分支再次真实运行通过。另直接调用 pinned upstream
  `RayPPOTrainer._dump_generations`，确认 numpy string array 可生成包含 `acc` 与
  `vtool_action_credit_audit_json` 的 JSONL，并可无损反序列化；decision 为
  `upstream_rollout_json_string_export_passed`。
- 解释边界：这是让 H5 结果可审计的必要工程修复，不是 VTool 内部等价性审计，也不
  构成方法效果证据。没有读取 validation/test/reserve，没有改 prompt、seed、采样、
  reward、credit、threshold 或 baseline。
- 结果：`paired_signed_g1_observability_gate_passed`。
- 下一步：本地 commit（不 push GitHub），实时复核 quota/queue/disk，再以新 revision
  只提交 paired-signed G1，并监控同一 job 到终态。

## E-20260903-01：Paired-signed G1 worker jq 前置断言失败

- 假设：commit `8c0540dceb2304ef9dfa8cc8993a560b9dbb269f` 上已通过的 dataset、
  runtime、Hydra 与 rollout 可观测性合同足以让 paired-signed G1 进入真实模型加载、
  paired rollout 和最多 2 个 optimizer steps。
- 提交：Job `205902`，4×H800、48 CPU、384 GiB、2 小时上限，
  `--mail-user=yihangc@connect.hku.hk --mail-type=ALL`；submit time
  `2026-09-02 23:33:39`。科学配置、数据、模型、seed、prompt、reward、stop rule 与
  E-20260902-16 完全相同。
- 结果：`worker_preflight_failed_no_scientific_result`。Job 于 2026-09-03 00:56 HKT
  获得资源后同秒退出。worker status 的 `decision=failed`、`exit_code=2`、
  `scientific_decision=not_available`，开始/结束 epoch 均为 `1788368160`。没有创建
  `g1-runs/.../job-205902`，因此没有模型加载、rollout、optimizer step、checkpoint 或
  task/utility/harmful-call 指标；不能把本项解释为 H5 失败。
- 根因：worker 使用 `.checks | all(.[] == true)` 检查 JSON object 的所有值。
  jq 的单参数 `all(condition)` 已逐元素迭代输入，随后 condition 中的 `.[]` 又尝试迭代
  布尔值，故对全真 report 也报 `Cannot iterate over boolean (true)`、exit 5。原命令
  已在相同 frozen 72 行 runtime audit report 上稳定复现。
- 修复：runtime dataset audit 与最终 rollout analysis 两处断言均改为二参数 generator
  形式 `.checks | all(.[]; . == true)`。相同 report 返回 true/exit 0；合成 object
  全真返回 0、含假返回 1。新增 pytest 直接调用冻结 jq 并要求 worker 中恰有两处正确
  predicate，避免仅做 shell 静态字符串检查。
- 产物：worker status SHA-256
  `27559a29b9f5370ad1443f72789d9de9951ed1e2f78e890288c7dc5971c4e01d`；Slurm log
  SHA-256 `0b21a496e2a30cf4092b9f117e0c523c3bcd6817d988e96a56bf36a2a4519387`。
  `sacct` 因集群 accounting storage 禁用而不可用；09:40 HKT live quota 为 222,000
  GPU 分钟总额、42,242 已用、179,758 剩余。
- 验证：完整仓库 `516 passed, 34 skipped`；skip 均为 base env 缺少可选 Torch/资源项。
  12-file mypy、Python compileall、Black in-process、全部 shell/JSON syntax、隔离环境
  `pip check`、实际 72 行 report jq、credential scan 与 `git diff --check` 全部通过。
  目标 preflight test 为 `7 passed`。修复 commit
  `8c0f6c010a4dfeb1bf01d955054da2287691896e` 的 clean-revision Hydra v13 进一步以
  59/59 checks 全真通过；launch manifest SHA-256
  `29e24dabdbf4986c981a737297227d4f6b2ee365ce45c6e9a9b9aff7aca28fe8`，resolved config
  SHA-256 `cda71307a9c6ebc7fd634114bf5cac76664723d0e124c83351f3d8a12680ac43`。
- 下一步：提交本次 docs-only 记录后在最终 HEAD 复跑同一 Hydra gate，实时复核队列、
  磁盘与预算后只重提 paired-signed G1；不运行 controls、不改变科学配置。

## E-20260903-02：G1 FSDP actor FlashAttention 启动失败与 SDPA 运行时修复

- 假设：jq 修复后的 clean revision `1fd694c96c6a5bb6800bb2a1a6049b1b6c251cc3`
  能让 paired-signed G1 进入真实 actor 权重加载、paired rollout 与最多 2 个 optimizer
  steps。
- 提交：Job `206170`，4×H800、48 CPU、384 GiB、2 小时上限，
  `--mail-user=yihangc@connect.hku.hk --mail-type=ALL`；2026-09-03 09:55:15 HKT
  开始，09:57:18 结束。科学配置、数据、模型、seed、prompt、reward、stop rules 与
  E-20260903-01 后的冻结设置相同。
- 结果：`fsdp_actor_attention_backend_failed_no_scientific_result`。worker status
  `decision=failed`、`exit_code=1`、`scientific_decision=not_available`。Ray core 已
  启动并建立四个 actor rank，但四个 rank 均在 Hugging Face
  `from_pretrained` 的 attention backend 检查处报
  `FlashAttention2 ... flash_attn seems to be not installed`。输出目录只有空 Hydra
  log、launch manifest 与 execution report，没有 rollout、optimizer step、checkpoint、
  task score、utility 或 harmful-call 指标，不能解释为 H5 正/负结果。
- 根因区分：冻结环境为 torch `2.9.0+cu128`、Transformers `4.57.6`，没有
  `flash_attn`；pinned `fsdp_workers.py` 在未提供 override 时显式默认
  `flash_attention_2`。进一步源码审计发现 `use_remove_padding=true` 会在加载后把
  Qwen attention forward 替换为自定义 FlashAttention 实现，因此只设 SDPA 会把失败
  推迟到第一次 actor forward，并不构成完整修复。Ray dashboard MetricsHead 曾报
  EOF，但 Ray core 随后成功启动，且四个 rank 有一致、更下游的 ImportError，因此
  dashboard 事件不是本次根因。
- 修复：commit `22b89a5ca872356c621203f7bb724042c846a091` 在所有四个实验臂共同的
  model config 上冻结 `attn_implementation=sdpa` 且
  `use_remove_padding=false`；launcher、Hydra resolved-config audit 与 worker 均
  fail closed 拒绝漂移。没有安装/编译新的 `flash_attn`，避免在仅约 40 GiB 可用磁盘
  上引入未冻结 wheel/source build。该更改发生在任何 G1 科学结果前，且对 signed、
  zero、shuffled、outcome-only 对称，不改变方法、数据、reward、seed、prompt 或指标。
- 新最小 gate：新增单 H800、30 分钟上限的 HF actor-load smoke。它复用 official-train
  单行真实图片，按 FSDP worker 相同的 AutoModel class dispatch 加载权重、应用 verl
  多模态 monkey patch，并完成一次 SDPA actor forward；不执行 optimizer。worker
  绑定 code/config/data/model/runtime/patch 哈希、拒绝覆盖产物、保存失败 status，并用
  `--mail-type=ALL` 覆盖全部状态通知。
- CPU/meta 证据：最终脚本在不加载权重时得到
  `vtool_hf_actor_meta_dispatch_passed`；actor class 为
  `Qwen2_5_VLForConditionalGeneration`，model/text/vision 三层 backend 均为 SDPA，
  原生 attention forward 保留，verl 多模态 model forward 已应用，全部 4 项 checks
  为真。报告 SHA-256
  `6e6afe772e058ca31e11b8f49eb6a8bc5196e57231260bf35ab71cf2151bbda5`；模型权重未
  加载、protected split 未访问。
- Hydra：dirty-worktree 诊断 gate 对新增两项在内的 61 项 resolved values 全真，
  launch manifest/resolved config SHA-256 为
  `75a55090da833e2bcf8946d3da95afe1212d044bba1961c49ea7ec83c89a573f` /
  `afceaecee891c736a37108b1fc2e1f022b8f419ac01d4116d9e239f7fac798bb`。该 gate 只
  验证 override 能被 Hydra 正确解析；最终 clean revision 必须重跑。
- 失败产物：Job `206170` worker status SHA-256
  `316e6038465b55c5d85d5c3572d857d5594bfdc3f3ae6283969fb05b9fdc27a0`；Slurm log
  SHA-256 `38ceb1882ac46cf34216b00fd57f9bca0095b787f0eb7b7850fb1c91235a179c`。
- 验证：完整仓库 `518 passed, 34 skipped`；15-file mypy、Python compileall、
  Black in-memory formatter、四个相关 shell syntax、JSON、隔离环境 `pip check`、
  credential scan 与 `git diff --check` 通过。一次检查命令误把 Python 文件传给
  `bash -n` 并如预期报 Python 语法；修正目标后全部实际 shell 通过，不是代码失败。
- 当前最佳结果：仍无 H5 性能结果；本项只把失败从不可执行默认 backend 收敛到一个
  可独立验证的 runtime amendment。
- 下一步：先在最终 clean commit 复跑 Hydra gate 并提交唯一 1×H800 actor-load/真实
  图片前向 smoke。只有权重加载、patch、forward、显存与 artifact checks 全部通过，
  才重提 4×H800 paired-signed G1；若失败，按新错误定位，不重复四卡作业。

## E-20260903-03：单 H800 HF actor SDPA 真实图片前向 gate

- 假设：冻结的 `attn_implementation=sdpa`、`use_remove_padding=false` 不仅能构造
  meta actor，还能在 H800 上加载完整 Qwen2.5-VL-3B 权重，应用与 FSDP actor 相同的
  AutoModel dispatch 和 verl multimodal monkey patch，并对真实图片输入完成有限值
  forward；若失败则不得再次提交四卡 G1。
- 提交：Job `206174`，clean revision
  `86a345a73c5df74eb7748fd402b1c46fc0a46bc9`，1×H800、8 CPU、64 GiB、30 分钟
  上限，`--mail-user=yihangc@connect.hku.hk --mail-type=ALL`。Slurm submit/start
  均为 2026-09-03 10:39:34 HKT，10:40:24 结束；终态 `COMPLETED`、
  `ExitCode=0:0`、`RunTime=00:00:50`、`Restarts=0`。Job `206173` 只是
  `sbatch --test-only` 的预测编号，不是运行任务。
- 输入/绑定：模型 revision `66285546d2b821cf421d4f5eb2576359d3770cd3`，两块
  权重与 runtime commit/patch hashes 均在 worker 重新验证；单行 official-train
  dataset SHA-256 `0de5b142...66199`，1 张原图、966 prompt tokens，无 video，未访问
  protected split。code/config/smoke/dataset hashes、runtime status 与唯一输出路径均
  fail closed。
- 结果：`vtool_hf_actor_gpu_forward_smoke_passed`，6/6 checks 全真。actor class 为
  `Qwen2_5_VLForConditionalGeneration`，model/text/vision 的 resolved attention
  backend 均为 SDPA；Qwen 原生 attention forward 未被替换，verl multimodal model
  forward 已应用。权重加载 3.1566 秒，真实图片 forward 0.8175 秒，logits shape
  `[1,966,151936]`，最后 token logits 全部有限。GPU 总显存 85,017,493,504 bytes，
  peak allocated 7,972,130,816 bytes（约 7.42 GiB）。未执行 optimizer step。
- 产物：report SHA-256
  `48e2f12f0fcb910d87761b950af26ba83836033b1b2a0e9149b7a4720a318742`；worker status
  `1bf6acdf2228b08a023d303c67d3689926a86d5c6472ab73eb73ee9c129c634b`；Slurm log
  `8905da36ff9b1ec0411359c05d4071fac55c2096d67d37dd903908bbdc67cb93`。
- 证据绑定：commit `0122689050a4bfc91df0a55dd154dc52d7fce83d` 把 report 路径/
  SHA-256 加入冻结 G1 配置；launcher 还要求 model/runtime/data/backend/remove-padding、
  smoke script hash、真实权重、H800、prompt/logits shape、forward time、显存与全部 checks
  一致，并把该 report hash 写入 launch manifest。篡改或缺失会在模型启动前失败。
- 验证：报告绑定后的目标测试通过；完整仓库 `518 passed, 34 skipped`；15-file mypy、
  compileall、Black in-memory formatter、相关 shell/JSON、隔离环境 `pip check`、
  credential scan 与 `git diff --check` 全部通过。
- 解释边界：本项证明 SDPA/no-remove-padding 可执行且显存宽裕，消除了 Job `206170`
  的已知 backend 阻塞；没有覆盖 FSDP2 sharding、Ray 多进程、torch compile、paired
  generation、action-credit optimizer 或 checkpoint，因此不是 H5 性能结果。
- 当前最佳结果：仍无 H5 task/utility/harmful-call 指标；工程证据从 meta dispatch
  前进到完整权重真实图片 forward。
- 下一步：在报告绑定后的最终 clean commit 复跑 Hydra gate，再实时复核资源并只提交
  4×H800、最多 2-step paired-signed G1。只有其真实 rollout、pair validity、tool-call
  rate、optimizer 与 checkpoint 通过冻结 gate，才启动 matched controls。

## E-20260903-04：四卡 G1 DataProto 分片失败与 exact runtime 修复

- 假设：actor SDPA 真实图片 smoke 通过后，clean revision
  `89978bddbb950d9aeebc416d969de2b602ce9a67` 能完成至少一个 paired rollout、
  token-local action-credit optimizer step，并保存可恢复 checkpoint。
- 提交：Job `206179`，4×H800、48 CPU、384 GiB、2 小时上限，
  `--mail-user=yihangc@connect.hku.hk --mail-type=ALL`。提交时间 10:47:54 HKT，
  10:53:38--10:57:06 在 `gpucluster-g1` 运行；Slurm 终态 `FAILED`、
  `ExitCode=1:0`、`RunTime=00:03:28`、`Restarts=0`，约占用 13.9 GPU-minutes。
- 新证据：四个 FSDP actor 均以 SDPA/no-remove-padding 加载 3.75B 模型，四个 vLLM
  engine 和 agent-loop worker 启动，trainer 显示 `Training Progress: 0/2` 并进入第一
  次 `_update_actor`。因此 Job `206170` 的 FlashAttention 阻塞已真实解除；Ray dashboard
  MetricsHead EOF 之后 core 正常启动，不是本次终止原因。
- 结果边界：没有 `rollouts/1.jsonl`、optimizer step、checkpoint 或 rollout-analysis；
  worker status 为 `failed`、`scientific_decision=not_available`。不能解释为 H5 正/负
  结果，也不能推断真实 tool-call rate、pair validity 或 credit 趋势。
- 根因：`inject_token_local_action_credit()` 新增
  `vtool_action_credit_donor_trajectory_id` 时使用 Python `list`。pinned verl
  `DataProto.chunk()` 在按 4 个 data-parallel rank dispatch 前要求每个
  `non_tensor_batch` value 都是 `numpy.ndarray`，因此在 actor optimizer 实际执行前
  触发 `AssertionError`。相同 runtime 的 2-row `DataProto.chunk(2)` 已稳定复现：其余
  6 个字段均为 ndarray，唯一 donor 字段为 list。
- 修复：commit `ea489f7c520880ab087af761a620b03f357b18e0` 把 donor IDs 改为
  object ndarray。新增 `smoke_vtool_action_credit_dataproto.py`，在冻结训练环境构造
  rescue/direct/harm/direct 四行 batch，执行真实 credit injection 和
  `DataProto.chunk(4)`，验证全部非张量字段、四个等分 shard、donor identity、action/
  answer masks、tool count 与无 CUDA 初始化共 7 项。submitter 在 `sbatch` 前执行；
  worker 在 Ray/模型启动前复查并把 JSON 写入日志，任一 false 均 fail closed。
- 复现与验证：修复前 exact runtime 命令稳定报同一 `AssertionError`；修复后生成四个
  size-1 shard，decision 为 `vtool_action_credit_dataproto_chunk_passed`、7/7 checks
  全真。全仓 `519 passed, 35 skipped`、目标回归、4-file mypy、compileall、Black
  formatter API、两份 shell syntax、credential scan 与 `git diff --check` 全部通过。
  protected split 未访问、模型权重未加载，科学数据、prompt、reward、seed、sampling、
  credit、threshold 与 controls 均未改变。
- 产物：worker status/log/launch manifest/execution SHA-256 分别为
  `9e9c21b0115c28239c71bcb390c078ae8631566a615c963f48cd983c5d42a07d`、
  `1afe38677d13a8330942876376d3c0e0f0c27c01c5d6795f37aba580937c7906`、
  `ddb406766190a5d51c339af12165e3cf4fb2dbcb525e19a99633bd93196594ff`、
  `e660444d63c112e24a0257d81c1783b31e08965aa7b8294d480ecbfc3855f1b2`。
- 当前最佳结果：H5 仍无性能指标；工程 gate 从单卡 actor forward 前进到真实四卡
  actor-update dispatch，失败点已由 exact CPU runtime contract 覆盖。
- 下一步：提交本条中文记录后，在最终 clean revision 重跑 Hydra resolved-config 与
  DataProto chunk gate，实时检查 quota/queue/disk，再以同一科学配置仅重提 signed G1。
  只有真实 rollout、pair/tool-call、optimizer/checkpoint gate 通过，才补齐并运行
  zero/shuffled/outcome-only controls；否则按预注册规则停止或仅修复新证据明确指向的
  工程错误。

## E-20260903-05：四卡 G1 checkpoint 写满磁盘与存储 gate 修复

- 假设：DataProto 类型修复后的 clean revision
  `83f6e53fe1fdd4d83e4d5cd1bae7a0ab00e02b80` 能完成两步 paired rollout、token-local
  optimizer、最终可恢复 checkpoint 与冻结 stop-rule analyzer。
- 提交：Job `206184`，4×H800、48 CPU、384 GiB、2 小时上限，
  `--mail-user=yihangc@connect.hku.hk --mail-type=ALL`；11:27:40--11:32:35 HKT，
  此前实时 Slurm 查询记录为 `FAILED`、`ExitCode=1:0`、`RunTime=00:04:55`、零
  restart。12:09 HKT 当前队列为空，quota helper 为 222,000 GPU 分钟总额、42,268
  已用、179,732 剩余；controller 此后已清除该短期 job record。
- 新证据：worker 内 exact DataProto smoke 7/7 checks 全真；四个 actor/vLLM/agent-loop
  启动并实际保存 `rollouts/1.jsonl`。32 行 schema、score、trajectory、direct/tool、
  pair 合同 10/10 全真，task score mean `0.53125`，pair/judge failure 均为 0。
- 单步风险：step 1 的工具调用为 `0/32`。临时单步 analyzer 会触发 `<1%` 条款，
  但冻结正式协议要求 step 1/2 共 64 行，当前必须保持
  `scientific_decision=not_available`。第二步至少一次调用会得到 aggregate 1.5625%，
  所以不得提前关闭，也不得把临时诊断当正式结果。
- 失败根因：worker 仅要求 32 GiB 空闲。checkpoint 写满盘前已生成四个
  3,755,065,659-byte model shards，以及四个 6,945,898,496 / 7,189,561,344 /
  7,062,945,792 / 7,148,797,952-byte optimizer shards，合计 43,367,466,220 bytes
  （约 40.39 GiB）；extra-state、dataloader state、tracker 尚未完成。upstream 在
  step-2 actor update 后先保存 checkpoint、再写当步 rollout，因此盘满同时导致
  `rollouts/2.jsonl` 和 analyzer 缺失。
- 产物：step-1 rollout SHA-256
  `a2118345d04633d235659812ba5319284131c3a5b782ef4575f576598ad9e75c`；Slurm log
  `7e88271055f5ead7b5ca641706f6099b81efebbd20ab23d6dd3aba376e4f6722`；launch manifest
  `f8e6383718bb74c12f618606314e8cfce7a1f0ec828f9ade3c99062aa45d894b`。没有完整
  checkpoint、step-2 rollout、execution、worker status 或正式 analysis。
- 清理：用户明确授权后永久删除不可恢复 checkpoint，并删除约 37 GiB 可重建的
  Hugging Face Arrow dataset cache；Qwen cache、Hub source blobs、正式 artifacts、
  rollout 与日志保留。可用空间恢复至约 77.1 GiB。
- 修复：资源合同新增 `minimum_free_persistent_disk_gib=64`；submitter 在 `sbatch`
  前、worker 在模型加载前分别从冻结 config 读取并校验同一数值。相对已知 40.39 GiB
  shards 留约 23.6 GiB 余量，不改变任何科学设置。
- 验证：目标测试通过；显式 `PYTHONPATH=.:src` 的全仓回归为 `519 passed, 35
  skipped`，零 failure/error；4-file mypy、compileall、Black formatter API、两份
  shell syntax、JSON、exact pinned-runtime DataProto 7/7、credential scan 与
  `git diff --check` 全部通过。首次在非登录 shell 直接运行 pytest 因仓库根目录未在
  import path 而出现 16 个 `scripts` collection errors；修正命令环境后同一完整 suite
  通过，不是代码回归。Black CLI 在 NFS 上完成输出后未退出，改用同版本 formatter
  API 验证内容，未改格式规则。
- 解释边界：这是存储/可观测性失败，不是 H5 的正或负结果；但 0/32 工具调用是必须
  保留的负面风险证据。每臂只保存一个 `global_step_2` checkpoint；若 signed 通过，
  后续三臂的独立 checkpoint 不能在当前盘无界累积。
- 下一步：完成目标/full regression、静态检查、clean commit 与最终 Hydra/DataProto
  gate；实时复核至少 64 GiB 空闲后只重提 signed arm。正式两步 stop 失败则关闭路线，
  通过才规划 matched controls 的存储与执行。

## E-20260903-06：四卡 paired-signed G1 正常完成但零 parser-valid 工具调用，正式停止

- 假设：存储 gate 修复后的 clean revision
  `9c6bdc46f60b31d57b11d7a5c95a4712eef5fd44` 能完成两步 paired rollout、optimizer、
  完整 checkpoint 和冻结分析；若初始 policy 提供至少 1% 工具动作支持，则可进一步
  检查 signed action credit 的学习信号。
- 提交：Job `206205`，4×H800、48 CPU、384 GiB、2 小时上限，配置
  `--mail-user=yihangc@connect.hku.hk --mail-type=ALL`。Slurm 在短期记录清除前观测为
  `COMPLETED`、`ExitCode=0:0`、零 restart，12:26:27--12:31:10 HKT；worker elapsed
  `261.8567s`。科学配置 SHA-256 为 `3f2b1438...ebb3`，未改 prompt、seed、sampling、
  reward、credit 或停止阈值。
- 执行证据：两步训练、两份 rollout、正式 analyzer 与唯一 `global_step_2` checkpoint
  全部生成。两步 task score 分别为 `0.5625` / `0.53125`，总体 score 与 realized
  cost-adjusted utility 均为 `0.546875`。两步 actor grad norm 分别为 `66.4326` /
  `9.9353`，说明普通 outcome GRPO 更新执行；但 action-credit tool trajectory count、
  applied credit 与 tool-call rate 两步均为 0。
- 正式结果：严格 parser 与运行时 audit 记录 tool call `0/64`、rate `0.0`，低于冻结
  `0.01`；机械 decision 为 `paired_signed_g1_stop_rule_triggered`，唯一 stop reason 为
  `tool_call_rate_below_frozen_threshold`。10/10 rollout checks 全真，pair mismatch 与
  judge failure 均为 0，protected split 未访问。由于没有 tool pairs，harmful/rescue/
  no-effect/tool-success 和 mean signed credit 不可定义。
- Checkpoint：4 model、4 optimizer、4 extra-state shards、`data.pt` 与 metadata 完整，
  file payload 45,077,408,354 bytes，磁盘约 42 GiB；`latest_checkpointed_iteration=2`。
  全部文件已重新读取并由
  `vtool-g1-signed-checkpoint-job-206205-v1.sha256` 绑定。只完成结构与哈希验证，未另启
  resume job，不声称 resume 已实际执行。
- 产物：launch/execution/analysis/status/log SHA-256 分别为 `8a667ef0...cf5`、
  `5f1b2809...11f3`、`d8c49508...6c21`、`22eb7f41...dd5be`、`f5def800...facec`；
  step-1/2 rollout 为 `04ba5634...e6f` / `4b07ae08...55b`。完整审计见
  `vtool-g1-signed-result-job-206205-v1.md`。
- 结论：当前 sampled on-policy H5 路线因零 parser-valid action support 无法激活其特有
  credit，G1 正式失败并停止。按预注册规则不运行 zero/shuffled/outcome-only controls，
  不调整 seed/prompt/temperature/threshold 追结果；这些 controls 在没有有效 action
  token 时不能区分 credit 因果效应。
- 资源：12:41 HKT 队列为空；222,000 GPU 分钟总额、42,284 已用、179,716 剩余；
  checkpoint 后持久盘可用 37,988,859,904 bytes（约 35.38 GiB）。
- 下一步：在不提交 GPU 的前提下审计能在零 on-policy action support 下产生训练信号的
  新 estimand/算法。简单 forced-call、tool bonus、SFT/curriculum 或 off-policy hints 已有
  强文献碰撞，不能作为投稿方法；只有新候选通过一手文献与实现 gate 才进入实验。

## E-20260903-07：G1 raw 工具意图与格式/API 合同事后诊断

- 问题：E-20260903-06 的正式 analyzer 把所有 `vtool_tool_attempted=false` 响应统称为
  direct，但这一分类是否等价于“模型没有工具意图”尚未检查。该区别决定下一步应处理
  exploration、格式合同还是方法 credit。
- 假设区分：H-A 为 64 条均是自然语言 direct；H-B 为存在裸工具调用且只缺代码围栏；
  H-C 为存在工具意图但函数参数/response protocol 也不合法。诊断预期只读已有 raw
  outputs，不执行工具、不修复输出、不改变 reward 或正式 decision。
- 实现：新增 `scripts/analyze_vtool_g1_intent_format.py` 与独立回归测试。诊断器绑定原
  official analysis、step 1/2 SHA-256、trajectory ID、raw output 与 audit payload；逐条
  检查 final/focus prefix、Python fence、AST、唯一 allowed focus call、`display`、真实
  `(image_1, [labels], columns_bbox/rows_bbox)` 签名、prompt label membership 与
  fence-only recoverability。
- 数据/范围：只读 Job `206205` 已冻结的 64 条 official-train-derived rollout；不读取
  Parquet image/answer、validation/test/reserve，不加载模型，不执行工具，无新 outcome。
- 结果：13/13 artifact/runtime checks 全真；48/64 raw outputs 为 `FINAL ANSWER`，16/64 为裸
  `focus_on_*` intent。Step 1/2 裸意图分别为 12/32（37.5%）与 4/32（12.5%）。其中
  15 条是 AST-valid single focus expression，1 条混入 final answer 而语法无效；15 条
  均缺 `display` 且真实三参数签名合法为 0。因此 `fence_only_repair_executable=0/16`。
- 决定：`g1_zero_parser_valid_support_with_malformed_bare_tool_intent`。H-A 与 H-B 被
  否定，H-C 获支持。正式 `paired_signed_g1_stop_rule_triggered` 保持不变；不能事后
  把 16 条算作 tool calls，也不能说 latent tool intent 为零。
- Prompt/runtime 根因证据：冻结 V1 prompt 只列函数名和变量，未提供签名/可执行模板；
  parser 无 ` ```python ` 时无条件 `NOTOOL`；运行时六个函数均要求 image、label list、
  bbox mapping 三参数。更严重的是，prompt 声称 `x_values_bbox/y_values_bbox` 可用，实际
  agent context 只注入 `columns_bbox/rows_bbox`；前者会 `NameError`。这支持 baseline
  action contract 不充分，但不证明模板修复后会产生有用调用。
- Tokenizer 诊断：同一冻结 tokenizer 下 `FINAL ANSWER:` 首 token 为 `98848`，六个
  focus 函数共用 first token `17414`；` ```python ` 为 `[73594,12669]`，`<tool_call>`
  为单 token `151657`。这证明 intent boundary 可观测，但不构成新方法。
- 产物：JSON SHA-256
  `df19920bd426e62d1d2152d85bf2afce596c7b111c247e5807e4a5ba17a44160`；完整审计
  `vtool-g1-format-contract-and-next-route-audit-20260903-v1.md`。原 analysis/rollout
  SHA-256 仍为 `d8c49508...6c21`、`04ba5634...e6f`、`4b07ae08...55b`。
- 额外纠正：Job `206184` 的 step 1 严格 tool call 仍为 `0/32`，但 raw 文本是 19 条
  final answer 与 13 条裸 focus intent；由于缺 step 2，它仍不是正式 gate。
- 文献 gate：ToolVision 与 Tool-RL collapse 已覆盖 capability-aligned SFT、benefit
  reward、format/control-token collapse；Tunable Tool-Call Rates 与 black-box logit
  bias 已覆盖 call propensity steering；LIRE/LiPO/ToolPrefer 已覆盖 offline/listwise/
  step-wise preference；GapSight 已覆盖 candidate crop loss-gap router。因此 prompt、
  forced-call、SFT curriculum、steering、普通 listwise reward 和 crop utility router
  只能作 baseline/comparator，不能作为新方法。
- 当前最佳结果：正式方法结果未改善；新证据把根因从“零 latent intent”收敛到“有 25%
  裸 intent、但零 syntax/API-valid support”。
- 下一步：B0 独立建立 exact typed-action V2 baseline，先做 renderer/parser round-trip
  与真实 fake executor；N0 只保留 action-boundary interventional objective 作为待审
  主方法候选。它必须证明不同于 whole-response LIRE/LiPO、ToolVision benefit reward、
  GapSight router，并在 zero-valid-support 下有可验证梯度；若退化为普通 full-information
  contextual-bandit/listwise loss，则 CPU gate 关闭。N0 通过前不提交 GPU。

## E-20260903-08：Typed-action V2 baseline 的 CPU 与真实 runtime 合同

- 假设：不修改冻结 V1/Job `206205` 的情况下，可以定义唯一、可 round-trip 的 typed
  action grammar，并让它在 pinned VTool context 中真实执行，从而先修复 how-to-call
  baseline，再讨论新方法。
- 实现 commit：`150803ac113008f9ad5555f00f743aa17df9746c`。新增
  `RefocusTypedAction`、canonical renderer 与 strict AST parser；V2 prompt 通过独立
  `build_typed_action_prompt()` 暴露，不接入旧 converter。V1 prompt SHA-256 仍为
  `d8e1b93a3635901c6a5afcbf618e255e4923b01b11001ea56ca31de9fefca24f`，未改变。
- Runtime 合同修正：pinned agent context 实际只注入 `image_1`、`columns_bbox`、
  `rows_bbox`、`display`。因此 canonical x/y action 分别固定为
  `focus_on_x_...(image_1, [labels], columns_bbox)` 与
  `focus_on_y_...(image_1, [labels], rows_bbox)`；不再使用 V1 误称可用、实际会
  `NameError` 的 `x_values_bbox/y_values_bbox`。
- 拒绝边界：strict parser 拒绝无完整 Python fence、缺 `display`、非唯一 expression、
  kwargs、错误 image/bbox、空/重复/非 literal labels、越界 labels、额外 print/import
  或同 response final answer。它不执行任意 model text；只有 renderer 生成的固定
  canonical code 在测试中进入 `exec`。
- 真实最小执行：在 pinned `refocus_tools.py` context 中分别构造 vertical-bar x-label
  draw 与 horizontal-bar y-label highlight，两条 renderer→parser→runtime→display
  round-trip 均得到唯一非空 PIL image。72 行冻结 G1 数据的 source/axis 结构只读审计为
  41 条 `chartqa_v_bar` 且仅 x labels、31 条 `chartqa_h_bar` 且仅 y labels，和 alias
  映射一致。该检查读取既有 official-train Parquet 的 `source/extra_info` 列；后者使
  已开放的 train answer 同时被载入内存，但 answer 未用于统计/选择。Image 列与任何
  protected split 均未读取。
- 代码/测试 SHA-256：typed core
  `ab60245ca2b55faeb63bd413d022efdd709453af145e4ebcef404478e0c3c737`；dataset/prompt
  module `293352da81c5685c9ac394e1e64736a7da33a56a4eb2ad57315b3e5049ae2970`；typed tests
  `fa6973d9632e563bc014a768cd888a5a5c5bd25f429d4f6c0fd8d34aef2f8e19`；V2 prompt
  `7bf336ccba8044b011e569ade23bb021890e276511fe6dac88a5f44234325679`。
- 验证：相关 21 tests 全部通过；全仓 JUnit 记录 567 tests、0 failure、0 error、35
  expected skips，即 532 passed。5-file mypy、compileall、Black `24.8.0` in-process
  check、diagnostic deterministic byte comparison、JSON semantic assertions、credential
  scan 与 `git diff --check` 通过。
- 环境/资源：CPU-only；无模型生成、GPU、Slurm 或邮件事件。原 checkpoint 保留，当前
  约 35.38 GiB 可用空间不影响本项。
- 结果：`typed_action_b0_cpu_contract_passed`。这只证明语法、静态安全边界和 pinned
  executor 兼容，不证明 V2 prompt 能诱导合法调用，更不证明 action credit 有效。
- 下一步：为 V2 创建独立版本/hash 的 official-train-only 单行数据并通过真实 Qwen
  processor + fake executor；随后才允许一次无 checkpoint 的 1×H800 first-response
  generation smoke。必须分别报告 intent/syntax/argument/execution rate，不得用总
  tool-call rate 掩盖失败层级。N0 新颖性 gate 与 B0 性能不能混为同一主张。

## E-20260903-09：Typed-action V2 独立单行数据、真实 Qwen processor 与 fake executor

- 假设：在完全保留冻结 V1/Job `206205` 的条件下，可以把 V2 exact grammar 接入一个
  独立的 official-train 单行数据版本，并同时通过真实 Qwen processor 与 pinned VTool
  executor；若失败，则不得用 GPU generation 判断模型的格式遵循能力。
- 实现 commit：`47fde3717ba5f8d9f2d3ec5a7ae725e0da94be5c`。Converter 新增显式
  `--prompt-version v2`，并 fail closed 限制为 `--smoke-one-row`、`b0_smoke` 和 outcome-only
  `vtool_agent`；默认 V1 converter 行为不变。新增 runtime smoke 只把 renderer 重新生成、
  strict parser 已验证的 canonical code 交给 executor，绝不执行 raw model text。
- 数据：Apache-2.0 `ReFocus/ReFocus_Data` official train revision
  `6af42739216fd58047121bb51dba683277cfdfe3`；三个 shard SHA-256 重新验证。V2 仍选中与
  V1 相同的唯一 row/structural group，row manifest SHA-256
  `9d078c0d...9edf6`，从而只改变 prompt/data schema。V2 Parquet SHA-256
  `2c6a6c9b0a2329199ca750ad6489d3b1fafdf17b91a106ab426c634510e8184c`；独立 converter
  report SHA-256 `35c36a25384a09276686adb73f7068721109f1cc52ae61537eeaf3c88b496e8b`。
- V1 不变性：在临时目录重新运行默认 converter 后，V1 单行 Parquet 仍为
  `0de5b1421c765724e77432f2d176e33c2af6d6bc27652ca4e9d5393306e66199`、61,297 bytes，
  与冻结 artifact 字节级一致；V2 不能用于重解释 G1。
- 真实 processor：隔离环境 `beyond-entropy-vtool-g1`、Qwen2.5-VL-3B revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3` 的 fast processor；1 张原图、0 video、975
  prompt tokens，pixel tensor shape `[2320,1176]`。其 pixel SHA-256
  `4ab27a1d...e2d52` 与 V1 processor smoke 相同，确认图像输入没有漂移。
- Fake executor：selected row 为 `chartqa_v_bar`，固定选择 x 轴首标签与 `draw` mode；
  renderer→strict parser→pinned `build_refocus_context`→`display` 得到唯一 PIL image，
  原图/输出图像 SHA-256 分别为 `423ee10e...1f93` / `82f35d03...2bee`。运行时文件 SHA-256
  `04fc8f92...4b037`；raw model text 没有执行，reward target 没有参与 action selection。
- 结果：`refocus_typed_action_b0_real_runtime_smoke_passed`，26/26 checks 全真；runtime
  report SHA-256 `c25e28a7f3d146d470b35118c33e14af4440d957a664aa619804ebc8f0cd001d`。
  没有加载模型权重、optimizer step、checkpoint、GPU、Slurm 或邮件事件，也没有访问
  validation/test/reserve。
- 可复现性：第二个临时输出目录重新转换/运行后，V2 Parquet SHA-256 完全相同，剔除
  路径与 converter-report 路径哈希后的 runtime invariant JSON 字节级相同。报告内记录
  converter/smoke/dataset/typed-action/runtime 的精确源码哈希。
- 验证：最终代码全仓 `534 passed, 35 skipped`，skip 均为 base env 缺少可选 Torch/资源；
  6-file mypy、compileall、Black 24.8 in-process、隔离环境 `pip check`、JSON 全真断言、
  credential scan 与 `git diff --check` 通过。首次并行检查中的 mypy duplicate-module 与
  Black `NothingChanged` exit 是验证命令调用问题，分别用 `--explicit-package-bases` 与
  正确异常处理复验通过，不是代码失败。
- 解释边界：本项证明数据版本、真实 processor、严格 grammar 与 executor 接口闭环，
  但动作由 deterministic renderer 构造，不证明 Qwen 会在 V2 prompt 下生成 parser-valid
  或有用的工具调用，也不是主方法/性能证据。
- 下一步：冻结唯一的 1×H800、无训练、无 checkpoint first-response generation smoke。
  预先定义并分别报告 tool intent、完整 Python fence、严格参数合法、parser-valid 与实际
  execution 五层指标；无论结果正负均不修改 V2 prompt/seed/temperature 来追结果。

## E-20260903-10：Typed-action V2 真实生成复制元变量，参数合同 gate 失败

- 假设：在 CPU/真实 processor/executor 合同已通过后，Qwen2.5-VL-3B 能从同一独立
  official-train 单行输入产生至少 2 个 tool intents，且 intent-conditional strict execution
  rate 不低于 80%。本项只评价 baseline correctness，不训练、不改写 G1、不作为 N0
  方法证据。
- 冻结协议：config commit `96cd1665c72913b971ebb7c5aaa87f2bee15bbc0`；唯一 16 个
  seeds `2026090300..2026090315`，temperature `0.7`、top-p `0.9`、top-k `-1`、
  max tokens `128`。输入为独立 V2 `b0_smoke` 一行 Parquet（SHA-256
  `2c6a6c9b...e8184c`），模型 revision `66285546...0cd3`；validation/test/reserve 保持
  封存。V2 prompt、种子和 gate 在结果读取后不允许调整。
- 运行：Slurm Job `206227`，1×H800、8 CPU、64 GiB、30 分钟上限，邮件
  `yihangc@connect.hku.hk`、`--mail-type=ALL`。14:08:12--14:09:33 HKT；终态
  `COMPLETED`、`ExitCode=0:0`、零 restart、运行 81 秒。模型加载 16.2395 秒，16 次生成
  20.3844 秒，模型权重约占 7.16 GiB。
- 分层结果：tool intent `11/16`（68.75%）；完整 Python fence 与 syntax-valid 均为
  `7/16`（43.75%，intent-conditional 63.64%）；argument-contract、strict parser 和
  execution 均为 `0/16`。21/21 input/provenance checks 全真，机械决定为
  `typed_action_b0_malformed_tool_intent`。
- 失败分解：5 条为 direct final answer；7 条是完整、syntax-valid fence，但都调用
  不存在的 `focus_on_x_values_with_MODE`；另 4 条有 intent 但没有完整 fence，也都使用
  同一 `_with_MODE`。因此 11/11 intents 逐字复制 prompt 的元变量，没有替换为
  `draw/highlight/mask`。7 个 fenced calls 中另有 3 个扩张为全部 labels，其中 1 个还把
  x-axis 函数与 `rows_bbox` 配对；这些是函数名失败之外的可见附加错误。
- 解释：V2 的直接失败机制不是“零工具意图”，而是把元变量嵌入所谓唯一合法模板，模型
  将其视为可复制代码。证据只否定当前 V2 prompt 的 reliable baseline 资格；不证明
  concrete-template、structured decoding 或更大模型必然成功，也不能事后修复输出计数。
- 安全/资源：raw model text 从未执行；optimizer step `0`、checkpoint `0`、reward target
  未使用、protected split 未访问。输出目录只有 20,843-byte `report.json`；当前队列为空。
  14:18 HKT quota 为 222,000 GPU 分钟总额、42,275 已用、179,725 剩余，磁盘可用
  37,983,617,024 bytes。
- 产物：report/status/log SHA-256 分别为
  `9c6c763e10b0e005b605331bc2e3570fb3d841b6b36fc63f362c5f3dd741e875` /
  `c4731fc7d95fc8ed9957a9745f20f30985b6476dd6e49d54f60b6208ca2b461e` /
  `eaea14cbfb0636728d6a0c0c6f8c6736c6fe8ee750d2c2a9c24db46f829286d8`；完整审计为
  `refocus-typed-action-b0-generation-result-job-206227-v1.md`。
- 当前最佳结果：项目仍无 deployable 正主结果；B0 新增一个有明确层级和即时机制的强
  baseline 负例。它不改变 Job `206205` 的 `paired_signed_g1_stop_rule_triggered`。
- 止损与下一步：V2 不改 prompt/seed 后重跑。优先完成 N0 的 zero-support gradient 与
  文献区分形式化；若确需 V3 强 baseline，必须先让六个 concrete 模板逐一 strict-parser
  通过，并预注册未用于 V1/V2 的 official-train structural group、新种子和一次性 gate。
  V3 只能是 baseline correction，不是论文新颖贡献。

## E-20260903-11：N0 action-boundary objective 零支持与新颖性 gate

- 问题：能否把有限 typed visual macro-actions 的 boundary probability 与完整
  same-prefix intervention outcomes 结合，在当前 policy 的 parser-valid support 为零时仍
  得到新颖、可计算的 action-learning gradient，同时不退化为 forced SFT、listwise
  preference、utility router 或已有 action value？
- 形式化：对 `ANSWER_NOW` 与全部 typed actions 定义归一化宏动作概率 `p_theta(a|s)` 和
  干预净效用 `U(s,a)`。直接目标 `sum_a p_theta(a|s)U(s,a)` 的精确 logit gradient 为
  `p_theta(a|s)[U(s,a)-E_p U]`；即使完整枚举 `U`，近零 support 仍使梯度同阶消失，
  exact numerical zero 时严格为零。
- 替代目标：以 `q_tau=softmax(U/tau)` 做 cross-entropy 时梯度为 `p_theta-q_tau`，确实
  绕过当前 support，但原因是 intervention outcomes 已成为离策略 listwise target；
  one-hot 极限是 best-action SFT，pairwise 极限是 DPO/ranking，连续版本是 AWR。单独
  回归 `Q(s,a)`/boundary head 则回到 full-information contextual bandit 和 GapSight 类
  utility router。
- 数值 gate：dependency-free 三动作环境固定 utilities `0/+0.95/-1.05`、near-zero logits
  `0/-20/-20`、underflow logits `0/-1000/-1000`、target temperature `0.25`。10/10 checks
  全真；beneficial action probability `2.061e-9`，直接 gradient `1.958e-9`；underflow
  下二者为 0。Utility target 为 beneficial 分配 `0.9778`，CE gradient 为 `-0.9778`。
  解析/finite-difference 最大误差为 `2.34e-18` / `1.51e-9`。
- 一手碰撞：ToolVision 已用 student-scale evidence gain 搜索和 frozen-policy paired MUT
  先建立 useful behavior support；The Illusion 已在 fixed prefix/action 下替换 observation
  定义 VEG；GapSight 已从 candidate loss gaps 学 pre-crop gate/utility/box；LIRE/LiPO/
  ToolPrefer 与 AWR 覆盖 multi-response listwise、step-wise preference 和 advantage-weighted
  off-policy regression；Tool-RL collapse 已系统比较 off-policy/hint/interleaved SFT。
- 决定：`action_boundary_candidate_reduces_to_existing_objective_families`。当前 N0 没有
  同时满足“零支持非零梯度”与“不是既有离策略监督/价值学习”的第三种目标；在主方法
  实现和 GPU 前关闭。Token-local action mask 是可复用工程，不足以构成新颖性。
- 产物：机器报告 SHA-256
  `c1bfd08a571cab4dc8d5f017e681434e6fd7caf364808e7e5ffbb85dc474e4f1`；module/runner/test
  SHA-256 为 `a4bbbf78...3a2d` / `186ebfed...d53` / `9e8f17c8...416a`；完整审计见
  `action-boundary-interventional-objective-novelty-gate-20260903-v1.md`。
- 验证：9 个 targeted tests 全部通过；3-file mypy 与 compileall 通过。首次 test 因
  finite-difference 绝对误差 `1.51e-9` 略高于不合理的 `1e-9` 阈值失败，阈值按该
  O(20) loss 的 double-precision central-difference 误差改为 `1e-8` 后，解析梯度本身
  未改变且复验通过。首次 mypy 命令未给 `MYPYPATH=src` 并暴露两个动态 payload 类型，
  改为直接读取 typed dataclass、收窄测试 kwargs 后复验通过。
- 资源/泄漏：CPU-only；无模型权重、Slurm、GPU、optimizer、checkpoint 或新 outcome；
  validation/test/reserve 未访问。
- 下一步：N1 只读盘点现有 sibling bank 是否能支持 stop regret、action-selection regret
  与 evidence-use regret 的三项可识别分解，以及多数据集/多 backbone/动作族规模是否
  足以形成区别于 The Illusion 和 GapSight 的顶会 benchmark/estimand。盘点失败则关闭，
  不先生成新数据再寻找主张。

## E-20260903-12：N1 现有 sibling-bank 三段 regret 可识别性盘点

- 假设：仓库中已有的大规模完整 sibling banks 同时支持 stop regret、注册 action bank
  内的 action-selection regret 和 fixed-prefix evidence-use regret，并具备多数据集、
  同数据集多 backbone、多动作族、source-level 推断与不可变复现元数据，足以形成区别于
  The Illusion/GapSight 的顶会 benchmark 起点。
- 实现 commit：`f7c944948942b28e4c4c2030b21138fe2930d436`。新增 dependency-free
  streaming inventory，不把约 30 万行一次性载入内存；逐 decision 检查 sibling/action/
  seed/identity，逐 row 检查模型 revision、intervention fields，并从 provenance 读取模型、
  proposer、manifest/rollout hash 和 code revision。
- 主数据：InfographicVQA-7B `23,946`、ScreenQA-3B `14,511`、DocVQA-3B `13,580`、
  TextVQA-3B `7,912` decisions；合计 `59,949` decisions、`299,745` rows、按数据集求和
  `12,214` sources。全部为 answer-now + 4 个 UG-grid ZOOM，完整性和不可变 provenance
  通过。
- 辅助诊断：ScreenQA-7B 为 `512` states，与 3B bank 精确重叠 512，但角色是
  opened-development diagnostic；ChartQA-3B `4,500` decisions 的 provenance 明确标作
  diagnostic，不计入主 gate。
- 结果：10 项 gate 通过 6 项。Stop regret 与 registered-bank action-selection regret
  可识别；evidence-use regret 不可识别，因为 `239,796` 条主 ZOOM 中完整保存 fixed action
  prefix、matched factual/counterfactual observation 与 controlled continuation 的记录为
  `0`。同数据集 multi-backbone main factor、多工具动作族和每状态多个 stochastic
  replicate 也失败。
- 细节：UG-grid bbox 随图像长宽比变化，不能误称四个固定坐标框；但 proposer/tool type
  仍只有一个 ZOOM family。所有主 state 都只有 `replicate-000`；主数据中的 3B/7B 与
  数据集混杂，不能据此声称 backbone robustness。
- 决定：`n1_existing_assets_insufficient_for_top_tier_regret_benchmark`。关闭直接用现有
  assets 做完整 N1 benchmark；不把前两项统计冒充三段因果分解，也不先增加同构 UG-grid
  行数或随机 seed。
- 产物：机器报告 SHA-256
  `d17bb8eec9bf0f5cce89105d43c0a676b134ce0779b9f799a4c02903ae3d62c7`；module/runner/test
  SHA-256 为 `60b5fa33...bcd60` / `1daab181...9fd5a` / `b72515bf...6f5b`；完整审计见
  `n1-existing-sibling-regret-benchmark-feasibility-audit-20260903-v1.md`。
- 验证：5 个 targeted tests、三文件 mypy、compileall、Black in-process check、第二次
  independent output byte comparison、JSON decision assertions、credential scan 与
  `git diff --check` 通过。
- 资源/泄漏：CPU-only 流式只读；无模型加载、Slurm、GPU、optimizer 或 checkpoint；
  validation/test/reserve 未访问。
- 当前最佳结果：项目仍无 deployable 正主结果；N1 给出可审计的资产边界，防止把大样本
  误当成完整可识别 benchmark。
- 下一步：N2 先形式化严格可加的 stop/selection/prefix/evidence decomposition，做一手
  新颖性碰撞与最小 factorial augmentation 的 sample/GPU-hour/storage audit。未同时通过
  novelty、identifiability、同数据集多 backbone、多动作族和 power gate 前不生成新数据。

## E-20260903-13：N2 严格可加 causal-regret 新颖性与识别 gate

- 假设：可以把 tool-agent 总 regret 严格分为 stop、action selection、action-prefix 与
  visual-evidence use 四个可识别、非负、可加的项，并以此形成不同于 The Illusion 和
  GapSight 的主贡献。
- 实现 commit：`0cc20ab7235b1be4c0af4fbe6d264c854f8cecaf`。新增 dependency-free
  数值审计，绑定 N1 report SHA-256，覆盖三类 stop/call/selection 行为、固定 prefix 的
  real/counterfactual observation、两组 observationally equivalent ideal continuations
  与 best-of-k replication sensitivity。
- 正结果边界：stop regret 与 selection regret 是非负严格分解；三个注册例的 additive
  residual 都为 0。固定 action 后，`real-direct=(counterfactual-direct)+
  (real-counterfactual)` 也严格成立。
- 否决证据：后两个量是 signed effects；例中 prefix/evidence 分别为 `+0.1/-0.3`。真正
  evidence-use regret 需要 ideal continuation；完全相同观测可产生 `0` 或 `0.4` regret，
  因而不可识别。单次成功率 `0.6` 的 best-of-1/2/4/8 ceiling 为
  `0.6/0.84/0.9744/0.99934464`，不是对 replicate 数不变的 estimand。
- 文献：The Illusion 已以 causal graph 分离 action-induced shortcut 与 observation-mediated
  path，并定义 fixed-prefix Visual Evidence Gain；GapSight 已从 global/crop loss gap 学
  stop、utility 与 box；ToolVision 已把 stepwise evidence gain 和 with/without-tool benefit
  用于 SFT/RL 数据构造。N2 不具备不可约新颖性。
- 决定：七项 gate 通过四项，最终
  `n2_additive_causal_regret_candidate_not_identified_and_not_novel`。关闭当前 causal-regret
  benchmark/decomposition 主路线，不做 augmentation 资源估计或数据生成。
- 产物：N2 JSON SHA-256
  `60b398454f6a495c4fbcb337a0c1eae075cc1536ea09f2f78b2f0a2c2ac99404`；module/runner/test
  SHA-256 为 `d5b94e5d...eacd2` / `27ecaf3e...85fa` / `e89a71ad...3295`；完整审计见
  `n2-additive-causal-regret-novelty-identifiability-gate-20260903-v1.md`。
- 验证：12 个 targeted tests、三文件 mypy、compileall、Black、deterministic report、
  N1 hash 正/负路径、JSON decision、凭证扫描和 `git diff --check` 通过。
- 资源/泄漏：CPU-only；无模型加载、Slurm、GPU、optimizer、checkpoint 或 protected
  split；机器报告冻结 `authorized_new_gpu_jobs=0`、`authorized_new_checkpoints=0`。
- 当前最佳结果：项目仍无 deployable 正主结果；N2 防止把 causal effect 改名成 regret，
  或用可调 best-of-k privileged oracle 制造贡献。
- 下一步：N3 只读审计公开 tool-capable checkpoint 的许可、固定 revision、prompt/parser
  compatibility 与真实 tool support。它只建立强 baseline；任何训练前还需单独证明 signed
  same-prefix credit 与 ToolVision/TACO/CodeVision 不同。

## E-20260903-14：N3 公开 tool checkpoint 与独立新颖性联合 gate

- 假设：存在一个公开、许可清晰、可固定 revision、与当前 runtime/tool schema 精确对接
  且已有 parser-valid 非零执行证据的 checkpoint；同时，same-prefix signed action credit
  相对近期训练方法仍有不可约的新颖核心。两项同时通过才允许下载与无训练 GPU smoke。
- 实现 commit：`aa332977869d5b603f839745ad596df2d3f6d1cc`。新增 normalized registry、
  dependency-free joint gate、local cache/两个代码仓 revision/license 检查、N2 hash gate
  和六个单测；不把“候选可下载”误写成“baseline 已验证”。
- 公开权重：VTool 3B 为 public/ungated/MIT、4,065,787,904 parameters、
  8,143,089,840 bytes，revision
  `0ca11e812287b5c024c7277db71859da5bda17ac`；VTool 7B 为 8,292,166,656 parameters、
  16,595,836,368 bytes，revision
  `b5c901087a12796ab1a783520e1098a194eaa540`。二者均为 Qwen2.5-VL family，均不在
  本地 cache；若获授权，机器选择较小的 3B。
- Baseline gate：7 项通过 4 项。公开、full revision、MIT 与 runtime model family 通过；
  当前代码没有精确映射新 checkpoint ID，model card 没有 prompt/parser contract，也没有
  exact artifact 的 raw response→parse→execution trace。旧 eval 脚本使用不同的旧 VTOOL
  model IDs；新 `training-v2` 默认从 base model 训练并由 Parquet 提供 prompt。
- 新颖性 gate：TACO 已用 tool-off/tool-on answer outcome difference 定义 signed tool
  value，并做 responsibility-aware token routing；TAPO 已做 action-level counterfactual
  witness/credit transfer；The Illusion 已定义 fixed-prefix observation intervention；
  ToolVision 已用 with/without-tool benefit supervision。五个候选 core claims 全被覆盖，
  novelty checks 为 0/6。
- 决定：`n3_public_initializer_exists_but_joint_gate_failed_before_download`。VTool 3B
  是可复现强 baseline 候选，但当前 H5 只剩现有机制的组合/约束差异，不能用换 initializer
  恢复顶会新颖性。
- 产物：registry SHA-256
  `26e11938323078005d47053f25fe2b3909bc1f0ef2a62ce0b8e3344f4110ab2e`；机器报告 SHA-256
  `6d145cba4846ff608788b5dc8791d7fabcd0cdd1380ff9f2907bea5be3394f5c`；module/runner/test
  SHA-256 为 `be8a6334...a842` / `5811bce8...010d` / `30d9f10e...dc86`。完整审计见
  `n3-tool-checkpoint-and-novelty-joint-gate-20260903-v1.md`。
- 验证：N3+N2 共 18 tests 通过；三文件 mypy、compileall、Black check、两次报告
  byte comparison、N2 SHA-256 gate 与 `git diff --check` 通过。第一次 mypy 未提供
  `MYPYPATH=src` 而出现 import-not-found，按仓库 src-layout 修正验证命令后通过，代码未改。
- 资源/泄漏：只读取公开元数据、本地代码/cache 与 N2 报告；没有下载模型、Slurm、GPU、
  optimizer、protected split 或新 outcome。`downloaded_checkpoint_bytes=0`、
  `authorized_new_gpu_jobs=0`、`authorized_new_checkpoints=0`，没有计算任务邮件事件。
- 当前最佳结果：项目仍无 deployable 正主结果；N3 明确区分“强 baseline artifact 存在”与
  “当前方法值得训练”，避免为已碰撞的 H5 消耗 8.14 GB 下载和 GPU 排队。
- 下一步：N4 先做零成本 problem-selection gate。候选必须给出不依赖答案标签/既有 tool
  rollout 的独立机制、明确 estimand、能在 CPU/已有资产上推翻它的预测，以及相对
  The Illusion/TACO/TAPO/ToolVision/普通 value router 的不可约差异；未满足前不实现或开 GPU。

## E-20260903-15：N4 selector information-boundary 形式化与碰撞 gate

- 实现 commit：`42ce7ea66d4b25e10746b1ba4e5a144f3807dae9`。
- 问题与假设：视觉工具/裁剪方法的结论可能依赖 selector 在动作前获得的信息。如果逐方法
  显式登记 selector-visible fields，并在相同信息集、action bank 和完整净效用下比较，
  则已有方法排序可能相对允许 full-resolution selector leakage 的结果发生实质反转。
- 候选更换：原拟议“低分辨率 preview 自监督预测未观察 crop，再计算 prospective VOI”
  因 VOILA、Learning to Look Around、AdaptVision 与 Starve to Perceive 的直接邻近而在
  GPU/模型实现前放弃。N4 改为 information-set-correct evaluation 候选。
- 形式化：对世界 `w`、允许观察状态 `z(w)` 和已扣除 acquisition/proposer cost 的
  `U(w,a)`，验证 `V_full-V_pi=(V_full-V_obs)+(V_obs-V_pi)`。Exact visual alias fixture
  中两个不同 2×4 raster 具有相同 1×2 preview，`V_full=1.0`、`V_obs=0.5`、aliasing
  regret `0.5`；细化观察或 alias cell 共享最优动作时该项为 0。
- 新颖性边界：Self-Certification of Representation Adequacy 已直接覆盖 representation
  aliasing regret/action conflict，VQABench 已部分覆盖 preprocessing/end-to-end cost；这两
  项不能主张新颖。初筛后暂未发现直接覆盖的联合单元仅为 selector-input ledger、matched-
  visibility comparison 与 cross-information-set rank-reversal test；“暂未发现”不是证明。
- 机器结果：`n4_information_boundary_candidate_survives_formal_gate`，14/14 checks 全真。
  Preview-only toy 排名为 conservative `0.6` > adaptive `0.5`；full-resolution selector
  条件为 adaptive `1.0` > conservative `0.6`，严格 pairwise reversal 被检出。另一个 fixture
  中 raw task utility `0.70` 的方法扣除 `0.05` acquisition 与 `0.06` proposer cost 后为
  `0.59`，低于无额外成本 baseline 的 `0.65`；信息边界不匹配的比较 fail closed。
- 真实数据 seed：绑定既有 RICO integrity report，35,352/35,352 required images decode，
  三项必要可用性 gate 全真；19 个 dimension mismatch 原样保留为 QC 风险。ScreenQA 既有
  allocation 含 6,007 ranker-training、4,001 risk-calibration、6,000 formal-test、1,004
  reserve、11,348 untouched images，image/component 跨角色 overlap 均为 0。本项没有读取
  action outcome。
- 产物：registry/module/runner/test/report SHA-256 分别为
  `25dd301a...eebe12f` / `eea7468b...25d7a0d` / `5740d5d6...e0e0fc6` /
  `8e2baa9e...2c925a0` / `d34449b6...aa6ef5c`；上游 N3 report hash 为
  `6d145cba...394f5c`。完整审计见
  `n4-selector-information-boundary-formal-gate-20260903-v1.md`。
- 验证：12 个 N4 tests、N4+N2 共 24 个 targeted tests 与全仓 pytest 均通过；三文件
  mypy、compileall、Black in-process check、两次 report 字节比较、N3 hash 负路径、JSON
  断言、凭证扫描和 `git diff --check` 通过。Black CLI 在 NFS 环境等待超过 60 秒后终止，
  用同版本无 cache formatter 复验通过；不是代码或格式失败。
- 资源/泄漏：CPU-only；existing outcomes opened `0`、Slurm/GPU `0`、optimizer `0`、新增
  checkpoint `0`；formal-test/reserve 未访问，没有计算任务邮件事件。
- 当前最佳结果：项目仍无 deployable 正主结果。N4 只通过 formal/collision-screen gate，
  尚无真实 rank reversal、实际效应、跨数据集结果或论文主张。
- 下一步与止损：N5 在 outcome 前冻结 selector 信息集、相同 UG action bank、完整成本、
  entropy/random/fixed/exhaustive/learned-router 强基线、source bootstrap、primary reversal
  statistic 与最小实际效应。只用 ranker-training 拟合、risk-calibration 一次性 screen，
  继续封存 formal-test/reserve。若无稳健实质反转、效应仅在 privileged full-resolution
  成立或文献直接覆盖剩余三项，则关闭 N4，不提交 GPU 追结果。

## E-20260903-16：N5 同预算信息集效应回顾性否证

- 假设与诚信边界：在同一 DocVQA sibling bank、同一 5% call budget、相同 action bank
  与成本定义下，较高信息的 semantic-context router 应显著优于较低信息的
  context-geometry router；同时 ScreenQA OOF 应出现至少 `0.001` 的一致增量。旧的两个
  question-weighted aggregate 和 ScreenQA OOF aggregate 已知，所以本项只定义为
  retrospective route-falsification，不冒充盲测、confirmatory 或 formal result。
- 冻结：协议/config 由 commit `9e674abb6ca08ab21266f5ddc308579cfa9f0dff`
  在逐 decision 配对结果读取前固定。Config/protocol SHA-256 分别为
  `c661c7fd4362cb2abf62058a769b1d5f557ae6132abdf85b8e20639ced1f0b00` /
  `276b936309efbf814d3b2467526e75385bad929849bf2b66864d1bb6acc44d74`；primary 使用
  20,000 次 source-balanced paired bootstrap、97.5% CI、seed `20260903`。
- 实现：commit `2df1ad20e05740b34a5d32ce761f1175891173ba` 新增 dependency-free
  evaluator、runner 和 8 个单测；验证输入哈希、N4 合同、模型/feature coverage、旧 frozen
  evaluation 复现、outcome-blind matched call set、成本单调性和 ScreenQA role 封存。
  Module/runner/tests SHA-256 为 `16a4fc85...fd003` / `97eaab35...61b0` /
  `6085976b...e7a5`。
- 数据：此前已打开的 DocVQA 1,608 decisions、400 sources；每个 decision 都有
  `ANSWER_NOW + 4 UG-grid ZOOM`。Matched call set 为 80/1,608（4.9751%）；两个 learned
  router 的 call overlap 为 60、Jaccard `0.60`，crop action agreement `0.300995`。
- Primary source-balanced 结果：context utility `-0.0027431835`；semantic utility
  `-0.0029674190`，97.5% CI `[-0.0075754216, 0.0009648282]`；higher-minus-lower
  `-0.0002242355`，paired 97.5% CI `[-0.0052888564, 0.0043010909]`。给 semantic
  feature acquisition cost 取最乐观的 0 时已失败；再扣 `0.001/0.005/0.01` 时 utility
  单调降为 `-0.003967/-0.007967/-0.012967`。
- 聚合敏感性：同一 matched call set 的 question-weighted context/semantic utility 为
  `-0.00509079/-0.00395807`，差 `+0.00113272`；source-balanced 后差反号为
  `-0.00022424`。因此后续多 QA source 评测必须同时报告两种聚合，并以 source-level
  推断为主；不能选择有利权重救活候选。
- 强基线与 headroom：source-balanced deployable 第一名是 entropy gate + fixed
  `ug-grid-01`，utility `+0.00125919`，但 97.5% CI `[-0.00357436, 0.00715268]` 跨零；
  answer-now 为 0，两个 learned routers 排第 6/7，四成本 exhaustive UG-style 为
  `-0.00838393`。Idealized post-action entropy 为 `+0.00047844` 且 CI 跨零；matched
  privileged oracle 为 `+0.03117658`，97.5% CI `[0.02145013, 0.04238910]`，只证明
  未利用 headroom，不是 deployable 正结果。
- 跨域止损：ScreenQA ranker-training OOF 的 context/semantic utility 为
  `0.00006547/0.00011371`，差 `0.00004824`，约比门槛小 20.7 倍；两者均无 safe
  non-degenerate threshold。因此没有打开 4,001 张 calibration 图像、9,951 decisions、
  49,755 action rows；formal-test/reserve 继续封存。
- 决定：10/10 artifact checks 通过，但 8/8 scientific conditions 失败；机械决定为
  `n5_current_information_boundary_candidate_not_supported_before_calibration`。关闭当前
  N4/N5 candidate，不调阈值、不换权重、不加相似特征、不提交 GPU。
- 产物与复现：机器报告/文字审计 SHA-256 为
  `ed657489ee63950c73ec685ce24d023ac873f09d4252a752b70faacb752bad0a` /
  `d5556063b2d712fe0b5509f841e0579ad567dea79b5ff2e4c852a33a619a2341`。运行命令为
  `PYTHONPATH=.:src /userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python scripts/audit_n5_information_set_retrospective.py --config configs/n5_information_set_retrospective_v1.json --output <new-output.json>`；
  第二次独立输出与正式 report 字节级一致。CPU elapsed `11.4s`，峰值 RSS 约 540 MiB。
- 资源：CPU-only；新 Slurm/GPU/checkpoint `0/0/0`，protected outcome `0`，没有计算任务
  邮件事件。当前唯一 checkpoint 仍是 Job `206205` 的 `global_step_2`，约 42 GiB。
- 当前最佳结果与下一步：项目仍无 deployable 正主结果。下一候选重新从 problem selection
  开始，必须解释 privileged oracle 与跨-source router 失败之间的残差，并先通过一手文献、
  可识别性和零成本数据 gate；不把 source weighting 或 fixed-crop 偶然正点包装为贡献。

## E-20260903-17：固定工具 pre-action predictability 协议与资产 smoke

- 假设：如果 pre-action VLM state 含有稳定的工具效用信息，随着 L0 到 L3 representation
  增强，至少一个成本无关 target 应在 image/source-disjoint test 上转化为优于强基线的
  正 utility；若只有 post-action probe 成功，则应转向表示/信息获取而不是继续调 gate。
- 冻结任务：二元 `ANSWER_NOW/USE_VISUAL_TOOL`；工具固定为四-crop exhaustive entropy
  search，最低 post-action entropy、`action_id` 升序 tie-break、收费四次；不学习 `where`。
  `lambda=0.05` 仅在 policy time 使用。
- 协议：ChartQA/DocVQA/HRBench × L0/L1/L2/L3 × direct-gain/rescue-harm/factorized，严格
  36 cells；seeds `17/29/47`；source 与 decoded-RGB hash 双重隔离；validation-only
  selection；test 单次冻结评估；20,000 次 paired source bootstrap、95% CI。
- 实现：新增 typed pre-action allowlist、L0--L3 feature-vector contract、固定工具 sibling
  collapse、source-balanced headroom、36-cell checker 和确定性终局 verdict hierarchy。
  raw feature dict 即使含 outcomes，也只能显式拷贝 allowlist；任何 target 字段进入
  `pre_action` namespace 会 fail closed。
- Retrospective 输入：既有 opened ChartQA 2,500 decisions、DocVQA 13,580 decisions、
  V* proxy 191 decisions；rollout SHA-256 分别为 `881526cc...973c`、`9109d5c8...8fc3`、
  `e4d1a0b0...2aec`。旧 `.pt` bundle 只做存在性/hash 审计，不加载到 predictor。
- 结果：always-call source-balanced utility 为 `-0.180800/-0.194816/-0.184293`；privileged
  binary-oracle utility 为 `+0.028160/+0.010270/+0.054450`，oracle call rate 为
  `3.52%/2.38%/6.81%`。这些结果证明标签实现和非零 headroom，不是正式预测结果。
- 机械决定：`retrospective_assets_only_formal_matrix_incomplete`；正式矩阵 `0/36`。原因是
  缺 max probability、top1-top2 margin、完整 L3 states、HRBench 和 untouched test；旧
  ChartQA split 也不满足当前 RGB/source 双重隔离合同。
- 验证：16 个 predictability targeted tests 通过；相关 11 文件 mypy 无错误；compileall、
  Black in-process check 与 `git diff --check` 通过。正式 asset report 可由
  `PYTHONPATH=src python -m beyond_entropy.predictability_asset_audit --config configs/predictability_retrospective_assets_v1.json --repository-root . --output <new-output.json>`
  重建。
- 产物：protocol config SHA-256 `0618583e...df7a`；asset config SHA-256
  `7f06d11a...794`；机器报告 SHA-256 `14c111b1...c0f`。CPU-only，无 Slurm/GPU、无新
  checkpoint、无 protected outcome、无邮件状态事件。
- Full synthetic smoke：三 synthetic benchmark、36/36 cells、seeds `17/29/47`，共 108
  seed-runs；三组 source/RGB split audit 全通过，`formal_claim_eligible=false`。首次完整
  运行因单类 AUROC/AUPRC 产生 NaN 而被 strict JSON 拒绝；改为显式 `null` 后原配置重跑
  通过。报告 SHA-256 `a458cad9...c0f9`。小样本 MLP 的 500-iteration convergence warnings
  原样保留，未通过调 optimizer/iteration 追 synthetic 指标。
- 下一步：实现 RGB/source split allocator、统一 L0--L3 exporter、三 target trainer、
  validation-only threshold/calibration、全指标和 paired bootstrap；先在 synthetic 与旧
  opened bank 完成真实输入 smoke，再决定最小 GPU feature extraction。

## E-20260903-18：三 benchmark 无泄漏数据冻结

- 目的与诚信边界：在任何新 test rollout 前固定 ChartQA、DocVQA、HRBench 的
  train/validation/test；selection 只读取公开数据身份、source 和解码 RGB 内容，不读取
  Qwen 输出或工具 outcome。历史 opened 数据只进入 train/validation，新 test 在本项后
  继续封存。
- 分配：ChartQA `3600/900/1000` states（每 state 独立 source/image）；DocVQA
  `10861/2719/2147` states，对应 `2800/700/500` documents 与 images；HRBench
  `480/160/160` states、`480/160/160` source IDs、`89/31/31` decoded-RGB images。
  HRBench 重复图片通过 source/RGB 联合连通分量整体分配，未跨角色拆散。
- 完整性结果：三个 benchmark 的 train-vs-validation、train-vs-test、
  validation-vs-test 均为 `source_overlap=0`、`decoded_rgb_overlap=0`；报告字段
  `selection_used_model_outcomes=false`、`new_test_rollouts_opened=false`。独立复核遍历
  22,027 条 manifest，确认 9,651 个唯一实际图片路径存在，并对 HRBench 151 个唯一图片
  重新解码计算 RGB 哈希，151/151 一致。
- 实现与性能修正：最初 HRBench parquet 整表 `to_pylist()` 造成约 3GB base64 字符串的
  内存/换页开销，逐图重编码 PNG 又预计占约 30GB。最终 commit
  `afa103f596896088bd8eea358301399bc98f49b4` 改为 parquet 逐行流式处理、保存已验证的
  RGB digest、直接写官方 JPEG/PNG bytes；科学 split 身份不变，最终全部冻结数据约
  1.2GB。34 个相关单测、mypy、Black、bash syntax 与 `git diff --check` 通过。
- 产物：allocation config SHA-256
  `542efa523d19e652521aea01c06fea507b5d99a82f166f344d7be410dc166c75`；allocation
  report SHA-256 `4c072355b75dcd7b228267f30c4790efa3d9facbdae1a731ac903ec351efb468`。
  数据文件保持在 ignored `data/predictability-audit-v1/`，不把约 1.2GB 图片加入 git。
- 资源与下一步：本项 CPU-only，无 Slurm/GPU/checkpoint、无 test outcome。下一步只在
  opened ChartQA train 的一个状态上执行真实 Qwen baseline + 四 crop rollout 与 L0--L3
  extractor smoke；通过后再估算并冻结正式 train/validation 分片，test 仍不打开。

## E-20260903-19：真实 L0--L3 单行 smoke 首次 fail-closed

- 范围：Job `206627`，1×H800，opened ChartQA train 首行，code revision
  `2bfb11b078e2cdaf6ac15b9a5cc1400c3bf86e35`；不读取新 test。Slurm 邮件为 `ALL`。
- 已通过：Qwen2.5-VL-3B pinned revision 完成 baseline + 四个固定 UG crop，持久化 5 条
  sibling rollout；manifest/rollout SHA-256 为 `136d9d89...f11` / `ede270f7...78db`。
  第二次模型加载也完成，故不是 GPU、权重、离线 cache、rollout 或显存故障。
- 失败：L3 `encode_multimodal_states` 调用当前 Transformers
  `apply_chat_template(tokenize=True)` 时，system message 的字符串 `content` 被当成
  content block 序列，报 `TypeError: string indices must be integers`。Job 在 25 秒后
  `FAILED`、`ExitCode=1:0`；execution report SHA-256 `38628d96...45b1`。未产生 feature、
  checkpoint、test outcome 或科学指标，因此不是方法负结果。
- 修复合法性：同一真实 processor 上比较原字符串 system content 与结构化
  `[{type:text,text:...}]`，两者模板文本字节完全一致，SHA-256 均为
  `84690aefd39673f4a571ec0701059d140c50aa32bc1d44759f3aaf8ab3fd2d84`；结构化形式成功
  得到 `[1,317]` input IDs 和 `[1088,1176]` pixel values。修复只改变 API 容器形状，
  不改变 prompt token、模型、数据、特征定义或 protocol。
- 验证与下一步：新增 structured-system helper 单测；相关 semantic/predictability tests、
  mypy、Black、bash syntax 与 diff check 通过。下一步在新 artifact root 重跑同一单行
  smoke；通过前不提交正式 train/validation，也不打开 test。

## E-20260903-20：真实 L0--L3 单行 smoke 通过

- 范围：修复后 Job `206628`，1×H800，opened ChartQA train 同一首行，code revision
  `a1306bd4200734bb5527da93777e5f097619bc1b`；不读取新 test，邮件为 `ALL`。
- 结果：Slurm `COMPLETED`、`ExitCode=0:0`、runtime 21 秒。完整生成 5 条 sibling rollout
  与 1 条 feature；固定工具 `tool_calls=4/tool_cost=4.0`。L0/L1/L2/L3 实际维度分别为
  `3/22/6147/6147`，所有元素有限。rollout SHA-256
  `ede270f74541da33e9652838ef62532d2b4b51771f124a59cdb0fa9f73f278db` 与失败前完全相同；
  feature SHA-256 `a8c13a1e50dacd1d1df37b77c345a633ebaba17f6dd686b90a8853edd3221aba`。
- 独立复核：report、execution、5 siblings、1 feature row、四次收费、role、code revision、
  manifest/rollout/feature hashes 与四层有限数值共 11/11 checks 全真。smoke report/execution
  SHA-256 为 `9dd31b91...be88` / `fae94194...40a6`。
- 运行合同：Qwen2.5-VL-3B revision `662855...cd3`，Transformers `5.4.0`、PyTorch
  `2.4.0+cu121`、bf16、SDPA、离线 cache；H800 peak allocated/reserved bytes 为
  `7658491392/7786725376`（约 `7.13/7.25 GiB`）。
- 边界与下一步：这是 opened-data 工程证据，正式矩阵仍为 `0/36`，不能推断效用可预测。
  下一步用相同代码扩为 32-state opened ChartQA throughput smoke，以剔除两次模型加载的
  固定开销并冻结 states/hour、GPU-hours、shards 与 checkpoint cadence；test 继续封存。

## E-20260903-21：ChartQA 32-state 端到端吞吐 gate

- 范围：Job `206629`，1×H800，opened ChartQA train 前 32 states，code revision
  `ebf859f16a0c1f5d9278fea9cd4add0322ebe100`；不读取 test，Slurm 邮件为 `ALL`。
- 结果：`COMPLETED`、`ExitCode=0:0`、67 秒；160 条 sibling rollout、32 条 feature，
  固定工具每 state 恰为 4 calls。L0/L1/L2/L3 维度在全部 32 states 稳定为
  `3/22/6147/6147`。独立复核 report/execution、coverage、四次收费、唯一 identity、
  provenance hashes 与全部有限数值共 7/7 checks 全真。
- 哈希/资源：manifest/rollout/feature SHA-256 分别为 `a19b78a5...bcb`、
  `be2939ed...78a`、`b84b93f3...8a8`；report/execution SHA-256 为
  `2dd00b57...2cf` / `01895562...ce8`。H800 peak allocated/reserved bytes 为
  `7858722816/8319401984`（约 `7.32/7.75 GiB`）。
- 吞吐/存储：包含两次模型加载的端到端速率为 `32/67*3600=1719.4`
  states/H800-hour；最终 rollout `578220B`、feature `1381122B`，合计约
  `61229B/state`。三 benchmark train+validation 共 18,720 states，线性外推
  `10.9 H800-hours` 与约 `1.15GB` 最终文件；未测跨数据集前使用 1.5 倍保守预算
  `16.4 H800-hours`。
- 边界与下一步：smoke 不用于查看或选择 endpoint，正式矩阵仍为 `0/36`。下一步以同一
  code path 在 DocVQA 与 HRBench train 各跑 8 states，验证 scorer、重复图片、文档图像和
  高分辨率路径；二者通过后才冻结正式 shard size/checkpoint interval 并提交开发数据。

## E-20260903-22：DocVQA/HRBench 跨域真实 feature smoke

- 范围：同一 clean code revision `b071de227c1953266caf7eba0688d7d7d58b6edd`，分别读取
  DocVQA 与 HRBench opened train role 前 8 states；不读取任何 test outcome。Job
  `206630/206631` 均为 1×H800，Slurm 邮件为 `ALL`。
- Slurm 结果：两者均 `COMPLETED`、`ExitCode=0:0`，runtime 分别为 39 秒和 84 秒，零
  restart。每个 run 均生成 8 条 feature、40 条 sibling rollout；逐 state 为一条
  `ANSWER` 和四条 `ZOOM`，固定工具 collapse 恰为 `tool_calls=4/tool_cost=4.0`。
- Feature contract：DocVQA 与 HRBench 的 L0/L1/L2/L3 维度都稳定为
  `3/22/6147/6147`；独立重新加载两份 `.pt` 后确认 16/16 rows 的四级向量全部有限。
- DocVQA hashes：manifest
  `891f0739970e531a881395c0a5899a87c161525231cb9ddc1c47d40148b8354c`，rollout
  `6509c8ae866474d2323dd68fb2cefdff1ac85668ba092855aefbd2271a1297df`，feature
  `ac3c16794a839315538b22efb9a0386f9ae2991f05cdd917853c8a8da896aeb0`，provenance
  `a162bc042ad30060a86fedfa4a928568e3456416e75a9254ff0644919672979c`，smoke report
  `481f2f390c4547f26884ebb49c2a8c89faa5c94c4bd653c4c0830e851d0e3332`，execution
  `9f56b81928fd587711b08b45367ebe763bfb2003f07538cb8f51d9f4e4ac5c65`。
- HRBench hashes：manifest
  `311004d3bfc64804d3921d1f49c04fe8561f569f8fcae4a206cadc8a6c0aa64d`，rollout
  `23897dd6a84189b910c23f724e1323e541aea6f3a91271d1fe4c3fe4dcfd6328`，feature
  `3a29133fe055ab1de43e3500ae806e78b193726413a36911d2d17b7fbed31c59`，provenance
  `e9dcc31677d94ccf694da7cabc0fd145a5ce8f799bfa4f2230a5b85f450910ca`，smoke report
  `a31217cd9fba06cbf2e91a796ac987991c85f82c179ea911c4e5561a5701e3ea`，execution
  `d2a826f689fc0de89c82ed199b7687bbd7d8fba8b650f471539cd7b7bcf4a74d`。
- 边界：8-state gross throughput 约为 DocVQA `738 states/hour`、HRBench
  `343 states/hour`，但两次模型加载占比很大，不能替代正式分片预算。smoke endpoint
  不用于选择 predictor、阈值或方法；正式矩阵仍为 `0/36`。
- 下一步：在提交完整 train/validation 前，先实现并测试不同 outcome/cost ledger 的
  fixed crop、uniform-random crop、entropy gate 和 exhaustive UG 强基线比较；补齐 paired
  source bootstrap、唯一 post-action diagnostic probe 与 feature shard merge。test 继续
  封存，本项没有 optimizer 或 checkpoint。

## E-20260903-23：异构强基线与独立 outcome/cost ledger gate

- 假设与风险：one-crop fixed/random baseline 与 four-crop entropy-search tool 的实际
  `Ytool`、成本和调用数不同。如果只比较 call mask、却把所有方法套到同一个 exhaustive
  outcome/cost 上，会系统性错算强基线，也无法证明 strongest baseline 只由 validation
  选择。本项只修正 evaluator 合同，不读取正式 validation/test outcome。
- 实现：commit `daa43c148dc1f3a1e2fe5e1603ea1ae464ab7ed6` 冻结并实现六个基线：
  answer-now、entropy gate、SHA-256 random gate、matched-gate global fixed crop、四 crop
  的精确 uniform-random expectation，以及收费四次的 exhaustive entropy search。四个
  action ID 固定为 `ug-grid-00/01/02/03`；threshold、global fixed action 和 strongest
  baseline 只在 validation 选择，再原样应用到 test。
- 比较合同：learned candidate 与 baseline 各自保留逐 decision outcome、cost 与 call
  mask；比较前检查 decision/image/source/`Y0` 对齐，再按相同 source 做 paired bootstrap。
  no-call threshold 改为有限的 next representable float，使严格 JSON 不再写入 Infinity。
- 验证：23 个定向测试、mypy、逐文件 Black 与全仓 656 tests 全部通过。含六个真实基线
  合同的 synthetic 3 benchmark × 4 levels × 3 targets × 3 seeds 运行完成 36/36 cells、
  108 seed-runs，报告仍明确为 `formal_claim_eligible=false`。synthetic report SHA-256 为
  `e8f533572160a68956d9f62a5d9e789795d20c0336f7799e7477a9c1d6929aba`。
- 环境诊断：系统 Python 的 scikit-learn `1.5.1` 不支持当前 MLP `sample_weight` 合同；
  pinned `qwen-vl` 环境为 `1.7.2` 并完整通过。项目 optional semantic dependency 已要求
  `scikit-learn>=1.7`，因此前者是 fail-closed 环境负检查，不是方法结果。
- 边界与下一步：正式矩阵仍为 `0/36`，未提交 GPU、未生成 checkpoint、未打开 test。
  现在只剩唯一 post-action diagnostic probe 与可恢复 feature shard merge 两个代码 gate；
  二者通过后才允许生成完整 train/validation outcomes。

## E-20260903-24：Post-action probe、feature shard 与真实 v2 gate

- 冻结实现：commit `19631c853504981ba97617dfab44dc228e8baf4b` 将唯一 post-action
  probe 固定为 hidden sizes `[128,32]` 的 direct-gain MLP，不允许 architecture/target/
  feature variant 搜索。其输入只含 baseline confidence、四个 crop 按 action ID 排列的
  post-action confidence trace、entropy-selected action one-hot/bbox，以及“原图+选中 crop”
  的冻结 Qwen language/visual/fused prompt states；不含答案文本、ground truth、correctness
  或 target 派生 feature，并与 deployable typed view 分离。
- Shard 合同：feature format 升为 v2，extractor 接受与 rollout collector 相同的
  deterministic shard count/index/key/namespace 并可按完整 state checkpoint resume。merger
  会验证 manifest/full-rollout hashes、code revision、每 shard assignment、rollout 内容、
  fixed-tool label、feature dimension 和全量 decision coverage，再原子生成 merged `.pt` 与
  严格 JSON report。
- CPU/合成验证：全仓 662 tests、11 个 source file 的 mypy、Black、bash syntax、JSON 和
  diff checks 通过；torch 环境中的 16-state two-shard merge smoke 通过。加入 probe 后的
  synthetic matrix 再次完成 36/36 cells、108 deployable seed-runs，并额外完成 3 benchmark
  × 3 seeds 的唯一 post-action probe。严格 JSON report SHA-256 为
  `c912b3b1d1572c62973c911587ae5c33256a906f5e451351ddd624581500eded`。
- 真实 prompt 合同：pinned Qwen processor 对原图+选中 crop 的 structured-system 与 backend
  plain-system 消息生成完全相同的模板文本；SHA-256 为
  `2d449d286999f375190aa6f34b941d9d041af6c0f5ce13dbea9ff220a5eadc89`，实际 tokenized
  shapes 为 input IDs `[1,591]`、pixel values `[2176,1176]`、image grid `[2,3]`。
- GPU gate：Job `206664`，1×H800，opened ChartQA train 一个 state，邮件 `ALL`；
  `COMPLETED`、`ExitCode=0:0`、25 秒、零 restart。仍生成 5 条 sibling rollout 与一条
  feature；固定工具恰为四 calls/四 cost，pre-action L0/L1/L2/L3 维度仍为
  `3/22/6147/6147`，post-action probe 实际为 `6167` 维且全部有限。
- 独立复核：execution、coverage、format v2、code binding、固定成本、selected action、
  pre/post namespace 隔离、label exclusion、有限值和 report hashes 共 14/14 checks 全真。
  rollout/feature SHA-256 分别为 `ede270f74541da33e9652838ef62532d2b4b51771f124a59cdb0fa9f73f278db` /
  `f447e6b0a2f2aa8ad5b8755f4f1df94b7a0fdefa06678216489048030c10db84`；smoke report/
  execution SHA-256 分别为 `50ee5df8c5caa0327f550c7397641f6cddb214c7b2d01069432c72a9324680d9` /
  `5c41e736f1987dcfcd91043df4cc89a5d452a08e9f2f7307c30ff81de1704aa4`。
- 科学边界：本项只证明 privileged probe 与 recoverable export 可以真实执行，不是效用
  可预测性结果；正式矩阵仍为 `0/36`，test 未打开。下一步先用三域较大 opened-train
  shards 冻结吞吐、shard count、checkpoint cadence 和预算，再运行完整 train/validation。
