# 项目状态

更新时间：2026-09-03 14:18（Asia/Hong_Kong）

## 总体判断

项目仍然存活，但原始“训练一个 deployable pre-call value/gate 即得到顶会正结果”
路线现在应视为高概率失败，尚未形成 ECCV/ICCV/CVPR 可投稿主结果。最新
literature-attention 实验否定了 ViCrop/LASER；随后 answer-conditioned hidden-state
候选又因直接文献碰撞在实验前关闭。继续局部调 attention、hidden-state probe、
classifier 或 threshold 没有科学价值。

这不是工程失败，也不是证明研究问题不存在。完整 sibling outcomes 显示有大量
可获益状态，固定 raw action 的 privileged stopping utility 上界也显著为正；
失败点是 deployable pre-action prediction 无法跨 source 稳定识别稀疏正收益尾部。
当前最有价值的产出是严格的 stop/where 因子化与负结果证据，而不是一个成功策略。
下一步已转为机制不同的视觉工具 RL action-local credit feasibility，而不是给旧
gate 换特征。该方向已从 protocol/纯 synthetic G0 前进到真实四卡 actor、vLLM、
agent loop、两步 optimizer 与完整 checkpoint；但 Job `206205` 的 64 条 rollout 没有
parser-valid、可执行工具动作，action-local credit 从未激活，正式触发 `<1%` 停止规则。
当前 sampled on-policy H5 路线因此关闭，仍不能据此提高项目成功概率。

2026-09-02 已做范围纠偏：VTool 只保留为 Apache-2.0 的可运行 RL 骨架和
outcome-only comparator，不再把 pixel、thought 或内部实现与 VTool 的一致性当作研究
问题。此前继续做等价性审计属于范围漂移；它既不验证 H5，也不构成论文贡献。唯一
核心 empirical question 是：same-prefix signed action credit 能否在冻结的强基线与
matched controls 下改善任务分数、cost-adjusted utility 和 harmful-call rate。

当前工程 gate 已前进：官方 Apache-2.0 ReFocus train、token-local adapter 的真实
autograd、隔离 runtime、单行及完整 72 行 Qwen processor、paired fake-server、单卡
H800 vLLM 模型加载/真实首轮 generation，以及最终 Hydra resolved-config gate 均已
通过。Job `205902` 的 jq 前置错误修复后，Job `206170` 已进入四个 FSDP actor 的
Hugging Face 模型初始化，但 pinned runtime 默认强制 `flash_attention_2`，而冻结环境
没有 `flash_attn`，四个 rank 在权重加载前以同一 ImportError 退出。输出目录只有空
Hydra log、launch manifest 与失败 execution report，没有 rollout、optimizer step、
checkpoint 或 H5 指标。因此当前仍不是“新方法失败”，而是“关键性能实验尚未开始”。

根因诊断还发现：只设 `attn_implementation=sdpa` 会把错误推迟到首次前向，因为
`use_remove_padding=true` 会让 verl 把 Qwen attention forward 替换为自定义
FlashAttention 路径。commit `22b89a5ca872356c621203f7bb724042c846a091` 因此对所有
实验臂共同冻结 SDPA 并关闭 remove-padding，加入真实图片 HF actor-load/forward 单卡
smoke；这发生在任何科学结果前，不改变数据、方法、reward、seed、prompt 或对照。
无权重 meta gate 已通过，Hydra 对新增两项在内的 61 项 resolved-config checks 全真。
单 H800 Job `206174` 已进一步加载完整 Qwen 权重、应用 verl 多模态 patch，并用
966-token official-train 真实图片输入完成一次 actor forward；6/6 checks 全真，
`COMPLETED`、`ExitCode=0:0`、50 秒、零 restart。actor report 已由 commit
`0122689050a4bfc91df0a55dd154dc52d7fce83d` 绑定进 G1 launcher。该证据消除了当前
attention backend/首次前向阻塞，但仍未覆盖 Ray FSDP2、optimizer、paired rollout 或
checkpoint，因而只授权有界四卡 G1，不是方法效果证据。

四卡 Job `206179` 随后使用 clean revision `89978bd` 真正启动：四个 FSDP actor、
四个 vLLM engine 和 agent-loop worker 均完成初始化，权重加载与 SDPA 路径通过，
trainer 进入 `0/2` 并调用第一次 `_update_actor`。它在把 batch 分给 4 个 DP rank 时
失败，因为本项目注入的 donor trajectory IDs 是 Python `list`，而 pinned verl
`DataProto.chunk()` 要求全部 `non_tensor_batch` 值为 `numpy.ndarray`。本地已用同一
runtime 稳定复现；commit `ea489f7c520880ab087af761a620b03f357b18e0` 改为 object
ndarray，并新增真实 4-way chunk preflight，提交前及 worker 启动前均 fail closed。
该作业没有持久化 rollout、optimizer step、checkpoint 或 H5 指标，仍不是方法负结果。

修复后的 Job `206184` 首次持久化了 32 行 step-1 rollout，并在第二步 checkpoint
保存阶段因磁盘写满失败。原 gate 只要求 32 GiB，而四个 model 与四个 optimizer
shards 尚未写完就已达 43,367,466,220 bytes（约 40.39 GiB）；extra-state、dataloader
state 与 tracker 均缺失，checkpoint 不可恢复。授权清理后持久盘可用约 77.1 GiB，
资源 gate 已提高为 submitter/worker 双重 64 GiB。step-1 rollout 的结构合同全部通过、
task score mean 为 `0.53125`，但 parser-valid 工具调用率为 0。事后 raw 文本复核显示
19 条 final answer、13 条裸 `focus_on_*` intent；由于冻结协议要求两步 64 条，当前
scientific decision 仍为 `not_available`。这不是 H5 的正式失败，但使最终触发 `<1%`
stop rule 的风险明显上升。

最终重提的 Job `206205` 于 12:26:27--12:31:10 HKT 正常完成两步训练。两步 score
分别为 `0.5625` / `0.53125`，总体 task score 与 realized cost-adjusted utility 均为
`0.546875`；64 条 rollout 中 parser-valid tool call `0/64`，两步 action-credit tool trajectory
count 与 applied credit 均为 0。正式 analyzer 10/10 checks 全真，pair mismatch 和
judge failure 为 0，decision 为 `paired_signed_g1_stop_rule_triggered`。唯一
`global_step_2` checkpoint 结构完整、约 42 GiB，全部文件已重新读取并做 SHA-256
绑定。这个结果不是工程失败，而是证明当前 policy 没有为 sampled action-local credit
提供可执行支持；按预注册规则不运行无法区分 credit 效果的三组 controls，也不事后改变
prompt、seed、temperature 或阈值。

随后只读 raw-response 诊断发现 48/64 是 `FINAL ANSWER`，16/64 是裸
`focus_on_*` intent；15 条可被 Python AST 解析，但 0 条满足真实三参数工具签名，另
1 条混入最终答案。因此正式 stop 不变，但“64 条全部 direct/latent intent 为零”被纠正。
当前 baseline 的 prompt 只枚举函数名与变量、未给签名或可执行模板；下一步必须把
typed-action V2 作为 baseline correctness 独立修复，同时继续对真正新方法做碰撞审计。
进一步检查发现 V1 prompt 所称可用的 `x_values_bbox/y_values_bbox` 并未注入实际
execution context，canonical prompt-following code 会 `NameError`；实际 aliases 是
`columns_bbox/rows_bbox`。B0 CPU core 已在 commit `150803a` 用后两者实现 strict
renderer/parser，并在 pinned runtime 对 x/y 两类调用真执行通过；这仍不是模型结果。

独立 V2 的唯一 H800 generation smoke（Job `206227`）现已完成。16 次生成中 11 次
有 tool intent，7 次形成完整且语法合法的 Python fence，但参数合同、strict parser 与
execution 均为 0。11/11 有意图输出都照抄 prompt 中的无效元变量函数名
`focus_on_x_values_with_MODE`，没有替换具体 mode。因此当前 V2 prompt 的 reliable
baseline gate 失败并关闭；不允许在相同 row/seed 上改 prompt 后追结果。这不改变
Job `206205` 的 G1 stop，也不是新方法失败或成功的证据。

## 已完成的证据链

1. InfographicVQA official-train 已完成 23,946 decisions、2,204 sources、4,406
   images 的完整 sibling bank 与 source-level audit；validation/test/reserve 未开。
2. DECAR nested OOF（Job `203049`）正式 `decar_not_advanced`；0.5% utility
   `-0.000273`，95% CI `[-0.000460, -0.000103]`，主要失败为 stopping enrichment。
3. Entropy-when / oracle-where 诊断（Job `203078`）得到
   `where_bottleneck_supported`，证明 entropy 所选状态中存在可由更好 action
   selection 释放的正收益，但该 oracle 不可部署。
4. Relative-where OOF（Job `203237`）失败，teacher-action agreement 仅
   17.4%--22.8%，暴露 source-generalization 问题。
5. Raw-attention 特征与评估（Jobs `203257`/`203276`）完成。Raw action 在 5%/10%
   显著优于 fixed、random、old-DECAR 与 relative where，但所有净 utility 为负；
   5% 为 `-0.000410`，95% CI `[-0.002438, 0.001681]`。
6. Fixed raw-action stop diagnostic（Job `203290`）显示 unrestricted privileged
   stop ceiling `+0.021318`，95% CI `[0.018447, 0.024444]`；attention max/margin
   均比 entropy 更差。
7. Fixed-action signed-value OOF（Job `203330`）在唯一 2% primary 失败：utility
   `-0.000063`，95% CI `[-0.000739, 0.000655]`；相对 entropy 的 paired interval
   跨零。该线性模型族已关闭。
8. Literature attention 特征（Job `203273`）完整抽取并通过无泄漏审计；评估
   Job `203340` 得到 `literature_attention_where_train_not_supported`。ViCrop 与
   LASER 在 0.5/1/2/5/10% 的所有 utility 点估计均为负，也未显著优于 raw
   attention。完整结果见
   `infographicvqa-literature-attention-where-result-job-203340-v1.md`。
9. Answer-conditioned evidence outcome-free feasibility audit 完成。代码层面可从同
   一 generation 返回 answer hidden states，但 ContextualLens、LRP、VRP、V-Loop
   等已直接覆盖核心 representation/probe/grounding 组合；决定码为
   `answer_conditioned_evidence_candidate_rejected_before_experiment`。没有模型拟合、
   新 outcome 或 GPU job。
10. VTool-R1 `training-v2` 已只读浅克隆并固定到
    `d2aa28353ec10c7f91b39f502925003a81d6982d`。静态 gate 确认 upstream 默认把
    tool/action tokens mask 掉；本项目已通过最小 patch 和独立 adapter 新增
    token-local credit 通路。VTool 到此只承担底座/对照职责，不再继续等价性审计。
11. Same-prefix action-credit protocol v1 与 dependency-free G0 core 已在 commit
    `56b990c767973a8a23060d63293db8657254b35d` 冻结。17 项新测试覆盖 exhaustive
    token roles、arm-specific cost、swap antisymmetry、完整 provenance、序列化防
    篡改、token-local advantage 与 shuffled no-self-donor；完整仓库共收集 509 项，
    479 passed、30 项依赖/资源相关预期 skip。该证据只支持实现定义一致，不支持
    方法性能。
12. 官方 `ReFocus/ReFocus_Data` train 的 Apache-2.0 license、revision 与三个 shard
    SHA-256 已固定；旧 derivative 的 metadata 对照只作为一次性转换证据。无需也不会
    再证明它与 VTool 的 pixel/thought 一致。正式训练输入只允许 official train。
13. 审计期间曾误读 Refocus test 的非图像 metadata/ground truth，并随后一次性枚举
    original ChartQA val/test PNG path IDs。事件已透明登记和隔离；这些 split 永久不能
    再作为本项目 sealed formal evidence。没有拟合、方法选择、GPU 或结果使用。
14. 新增 upstream-shaped adapter 和 paired agent overlay；PyTorch autograd smoke
    证明 signed credit 对 action tokens 产生非零梯度，zero control 为零，observation/
    padding 梯度为零。隔离环境通过所需 imports、版本、pinned commit 与最小 patch
    检查。该证据仍不代表真实 rollout 或性能通过。
15. Official-train converter 重新验证三个 shard 的完整 SHA-256，只读取 train；policy
    输入排除 answer、thoughts、edited image 和 teacher focus boxes。冻结 64 个 train
    structural groups（72 rows）与 32 个 curve-eval groups（33 rows），交集为 0。
    Paired 与 outcome-only Parquet 除 `agent_name` 外逐字段相同。
16. 单行真实 Qwen processor smoke 通过：1 张原图、966 prompt tokens、无 focus-area
    泄漏。Paired fake-server contract 通过：rescue `+0.95`、harm `-1.05`、失败/无收益
    `-0.05`、direct `0`；shared prefix/seed、image-only delta、union mask 和 fail-closed
    均验证。四组 G1 配置已在 `configs/vtool_action_credit_g1_v1.json` 冻结。
17. 单卡 H800 model-load/generation Job `205784` 以 `COMPLETED`、`ExitCode=0:0`、零
    restart 结束。精确 model/runtime/data/config binding 通过；vLLM `0.12.0` 加载 3B
    BF16 权重约占 7.16 GiB，完整 engine load `64.05s`，966-token 视觉 prompt 生成
    12 tokens 用时 `19.21s`。报告 SHA-256 `1a67365b...009a`。首轮为 direct answer，
    因此本项不验证真实 tool/paired branch，也不是性能证据。
18. G1 最终 preflight 已通过。完整 72 行 paired train 经冻结 `RLHFDataset` 与 Qwen
    processor 后全部满足契约：64 个 structural groups，prompt tokens 为 min 438、
    median 931、p95 1,657、max 1,914，小于 4,096；无 protected split、无权重加载，
    报告 SHA-256 `e468c367...c1ed`。Shuffled control 在 batch 少于两个 valid tool
    pairs 时 action loss 置零并计数；PPO mini-batch 已按 upstream prompt-count 语义从
    32 修正为 8。Hydra signed dry-run v9 的 59 项冻结配置检查全部为真，launch
    manifest SHA-256 `dff12612...3da`，resolved config SHA-256
    `5a5d354d...7b3b`。这只授权首次 4×H800、2-step signed G1，不是性能结果。
19. 首次正式提交 Job `205870` 后，在其仍为 `PENDING (Resources)` 时发现 upstream
    rollout writer 只导出 `reward_extra_info`，而当时其中只有 `acc`；若直接运行将无法
    保存 counterfactual score、signed credit、pair provenance、tool success 与
    harmful-call 证据。作业在启动前取消，最终 `CANCELLED`、`RunTime=00:00:00`、
    `Restarts=0`，未消耗 GPU。现已把全部核心字段封装为 JSON-safe audit payload，新增
    自动 analyzer 逐行重建 pair、验证 score/credit/trajectory，并输出 task score、
    cost-adjusted utility、harmful/rescue/tool-call rate。Fake-server v2 与 Hydra v11
    均通过；这是核心可观测性修复，不是 VTool 等价性审计。
20. 修复后提交的 Job `205902` 于 2026-09-03 00:56 HKT 获得资源，但 worker 在
    runtime dataset audit 断言处同秒退出，status `exit_code=2`、
    `scientific_decision=not_available`。原 jq 表达式
    `.checks | all(.[] == true)` 在对象输入上会对布尔值再次执行 `.[]`，以 exit 5
    报 `Cannot iterate over boolean`。相同 jq 与冻结 72 行 report 已稳定复现；改为
    `.checks | all(.[]; . == true)` 后，全真输入通过、含假输入拒绝。训练输出目录
    不存在，未发生模型加载、rollout、optimizer 或 checkpoint；本项不是 H5 结果。
21. jq 修复后的 paired-signed Job `206170` 使用 commit `1fd694c`，于
    2026-09-03 09:55:15--09:57:18 HKT 运行，worker status 为 `failed`、exit 1、
    `scientific_decision=not_available`。四个 actor rank 均在
    `AutoModelForImageTextToText.from_pretrained` 前置 attention dispatch 报
    `FlashAttention2 ... flash_attn seems to be not installed`；这把原因锁定为运行时
    backend，而非模型显存、数据、Ray 启动或 action-credit 数值。status/log SHA-256
    分别为 `316e6038...27a0` / `38ceb188...79c`。修复后的 meta report 决策为
    `vtool_hf_actor_meta_dispatch_passed`：actor class 为
    `Qwen2_5_VLForConditionalGeneration`，model/text/vision 三层均为 SDPA，原生
    attention forward 保留，verl 多模态 model forward 仍应用；report SHA-256
    `6e6afe77...bda5`。该报告不加载权重，仍需单 H800 真实图片前向 gate。
22. 单 H800 HF actor smoke Job `206174` 使用 clean revision `86a345a`，于
    2026-09-03 10:39:34--10:40:24 HKT 在 `gpucluster-g1` 运行，Slurm 终态
    `COMPLETED`、`ExitCode=0:0`、`RunTime=00:00:50`、`Restarts=0`。完整权重加载
    3.16 秒，966-token 真实图片 actor forward 0.82 秒；logits shape 为
    `[1,966,151936]` 且最后 token logits 全部有限，GPU peak allocated
    7,972,130,816 bytes。model/text/vision 三层 backend 均为 SDPA，原生 attention
    forward 保留，verl multimodal model forward 已应用。报告/status/log SHA-256
    分别为 `48e2f12f...8742` / `1bf6acdf...634b` / `8905da36...b93`；未执行
    optimizer，未访问 protected split。该报告已由 commit `0122689` 以内容哈希和
    语义合同绑定进 G1 launcher。
23. 四卡 paired-signed Job `206179` 于 2026-09-03 10:53:38--10:57:06 HKT
    在 `gpucluster-g1` 运行，Slurm 终态 `FAILED`、`ExitCode=1:0`、运行 3 分 28 秒、
    零 restart。它首次证明四个 actor/vLLM/agent-loop 可进入 actor update dispatch；
    随后的 `DataProto.chunk()` 因唯一新增 donor 字段为 list 而断言失败。status/log/
    launch/execution SHA-256 分别为 `9e9c21b0...a07d`、`1afe3867...7906`、
    `ddb40676...94ff`、`e660444d...f1b2`。修复后的同 runtime 4-way synthetic chunk
    7/7 checks 全真，且 submission/worker 均新增该 CPU preflight；没有 H5 性能结果。
24. 修复后的 Job `206184` 于 11:27:40--11:32:35 HKT 运行，保存 step-1 的 32 行
    rollout 后在最终 checkpoint 阶段写满磁盘。rollout/log/launch SHA-256 分别为
    `a2118345...e75c` / `7e882710...722` / `f8e63837...94b`。单步诊断 10/10 checks
    全真、score mean `0.53125`、tool call `0/32`；缺失 step 2，不能执行正式 stop
    decision。实测 checkpoint 已知 shards 至少 40.39 GiB，原 32 GiB gate 已被否定。
25. 存储修复后的 Job `206205` 在 clean revision `9c6bdc4` 上正常完成两步、64 行
    rollout 与唯一 `global_step_2` checkpoint。正式 analysis 为 parser-valid tool call `0/64`、
    rate `0.0`、overall score/utility `0.546875`、10/10 checks 全真，decision
    `paired_signed_g1_stop_rule_triggered`。当前 on-policy H5 路线正式停止，不进入
    zero/shuffled/outcome-only controls。完整审计见
    `vtool-g1-signed-result-job-206205-v1.md`。
26. Job `206205` raw-response 事后诊断 13/13 checks 全真：48 条 final answer、16 条
    裸 focus intent；15 条 AST 可解析但真实参数签名合法为 0，另 1 条语法无效，故
    `fence_only_repair_executable=0`。决定码为
    `g1_zero_parser_valid_support_with_malformed_bare_tool_intent`；正式 G1 decision 未改。
27. Typed-action B0 CPU gate 在 commit `150803a` 通过。V1 prompt SHA-256 未变；V2
    固定 x/columns 与 y/rows grammar，strict parser 覆盖非法 fence、display、axis、bbox、
    kwargs、label 和额外语句；两类 canonical call 在 pinned runtime 均返回 PIL image。
    全仓 532 passed、35 expected skips。本项不含模型 generation 或 GPU。
28. 独立 typed-action B0 单行真实 runtime gate 在 commit
    `47fde3717ba5f8d9f2d3ec5a7ae725e0da94be5c` 通过。Converter 强制 V2 只能使用 official
    train 的 `b0_smoke` 与 outcome-only `vtool_agent`；V2 Parquet SHA-256
    `2c6a6c9b...e8184c`，旧 V1 则继续字节级复现为 `0de5b142...66199`。同一 row/image
    经真实 Qwen processor 得到 975 tokens，26/26 checks 全真；固定 renderer-owned x/draw
    action 经 strict parser 和 pinned VTool context 真执行，输出图像 SHA-256 与原图不同。
    没有加载模型权重、执行 optimizer、写 checkpoint 或访问 protected split；本项仍不证明
    模型会按 V2 prompt 生成合法调用。
29. Job `206227` 在 clean revision `96cd166` 上用 1×H800 完成 16 次 V2 首轮生成，
    Slurm `COMPLETED`、`ExitCode=0:0`、零 restart、81 秒。Intent/fence/syntax/argument/
    parser/execution 计数为 11/7/7/0/0/0，决定
    `typed_action_b0_malformed_tool_intent`。11/11 intents 复制无效 `_with_MODE`；0 optimizer、
    0 checkpoint、无 protected split、raw model text 未执行。完整审计见
    `refocus-typed-action-b0-generation-result-job-206227-v1.md`。

## 当前最佳结果与解释边界

- 最强 deployable `where`：raw-attention action；它显著超过四个旧 where 基线，
  但在现有 stopping 下仍为负 utility，不能进入 calibration。
- 最清晰机制证据：固定 raw action 后，privileged stop 上界 `+0.021318`；这证明
  “值得调用的状态存在”，但不证明它们可由当前特征预测。
- 最新 OOF stop 候选有轻微 precision 改善，但自身 utility 与 paired lower
  endpoint 均未过门槛。
- ViCrop/LASER 是有效的 literature strong-baseline negative，而不是新方法成功。
- 目前没有可宣称正结果的 deployable candidate，validation/test/reserve 必须继续
  封存。

## 距离顶会目标

| 里程碑 | 当前状态 |
| --- | --- |
| 严格数据/无泄漏/强基线基础设施 | 既有数据完成；Refocus train audit 通过，test 已污染并隔离 |
| Action-credit protocol、synthetic G0 与 upstream adapter | 已完成；真实两步 optimizer 已执行，但零工具动作使 credit 通路未激活 |
| 官方 train license/identity 与可执行 runtime | 已通过；不要求 VTool pixel 等价 |
| 真实 converter 与 paired fake-server | 已通过 |
| 单卡 H800 vLLM model-load/generation preflight | Job 205784 通过；仅授权有界 G1 |
| HF actor backend/真实图片前向 | Job 206174 已完整通过；报告已绑定，授权有界四卡 G1 |
| 真实 paired rollout 与最多 2-step optimizer smoke | Job 206205 完成；tool call 0/64，触发冻结停止规则，当前 H5 不晋级 |
| Typed-action reliable baseline | V2 CPU/runtime 通过；Job 206227 真实 generation 因 11/11 intents 复制无效 MODE、0/16 参数合法而失败并关闭 |
| 可部署方法在 source-OOF train gate 取得正且显著 utility | 未完成 |
| 独立 calibration 通过 | 未开始；无候选获授权 |
| Sealed formal 一次性通过 | 未开始 |
| 第二数据集/骨干与外部方法比较 | 未完成 |
| 完整论文主张 | 仅有诊断与负结果骨架 |

所以当前离“可以承诺顶会结果”仍隔着至少 method gate、calibration、formal、
generalization 四个实质台阶。现在不能承诺日期。

## 正在运行

截至 2026-09-03 14:18 HKT，实时 `squeue -u yihangc` 为空；GPU quota 快照为
222,000 分钟总额、42,275 已用、179,725 剩余（约 2,995.42 GPU-hours）。Job `206227`
已正常结束，Slurm 仍可查询到 `COMPLETED` 终态。该任务配置了
`--mail-user=yihangc@connect.hku.hk --mail-type=ALL`，BEGIN/END 在 Slurm 状态通知
范围内；不能据此确认邮件客户端实际送达。当前没有训练在后台运行；B0 V2 的 CPU、
runtime 与 H800 generation 均已结束。持久盘可用 37,983,617,024 bytes（约 35.38 GiB）。
修改仅在本地，未 push GitHub。

## 已关闭的路线

- 当前 fixed four-box bank 上的 DECAR v1、relative-where、raw/literature pure
  attention where-only gate；
- attention layer/head/ratio、max/margin、entropy threshold 与 call-rate sweep；
- 当前 80 维特征上的线性 signed-value stop family，包括更换 C、权重、seed 或
  classifier family 的事后搜索；
- answer hidden-state/contextual embedding/grounding reliability probe 作为独立新
  方法；generic group-DRO、IRM 或 conformal threshold 的局部替代；
- 当前 sampled on-policy same-prefix action-credit H5；Job `206205` 的零 parser-valid
  工具调用使
  方法特有 credit 无支持，不运行无法区分该效应的后续 controls；
- 用已打开 train outcomes 选择有利 operating point，或用 privileged oracle
  冒充部署结果。

## 主要风险

- Positive-net calls 稀疏且跨 source 异质，现有表征只能得到弱排序信号。
- 固定四格 action bank 可能限制 proposal quality；但更丰富 proposer 会增加成本，
  并与 GapSight、CropVLM、AdaptVision 等工作产生更强新颖性碰撞。
- Generic pre-call classifier、necessity/harm learning、attention crop、hidden-state
  reliability probe 与 continuous crop routing 均已有直接相关工作；仅换模型不足
  以构成贡献。
- 即使新的 train OOF 候选通过，仍需独立 calibration、sealed formal 和至少一个
  generalization axis，时间不只由单次 GPU runtime 决定。
- Same-prefix action credit 与 ToolVision 的 stepwise evidence gain 存在强碰撞，且
  Job `206205` 已显示 sampled on-policy credit 有更基础的零 parser-valid support 问题。
  Raw 输出虽有 16 条 intent，但 0 条签名合法。新的
  解法若只是 forced-call、SFT/curriculum、tool bonus 或 off-policy hints，也会与近期
  工作直接碰撞，不能构成顶会主张。
- Upstream FP8 3B script 默认把 `test.parquet` 同时作为 train/val；直接运行会造成
  明确测试泄漏。必须只用 official train 派生开发 split，并关闭 train-time test。
- 旧 Refocus_Chart derivative 不再是训练数据；正式输入改为已固定 revision/hash 的
  Apache-2.0 official ReFocus train。Refocus test 与 original ChartQA test 已因 metadata
  暴露失去 sealed 资格，formal 必须使用从未打开的独立 benchmark/split。
- 隔离环境、完整数据 processor、fake paired generation、单卡 vLLM load/generation、
  Ray 多进程、两步 optimizer、rollout export 和 checkpoint 均已通过真实运行；但 64 条
  中没有 parser-valid action，真实 tool/paired branch 和 action-credit metrics 仍没有支持。
- Job `205902` 暴露出 shell jq object predicate 未被旧静态字符串测试实际执行；两处
  同类表达式现已修正并加入真实 jq 正/负回归；全仓测试与静态检查已通过。重新提交前
  仍需在 clean commit 上执行完整 worker/Hydra 前置合同，避免再次用排队换取可在登录
  节点发现的语法错误。
- Job `206170` 暴露出 vLLM model-load smoke 没有覆盖 FSDP actor 的 Hugging Face
  attention dispatch。单设 SDPA 又会被 remove-padding monkey patch 绕回
  FlashAttention，因此新 smoke 必须覆盖完整权重加载、verl patch 和真实图片前向；
  仅 meta gate 不能授权四卡训练。
- Job `206179` 暴露出 synthetic gradient smoke 没有覆盖 verl 在 DP dispatch 前对
  `non_tensor_batch` 的 ndarray 类型合同。根因已用同 runtime `DataProto.chunk(2)`
  稳定复现并修复；新的 4-way preflight 在 `sbatch` 前和 worker 启动前都执行，避免
  再用模型初始化验证纯 CPU 可检出的类型错误。
- Job `206205` 已确认真实 G1 为 parser-valid tool call `0/64`；不能靠事后改 prompt/temperature/
  seed 制造正结果。现在必须重新审计 exploration/support 假设，且新解法需要独立
  新颖性，不能套用已有 forced-tool curriculum。
- B0 V2 的真实 generation 已分层证实 11/16 intent、7/16 fence/syntax，但 0/16 参数/
  parser/execution。直接机制是所有 intent 都复制无效 `MODE` 元变量；不能在相同
  row/seed 上事后改 prompt 重跑。Concrete-template V3 若需要，只能作为独立预注册强
  baseline，不能包装为主方法。
- 一个完整 distributed checkpoint 实测至少约 40.39 GiB；64 GiB gate 只保证当前
  单臂有界运行。若 signed 通过，四个实验臂的 checkpoint 位于独立目录，必须先制定
  有哈希和可恢复性的迁移/保留方案，不能在当前盘同时无界累计约 164 GiB。

## 下一步最优行动

不再做 VTool 等价性审计，也不再重跑当前 G1 或 V2。Job `206205` 以 parser-valid tool
call `0/64` 正式关闭 sampled H5；Job `206227` 又以 argument/parser/execution `0/16`
关闭 V2 baseline，并把即时原因收敛到元变量模板复制。下一步优先形式化 N0 的
action-boundary interventional objective，要求它在零 valid support 下仍有梯度，且不
退化为 ToolVision、GapSight、representation steering、logit bias、LIRE/LiPO 或
ToolPrefer 已覆盖的目标；N0 新颖性未通过前不提交主方法 GPU。若候选形式化后只是普通
full-information/listwise loss，则关闭并继续选择实质不同路线，不降低投稿目标。若论文
强基线需要修复 V2，则 V3 必须先枚举六个 concrete、parser-valid 模板，并用独立
official-train structural group 与新种子预注册；它只承担 baseline correctness。
