# 项目状态

更新时间：2026-09-03 21:04（Asia/Hong_Kong）

## 当前执行状态

项目已从此前的开放式候选搜索收敛到一个固定、可证伪的审计：判断 pre-action VLM state
能否预测固定四-crop entropy-search 工具的效用。这里不学习 crop 位置，避免把 `where`
问题再次混入 `whether/when`。完整协议已冻结为 3 benchmark × 4 predictor levels × 3
targets 的 36-cell 矩阵，并固定 split、强基线、成本、指标、bootstrap 与终局判定规则。

当前完成的是冻结协议、基础 matrix runner、数据层和三域真实 feature smoke，不是实验
成功：typed pre-action allowlist 会阻止 post-action/label 字段进入模型；固定工具 collapse
会强制四个 sibling crops、稳定 tie-break 和四次成本；matrix checker 会在少于 36 cells
时拒绝完成；终局 classifier 只能按冻结规则输出 `GO/PIVOT/REPRESENTATION/STOP`，证据
不足则报 inconclusive。

对既有 opened bank 的只读 smoke 显示，固定工具的 privileged binary oracle utility 在
ChartQA/DocVQA/V* proxy 分别为 `+0.02816/+0.01027/+0.05445`，说明该二元任务至少有非零
headroom；但 always-call utility 分别约为 `-0.1808/-0.1948/-0.1843`，说明不能靠多调用
取胜。这些数据缺关键 L0/L3 特征与 untouched test，V* 也不是最终冻结的 HRBench，因此
正式完成度仍是 `0/36`，不能据此宣布 GO、PIVOT 或失败。

统一 feature/data contract 和 image/source-disjoint 数据分配已经完成。冻结规模为：
ChartQA `3600/900/1000` states；DocVQA `10861/2719/2147` states、对应
`2800/700/500` source documents；HRBench `480/160/160` states。三套数据任意角色间的
source 与 decoded-RGB overlap 都为零；22,027 条 manifest 的 9,651 个实际图片路径均
存在，HRBench 151 个唯一图片又经独立全量 RGB 哈希复核。分配未读取模型 outcome，三个
test 均未做新 rollout。allocation report SHA-256 为
`4c072355b75dcd7b228267f30c4790efa3d9facbdae1a731ac903ec351efb468`。

opened ChartQA 单行真实 GPU smoke Job `206627` 已运行 25 秒：baseline + 四个固定 UG crop
全部完成并写出 5 条 rollout；随后 L3 extractor 的真实模型第二次加载成功，但当前
Transformers `apply_chat_template(tokenize=True)` 不接受字符串形式的 system content，
以 `TypeError: string indices must be integers` 退出。Slurm 状态为 `FAILED`、
`ExitCode=1:0`；没有 feature、checkpoint、test 或科学 endpoint。这是运行合同 bug，
不是方法负结果。

真实 processor 预检确认，将 system content 改成结构化 text block 后，生成的模板文本与
原字符串版本字节完全一致（共同 SHA-256
`84690aefd39673f4a571ec0701059d140c50aa32bc1d44759f3aaf8ab3fd2d84`），且 tokenize 输出
具有 `[1,317]` input IDs 与 `[1088,1176]` pixel values。修复已加入单测并由后续同一
单行 smoke 验证。作业使用单 GPU、clean revision、输入/代码哈希、离线模型、原子产物、
四次工具收费和邮件 `ALL` 状态通知。

修复后的 Job `206628` 已于 21 秒内 `COMPLETED`、`ExitCode=0:0`。独立复核 11/11 checks
全真：5 条 sibling rollout、1 条 feature、四次工具收费、manifest/rollout/feature hash
绑定、code revision、role 与全部有限数值均通过。真实维度为 L0 `3`、L1 `22`、L2
`6147`、L3 `6147`；H800 峰值 allocated/reserved 显存约 `7.13/7.25 GiB`。该结果只证明
实现可运行，不说明 predictor 有用。后续 32-state opened ChartQA throughput smoke 已完成；
test 仍封存。

32-state ChartQA throughput Job `206629` 进一步 `COMPLETED`、`ExitCode=0:0`、67 秒：
160 条 rollout、32 条 feature、四次收费与全部数值/身份/hash checks 均通过。端到端速率
约 `1719 states/H800-hour`；rollout + feature 最终文件约 `1.96MB`，约 `61KB/state`。
将 18,720 个 train/validation states 线性外推为约 `10.9 H800-hours` 与约 `1.15GB`
最终特征/rollout，当前保守预算为 `16.4 H800-hours`。由于 ChartQA 不代表文档与超高
分辨率输入，又执行了 DocVQA/HRBench train 各 8-state smoke。

DocVQA Job `206630` 与 HRBench Job `206631` 均已 `COMPLETED`、`ExitCode=0:0`，runtime
分别为 39 秒与 84 秒。两者都得到 8 条 feature、40 条 sibling rollout，固定工具每 state
恰执行四个 crops；L0/L1/L2/L3 维度均为 `3/22/6147/6147` 且全部有限。manifest、rollout、
feature 与 execution hashes 已独立复核。至此真实执行路径跨 ChartQA、DocVQA、HRBench
全部打通，但这些都是 opened-train 工程 smoke，不构成任何方法效果证据。

matrix runner 的异构强基线缺口已经关闭。commit
`daa43c148dc1f3a1e2fe5e1603ea1ae464ab7ed6` 实现六个冻结基线、validation-only threshold/
fixed-action/strongest-baseline selection，以及 one-crop 与 four-crop 各自真实的逐样本
outcome/cost/call ledger；learned-vs-baseline 使用独立 ledger 做 paired source bootstrap。
包含全部六基线的 synthetic 36-cell × 3-seed smoke 与当时的 656 个全仓测试已通过。

commit `19631c853504981ba97617dfab44dc228e8baf4b` 又完成唯一 post-action oracle probe、
feature format v2 与可恢复 shard merge。probe 是固定 direct-gain `[128,32]` MLP，独立
typed namespace 不会进入 deployable predictors；merge 会验证 manifest、rollout、code
revision、每 shard 覆盖与逐 decision fixed-tool label，再原子写最终 `.pt`。含 probe 的
synthetic 36/36 × 3-seed 矩阵、662 tests、mypy 和 torch merger smoke 全部通过。

真实 Job `206664` 在 opened ChartQA train 一个 state 上 `COMPLETED/ExitCode=0:0`，runtime
25 秒。独立 14/14 checks 确认 format v2、四次工具成本、selected action、pre/post 隔离、
全部有限值与 hashes；post-action 实际维度为 `6167`。正式矩阵仍为 `0/36`，test 继续
封存；下一步是较大三域 opened-train shard throughput/recovery gate，之后才提交完整
train/validation。

上述 runner 现已完成一次非科学 synthetic 全矩阵 smoke：36/36 cells、每 cell 三个固定
seeds，共 108 个 seed-runs 全部结束；三个 synthetic benchmark 的 source/RGB overlap 均
为零，严格 JSON 报告成功写出。首次运行曾在序列化阶段因单类指标为 NaN 被拒绝，现已
冻结为 JSON `null` 并原配置复跑通过。小样本 MLP 多次达到 500 iteration 上限，属于
synthetic convergence 诊断；正式运行必须记录而不能通过增加迭代或改网络追结果。

## 总体判断

项目仍然存活，但尚未形成 ECCV/ICCV/CVPR 可投稿主结果。原始“训练一个 deployable
pre-call value/gate 即得到顶会正结果”的直接路线应视为高风险：literature-attention
实验否定了 ViCrop/LASER，answer-conditioned hidden-state 候选因直接文献碰撞关闭，
same-prefix action-credit 路线又在零 parser-valid tool support 处触发预注册停止规则。
继续局部调 attention、classifier、threshold 或 prompt 没有科学价值。

这不是工程失败，也不是证明研究问题不存在。完整 sibling outcomes 显示有大量
可获益状态，固定 raw action 的 privileged stopping utility 上界也显著为正；
失败点是 deployable pre-action prediction 无法跨 source 稳定识别稀疏正收益尾部。
当前最有价值的产出是严格的 stop/where 因子化、负结果证据，以及一次能够给出明确终局
判断的 fixed-tool predictability audit。此前 RL action-local credit 已从 synthetic G0
走到真实四卡 actor、vLLM、两步 optimizer 与完整 checkpoint，但 Job `206205` 的 64 条
rollout 没有 parser-valid、可执行工具动作，credit 从未激活并触发 `<1%` 停止规则；该
sampled on-policy H5 路线已关闭。当前不重开 H5，而是完成第六阶段的 36-cell audit。

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

N0 的新颖性/零支持 gate 随后在任何主方法 GPU 实现前完成并关闭。直接最大化完整
interventional action utility 时，macro-action logit 梯度精确等于当前动作概率乘相对
utility，因此在零支持处仍为零；用 utility target 绕过支持则退化为 listwise/AWR，
回归 action value 则退化为既有 router。数值报告 10/10 checks 全真；结合 ToolVision、
The Illusion、GapSight、LIRE/LiPO/ToolPrefer 等一手论文，决定为
`action_boundary_candidate_reduces_to_existing_objective_families`。没有提交 GPU、训练或
读取新 outcome。

N1 随后对现有完整 sibling assets 做了流式机器盘点。四个主开发 bank 覆盖四个数据集、
`59,949` decisions、`299,745` rows，全部 decision 都有 answer-now 与四个 UG-grid ZOOM，
source ID、模型 revision、manifest/rollout hash 均完整。它们足以支持 stop regret 和
注册 action bank 内的 selection regret，却不能支持 evidence-use regret：所有主 bank 的
`239,796` 条 ZOOM 都没有 fixed action prefix、matched real/no-op observation 与 controlled
continuation。与此同时，主证据只有一个工具动作族、每状态一个 replicate，3B/7B 与数据集
混杂；ScreenQA 的同数据集 3B/7B 重叠只有 512 个 opened-development diagnostic states。
机器决定为 `n1_existing_assets_insufficient_for_top_tier_regret_benchmark`。因此现有资产不能
直接包装为顶会完整 benchmark，本项没有提交 GPU 或产生 checkpoint。

N2 又检验了是否能通过严格可加 causal decomposition 修复 N1。结果是：stop 与 selection
regret 可严格、非负地相加；action-prefix 与 visual-evidence effect 也可严格相加，却可正
可负而不是 regret。真正的 evidence-use regret 依赖未观测 ideal continuation；相同
`(direct,counterfactual,real)` 观测可以对应不同 ideal regret，best-of-k 代理还会随 k
机械膨胀。The Illusion 已直接分离 action-induced shortcut 与 observation-mediated path，
GapSight 已覆盖 stop/action utility bank。N2 决定为
`n2_additive_causal_regret_candidate_not_identified_and_not_novel`，同样没有授权 GPU 或
checkpoint。这关闭了当前 causal-regret benchmark/decomposition 主贡献，而不是只说明
数据量不够。

N3 进一步完成公开 tool-capable checkpoint 与独立新颖性的联合 gate。VTool 3B/7B
权重是 public、ungated、MIT 且可固定 full revision；若只看尺寸与架构，8.14 GB 的
`VTOOL/VTool-Qwen2.5-3B` 是首选。但当前公开 artifact 没有把精确 checkpoint 与
`training-v2` prompt/parser 绑定，也没有 exact-artifact parser-valid execution trace，
baseline 只通过 4/7。更关键的是，TACO 已直接覆盖 signed before/after tool value 与
token-level responsibility routing，TAPO 覆盖 action-level counterfactual credit，The
Illusion 覆盖 fixed-prefix observation contrast，ToolVision 覆盖 with/without-tool benefit
supervision；novelty 0/6。N3 因
`n3_public_initializer_exists_but_joint_gate_failed_before_download` 关闭，未下载模型、未提交
GPU、未产生 checkpoint。公开权重仍可作为未来强 baseline，但不能单独恢复当前 H5。

N4 已完成零成本 problem-selection formal gate。最初的“从低分辨率 preview 自监督预测
未观察 crop，再算 VOI”候选因 VOILA、active visual completion、AdaptVision 等直接邻近
工作而在实现前放弃。替代候选是 information-set-correct visual acquisition evaluation：
逐方法登记 selector 动作前可见字段，只在相同信息集、action bank 和净效用定义下给主
排名，并检查跨信息集 rank reversal。机器报告 14/14 checks 全真；toy exact alias 的
`V_full/V_obs` 为 `1.0/0.5`，并成功检测 preview-only 与 full-resolution selector 输入下的
方法排序反转及 acquisition+proposer 双成本导致的排序变化。但 aliasing-regret 分解已被
Self-Certification 直接覆盖，cost ledger 也被 VQABench 部分覆盖，不能作为新贡献。当前
仅剩 selector information ledger、matched-visibility comparison 与 cross-information-set
rank-reversal test 三项暂未发现直接覆盖；它们仍需 N5 的真实数据、预注册结果才能成立。
本轮没有打开既有 action outcome、没有训练/GPU，新增 checkpoint 为 0。

N5 随后完成回顾性、同样本配对的现实效应 gate。协议在逐 decision 结果读取前由 commit
`9e674abb6ca08ab21266f5ddc308579cfa9f0dff` 冻结，并明确披露旧 aggregate 已知，因而不把
它冒充 confirmatory/formal。实现 commit 为
`2df1ad20e05740b34a5d32ce761f1175891173ba`，10/10 artifact checks 全真。在 DocVQA
1,608 decisions/400 sources、共同 5% 预算下，source-balanced context/semantic utility
分别为 `-0.00274318/-0.00296742`；higher-minus-lower 为 `-0.00022424`，paired 97.5%
CI `[-0.00528886, 0.00430109]`。8 项科学条件全部失败。Question-weighted 差虽为
`+0.00113272`，但 source-balanced 后反号，不能作为跨 source 证据。ScreenQA OOF 差仅
`0.00004824`，且两个 router 都没有 safe non-degenerate threshold。因此 N4/N5 当前候选
关闭，避免打开 ScreenQA 9,951 decisions、49,755 action rows 的 risk-calibration；没有
GPU、新 checkpoint 或 protected outcome。

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
30. N0 action-boundary objective 的 dependency-free 三动作 gate 10/10 checks 全真：
    beneficial action 概率 `2.06e-9` 时，expected-utility logit gradient 只有 `1.96e-9`；
    exact-underflow 下两者均为 0。Boltzmann utility target 虽产生 `-0.978` 的非零 CE
    gradient，但它明确是 listwise/off-policy supervision。有限差分误差低于 `1.6e-9`，
    report SHA-256 `c1bfd08a...e4f1`。N0 在 GPU 前因目标族碰撞关闭。
31. N1 对四个主 sibling banks 的流式盘点覆盖 59,949 decisions、299,745 rows；stop 与
    registered-bank action-selection regret 可识别，但 evidence-use regret、主证据的
    同数据集多 backbone、多动作族和多 replicate 不满足，现有资产 benchmark 路线关闭。
32. N2 证明 stop/selection regret 可严格非负相加，但 fixed-prefix/evidence 项是 signed
    effects；ideal evidence-use regret 不可识别，best-of-k ceiling 还随 replicate 数机械
    膨胀。路线因识别失败和 The Illusion/GapSight 碰撞关闭。
33. N3 确认 VTool 3B/7B 公开 checkpoint 可固定，但 baseline artifact gate 只过 4/7，
    新颖性 0/6；未下载权重、未提交 GPU，也未用更强 initializer 重开 H5。
34. N4 information-boundary formal gate 14/14 checks 全真；toy rank reversal 成立，
    但 aliasing/cost 主张已有碰撞，所以只允许 N5 用现实数据做一次低成本否证。
35. N5 在共同 5% 预算下得到 source-balanced higher-minus-lower `-0.00022424`，paired
    97.5% CI `[-0.00528886, 0.00430109]`；8/8 科学条件失败，ScreenQA OOF 增量仅
    `0.00004824`。当前 N4/N5 候选关闭，risk-calibration、formal-test 与 reserve 未打开。

## 当前最佳结果与解释边界

- 最强 deployable `where`：raw-attention action；它显著超过四个旧 where 基线，
  但在现有 stopping 下仍为负 utility，不能进入 calibration。
- 最清晰机制证据：固定 raw action 后，privileged stop 上界 `+0.021318`；这证明
  “值得调用的状态存在”，但不证明它们可由当前特征预测。
- 最新 OOF stop 候选有轻微 precision 改善，但自身 utility 与 paired lower
  endpoint 均未过门槛。
- ViCrop/LASER 是有效的 literature strong-baseline negative，而不是新方法成功。
- N5 matched-budget learned routers 均为负；固定 `ug-grid-01` 的
  `+0.00125919` 点估计置信区间跨零。Matched privileged oracle 为 `+0.03117658`，证明
  headroom 仍在，但不能当作部署结果。
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
| N0 action-boundary interventional objective | 零支持数值与一手文献 gate 完成；退化为既有 listwise/AWR/value-router 家族，GPU 前关闭 |
| N3 公开 checkpoint 与 novelty 联合 gate | VTool 3B/7B 可用，但 artifact provenance 不完整且 H5 core claims 全部碰撞；下载/GPU/checkpoint 均为 0 |
| N4/N5 selector information-boundary 现实效应 gate | N4 formal 14/14 通过；N5 现实效应 8/8 条件失败，当前候选关闭 |
| Fixed-tool predictability 数据冻结 | ChartQA/DocVQA/HRBench 的 source 与 decoded-RGB 双隔离 split 已完成，test 未打开 |
| 三域真实 Qwen rollout + L0--L3 feature path | Jobs 206628--206631 通过；只证明工程可运行 |
| 36-cell predictability 正式矩阵 | `0/36`；强基线、post-action probe 与 shard merge 已完成，待三域正式 train/validation |
| 可部署方法在 source-OOF train gate 取得正且显著 utility | 未完成 |
| 独立 calibration 通过 | 未开始；无候选获授权 |
| Sealed formal 一次性通过 | 未开始 |
| 第二数据集/骨干与外部方法比较 | 未完成 |
| 完整论文主张 | 仅有诊断与负结果骨架 |

所以当前离“可以承诺顶会结果”仍隔着至少 method gate、calibration、formal、
generalization 四个实质台阶。现在不能承诺日期。

## 正在运行

截至 2026-09-03 20:07 HKT，实时 `squeue -u yihangc` 为空。Job `206630/206631` 已分别
以 `COMPLETED`、`ExitCode=0:0` 结束，均配置
`--mail-user=yihangc@connect.hku.hk --mail-type=ALL`；这证明 Slurm 通知配置存在，不能
据此确认邮件客户端实际送达。GPU quota 同时刻快照为 222,000 分钟总额、42,239 已用、
179,761 剩余，即总计/已用/剩余约 `3700.00/703.98/2996.02 GPU-hours`，利用率约
`19.03%`。持久盘可用 350,277,861,376 bytes（约 326.22 GiB），使用率 73%。这些队列、
配额和磁盘数字是时点数据；当前没有训练在后台运行。

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
- 以公开 VTool checkpoint 直接重开 H5；N3 已证明 strong initializer 存在，但 exact
  prompt/parser/support provenance 未闭环，而且 intended credit/routing 主张与
  TACO/TAPO/The Illusion/ToolVision 碰撞；
- 当前 N4/N5 information-boundary benchmark 候选；真实 matched-budget learned-router
  比较没有稳健、实质的正效应，且 source-balanced 后表面优势反号；
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
- N0 的 same-prefix observation effect 本身与 Visual Evidence Gain 碰撞；直接 policy
  gradient 不解决零支持，非零监督版本又属于 listwise/off-policy/value learning。继续
  改 loss 名称或 token mask 不会产生方法新颖性。
- 一个完整 distributed checkpoint 实测至少约 40.39 GiB；64 GiB gate 只保证当前
  单臂有界运行。若 signed 通过，四个实验臂的 checkpoint 位于独立目录，必须先制定
  有哈希和可恢复性的迁移/保留方案，不能在当前盘同时无界累计约 164 GiB。
- N4 的 aliasing/representation adequacy 数学已被同期工作覆盖；如果真实评测没有显示
  selector 信息边界会造成稳健且实质的方法排名变化，剩余三项协议贡献不足以支撑顶会。
- 当前所谓“未发现直接覆盖”只是截至本轮的一手文献初筛；Self-Certification 与 VQABench
  已说明同期碰撞风险很高，N5 前后都需继续 collision audit。
- N5 显示 question-weighted 与 source-balanced 结论可反号；后续所有多问答 source 数据
  必须并列报告两种聚合，并以 source-level 推断为主，防止高 QA-count source 支配结论。

## 下一步最优行动

不再做 VTool 等价性审计，也不再重跑当前 G1、V2 或换公开 initializer 重开 H5。Job
`206205`、Job `206227`、N0、N1、N2、N3 与 N4/N5 均已按各自 gate 关闭。当前唯一行动是
fixed-tool predictability audit 的 pre-formal 代码合同已经完成：strong-baseline ledger、
独立 paired bootstrap、唯一 post-action probe 和 recoverable shard merge 分别绑定在
`daa43c1/19631c8`，真实单行 v2 gate Job `206664` 也已通过。现在用三域较大 opened-train
shards 冻结 throughput、shard count/checkpoint cadence 和总预算；随后运行三个 benchmark
的完整 train/validation。threshold、variant 和 calibration 只在 validation 选择，test 只在
全部选择冻结后打开一次。不得用更多相似 feature、
阈值搜索或扩大模型容量重开已关闭路线；正式 36 cells 完成前不生成终局 verdict。
