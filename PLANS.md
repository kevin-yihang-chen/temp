# 研究计划

更新时间：2026-09-03 14:38（Asia/Hong_Kong）

## 总目标与完成标准

目标不是获得一个局部正数，而是形成可投稿 ECCV/ICCV/CVPR 的完整证据链：

1. 与现有 uncertainty guidance、selective VQA、adaptive visual acquisition、
   attention crop 和视觉工具调用工作有清晰区别的新颖命题；
2. 至少一个预先冻结、无泄漏的主结果，优于 entropy、fixed/random crop、
   exhaustive/UG、现有学习式选择器及相关文献强基线；
3. 多数据集或明确的跨域证据，包含消融、风险/校准、置信区间和失败分析；
4. 从干净环境可复现的代码、环境、数据哈希、配置、种子、日志和产物；
5. 论文论证与实验支持一致，不把 exploratory/privileged ceiling 当正式结果。

当前尚未达到上述标准。

## 当前核心判断

InfographicVQA 上，raw attention 的 `where` 信号真实存在，但在相同 entropy
call set 上仍不能产生正净 utility。ViCrop 与 LASER 两个冻结文献 attention
候选也全部失败，且没有显著优于 raw attention。固定 raw action 后的
privileged stopping ceiling 很大，但 attention max/margin 与单一低容量
signed-value OOF stop 都无法利用它。

因此，当前失败不是“没有任何有用 crop”，而是现有 outcome-free representation
无法以足够精度预测哪些状态值得调用。Fixed four-box attention-localization、
entropy/simple-confidence stopping 与当前线性 signed-value stop 家族现已关闭。

原本唯一优先的 answer-conditioned evidence consistency 在实现前完成了代码与一手
文献审计：工程上可以从同一次 baseline generation 复用 hidden states，但
ContextualLens、LRP、VRP、V-Loop 等已直接覆盖 answer/image hidden-state probe、
grounding 与 reliability。该候选因此以
`answer_conditioned_evidence_candidate_rejected_before_experiment` 关闭，没有提交
GPU、拟合模型或读取新 outcome。

## 已完成的关键假设

### H1：文献 attention 方法能进一步改善 where

- 候选：固定 ViCrop Qwen relative attention 与 LASER contrastive all-head bank。
- 特征 Job：`203273`；评估 Job：`203340`。
- 结果：`literature_attention_where_train_not_supported`。两个候选在五个注册
  call rate 的 utility 均为负，Bonferroni-corrected 97.5% lower endpoint 均不
  大于零，也未显著优于 raw attention。
- 结论：关闭对 attention layer/head/ratio 与 fixed four-box attention scorer
  的继续搜索；不进入 calibration。

### H2：固定 raw-attention action 后，主要剩余 headroom 来自 stopping

- Job `203290` 的 unrestricted privileged fixed-action stop ceiling 为
  `+0.021318`，95% CI `[0.018447, 0.024444]`。
- Attention max/margin 在所有注册调用率均差于 entropy。
- 结论：stopping headroom 存在，但简单 attention confidence 不可用。

### H3：固定 raw action 的 signed net value 可在 source-held-out 条件下学习

- Job `203330` 在唯一 2% primary 上得到
  `fixed_action_signed_stop_train_not_supported`。
- Candidate utility `-0.000063`，95% CI `[-0.000739, 0.000655]`；相对 entropy
  改善 `+0.000522`，paired CI `[-0.000304, 0.001444]`。
- 结论：存在弱排序信号，但不允许在已打开 outcomes 上继续搜索 C、特征、权重、
  seed、call rate 或 classifier family。

## 下一主假设与路线选择

### H4：新的 pre-action hidden-state 信息足以形成独立方法

- 可行性：通过；Transformers `5.4.0` 可在 generation output 中返回 hidden
  states，旧 feature contract 确实没有 answer semantics。
- 新颖性：失败；与现有 hidden-state/grounding reliability probes 直接碰撞。
- 决定：实验前关闭，不实现、不提交作业、不把它记成负实验结果。

### H5：same-prefix signed action credit 能改善视觉工具 RL

下一唯一优先研究对象不再是 deployable pre-call classifier，而是对同一 agent
prefix 下 factual visual observation 与 no-op/固定 alternative observation 的最终
任务差定义 signed、cost-aware `A_visual`，并只分配给对应 tool/action tokens。
Final-answer/reasoning tokens 继续使用 outcome reward。

候选必须同时满足：

1. 与 VTool-R1 outcome-only、ToolVision committee evidence/MUT、AdaTooler-V
   query benefit 和 AdaptVision decoupled objective 明确区分；
2. 不是 question-level necessity label，而是具体 action/observation 的同 prefix
   signed rescue/harm/cost contrast；
3. 在 upstream 第一段 tool tokens 当前被 `response_mask=0` 的前提下，新增独立且
   可审计的 action mask/advantage 通路；
4. 先通过 synthetic sign/mask/unit test，再做极小 4×H800 smoke；
5. paired zero/shuffled controls 必须复用相同 pairs、mask、steps 与计算；另分别做
   outcome-only 的 trajectory/step-matched 和 GPU-hour-matched 比较，不虚构一个
   baseline 能同时匹配互相冲突的全部预算轴；
6. 短程学习曲线需同时改善 task score、cost-adjusted utility 与 harmful-call rate，
   否则不扩完整训练。

G0 当前已通过：protocol v1 冻结 arm-specific net-utility contrast、`lambda=0.05`、
`beta=1.0`、raw bounded action credit、zero/shuffled/outcome-only controls 与泄漏边界；
dependency-free core 已实现 action/answer/observation/padding masks、pair provenance、
token-local advantage、序列化与 deterministic derangement。G0 实现 commit 为
`56b990c767973a8a23060d63293db8657254b35d`；upstream-shaped adapter/overlay 的
pre-GPU contract 已通过。此后 Job `206205` 已完成真实 rollout、两步普通 outcome
GRPO update 与完整 checkpoint，但 64 条 trajectory 中没有 parser-valid、可执行的
工具动作，因此 H5 特有的 action-local credit 通路从未激活，不能据此获得方法性能证据。

G0 后已做范围纠偏：VTool 只作为 Apache-2.0 的可运行 RL 骨架和 outcome-only
comparator，不再审计 thought、pixel 或内部实现是否与 VTool 等价；该问题与 H5 的
成败无直接关系。训练数据改为固定 revision/hash 的 Apache-2.0 official ReFocus train。
token-local autograd、隔离 runtime import、official-train converter/processor、paired
agent fake-server contract、单卡 H800 vLLM model-load/真实首轮 generation、72 行完整
运行时数据审计、JSON-safe counterfactual rollout export、自动 stop-rule analyzer 与
最终 Hydra resolved-config gate 均已通过。四卡 Job `206179` 暴露并修复 DP batch
chunk 类型合同；Job `206184` 暴露并修复 checkpoint 空间 gate；最终 Job `206205`
完成两步运行。其正式结果为 tool call `0/64`、rate `0.0`，冻结分析器给出
`paired_signed_g1_stop_rule_triggered`。这关闭当前 sampled on-policy H5 路线，不进入
matched controls 或 G2。

独立 typed-action V2 的 CPU/真实 executor gate 随后通过，但唯一 H800 generation
smoke（Job `206227`）得到 11/16 tool intent、7/16 完整且 syntax-valid Python fence，
参数合同、strict parser 与 execution 均为 0/16。11/11 有意图输出都逐字复制 prompt
里的无效元变量函数名 `focus_on_x_values_with_MODE`，机械决定为
`typed_action_b0_malformed_tool_intent`。因此 V2 baseline 自身关闭，不改 prompt/seed 后
重跑；这个结果不改变 G1 stop，也不能充当 N0 方法证据。

N0 action-boundary interventional objective 的形式化与 dependency-free 零支持 gate 也
已完成。完整干预期望效用的精确梯度仍乘当前宏动作概率，在零/近零 support 下消失；
改用 utility-induced target 可恢复梯度，但就成为 LIRE/LiPO/AWR 类 listwise/off-policy
投影；单独回归 `Q(s,a)` 则回到 GapSight/既有 action-value router 家族。数值 gate
10/10 checks 全真，一手文献还确认 ToolVision 已用 evidence gain/paired MUT 建立 support，
The Illusion 已定义 fixed-prefix observation intervention。N0 因
`action_boundary_candidate_reduces_to_existing_objective_families` 在 GPU 前关闭。

N1 现有 sibling-bank 盘点也已完成。四个主开发 bank 跨 InfographicVQA、ScreenQA、
DocVQA、TextVQA，共 `59,949` decisions、`299,745` rows；完整 sibling、source IDs 与
不可变 provenance 均通过。因此 stop regret 和已注册四个 UG-grid 候选内的
action-selection regret 可识别。但 `239,796` 条主 ZOOM 记录中没有一条保存 fixed action
prefix、匹配 factual/counterfactual observation 与受控 continuation，evidence-use regret
不可识别。主证据还只有一个 ZOOM/UG-grid 动作族、每状态一个 replicate，且 3B/7B 与
数据集混杂；唯一同数据集多 backbone 只有 ScreenQA 的 512-state opened-development
diagnostic。N1 以 `n1_existing_assets_insufficient_for_top_tier_regret_benchmark` 关闭，
不把规模当成完整因果 benchmark。

2026-09-03 00:56 HKT，重提的 paired-signed Job `205902` 获得资源后在 worker
前置检查中同秒退出。原因是 shell 中的 jq 对象全真断言误写为
`.checks | all(.[] == true)`，当前 jq 会把单个布尔值再次送入 `.[]` 并以 exit 5
失败。没有创建训练输出目录，也没有 rollout、模型加载、optimizer step 或
checkpoint；因此这不是 H5 的正/负结果。两处同类断言已改为二参数形式
`.checks | all(.[]; . == true)`，并加入真实 jq 正/负回归。下一次重提前必须完成
全量回归、重新生成最终 Hydra gate 并以新 commit 绑定 worker 哈希。全量回归现已
通过；修复 commit `8c0f6c010a4dfeb1bf01d955054da2287691896e` 上的 Hydra v13
也以 59/59 checks 全真通过，launch manifest/resolved config SHA-256 分别为
`29e24dab...8fe8` / `cda71307...ac43`。当时计划是在最终 HEAD 再跑一次同一 gate，
然后实时复核资源并只重提 signed arm；该步骤后来已由 Job `206205` 完成。

2026-09-03 12:31 HKT，paired-signed Job `206205` 在 clean revision `9c6bdc4` 上
正常完成。两步 task score 为 `0.5625` / `0.53125`，总体为 `0.546875`；正式
parser-valid tool call `0/64`。两步 action-credit tool trajectory count 与 applied
credit 均为 0；结构审计 10/10 全真，pair mismatch/judge failure 为 0，唯一
`global_step_2` checkpoint 完整且全部文件已做 SHA-256 绑定。由于 `<1%` 是结果前冻结
的停止规则，当前路线必须停止，不能通过改 prompt、seed、temperature 或阈值追结果。

2026-09-03 13:11 HKT 的只读 raw-response 诊断纠正了“全部 direct”的过强描述：
48/64 以 `FINAL ANSWER:` 开头，16/64 以 `focus_on_*` 开头；后者 step 1/2 分别为
12/32 与 4/32。15 条可解析为单个 focus expression，但 0 条符合真实三参数 API，另
1 条混入最终答案而语法无效，所以只补 Python 围栏可恢复的调用仍为 0。正式 G1 stop
不变；新信息是 baseline 同时存在 prompt/API 格式支持缺口，不能再等同“无 latent tool
intent”。机器报告与路线审计见 `vtool-g1-intent-format-posthoc-job-206205-v1.json`
和 `vtool-g1-format-contract-and-next-route-audit-20260903-v1.md`。

### 顶会约束

所有后续候选仍必须满足：

1. 引入可解释的新信息来源或 action proposer，而不是 attention 层/head、阈值、
   线性 head 或 call rate 的变体；
2. 明确分离 `whether/when` 与 `where`，并对每个具体 action 的 signed rescue/harm
   保留完整 sibling supervision；
3. 在任何结果读取前冻结唯一候选、调用成本、source split、primary endpoint、
   强基线与停止规则；
4. 先做小规模真实输入 smoke 和成本审计，再决定是否值得 GPU 完整运行；
5. official-train 只作 exploratory/source-OOF screen；validation/test/reserve 继续
   封存，只有严格 train gate 通过才允许新 calibration 协议。

路线优先级：

1. Exact typed-action V2 baseline 的 CPU/runtime gate 与唯一 H800 generation smoke 已
   完成。V2 因 11/11 tool intents 复制无效 `MODE` 元变量、0/16 参数合法而关闭；不在
   已打开 row/seed 上改 prompt 重跑。若做 V3，只能以全部 concrete、parser-valid 的六个
   函数模板、独立 structural group 和新种子预注册为一次新的 baseline correction；
2. N0 action-boundary interventional objective 已因零支持/目标族 gate 关闭：不提交 GPU，
   不把 token-local mask 或 same-prefix effect 包装成新方法；
3. N1 现有资产路线已关闭：前两项 regret 可识别，但 evidence-use、同数据集多 backbone、
   多动作族与随机重复四项 gate 失败；
4. 下一候选 N2 只做严格可加 regret 分解的新颖性/可识别性证明与最小 factorial
   augmentation 的 sample/算力/存储审计。它必须同时区别于 The Illusion 的 fixed-prefix
   evidence gain 与 GapSight 的 crop-loss router；通过前不生成新 intervention data。

## 止损规则

- 不再运行 attention layer/head/ratio、entropy threshold、call-rate、随机种子或
  线性 classifier-family sweep。
- 不用 validation/test 帮助选路线，不把 privileged oracle 当部署结果。
- 下一候选必须先写 protocol，再实现，再 smoke；没有能区分科学假设的新信息时
  不提交 GPU job。
- 不再对 answer hidden-state/grounding probe、generic group-DRO/IRM 或 conformal
  threshold 进行局部变体搜索。
- 当前 sampled on-policy action-credit 路线已因 Job `206205` 的零 parser-valid 工具调用
  正式关闭；
  不运行无法区分 credit 效果的 zero/shuffled/outcome-only controls。
- 新候选若在文献审计或最小 gate 失败，继续选择实质方法/benchmark contribution；
  不以降低投稿目标作为完成条件。

## 紧接着的行动

1. Answer-conditioned candidate 已因文献碰撞关闭；VTool 仅保留为运行底座和
   outcome-only comparator，停止所有进一步 equivalence 审计。
2. Matched-control action-credit protocol 与 G0 synthetic implementation/tests 已完成；
   这只证明 arithmetic/schema，不是方法有效性证据。
3. 官方 Apache-2.0 ReFocus train 的 revision/shard hash 已固定；只从 official train
   构建 structural-group-disjoint 的最小 G1 数据，不使用旧 derivative 或 protected split。
4. token-local adapter、真实 autograd、pinned runtime、单行真实 processor 和 paired
   fake-server contract 已通过；两臂 shared prefix/seed、image-only delta、role mask、
   rescue/harm/failure/direct credit 均有运行证据。
5. outcome-only、paired-zero、paired-shuffled、paired-signed 四组配置，以及 task score、
   cost-adjusted utility、harmful-call rate 和 tool-call-rate stop rule 已在
   `configs/vtool_action_credit_g1_v1.json` 冻结。
6. 单卡 H800 model-load/单条 generation smoke（Job `205784`）已通过；72 行 frozen
   train 已全部经过真实 `RLHFDataset` 与 Qwen processor，prompt 最大 1,914 tokens，
   row/data/provenance 全部匹配。最终 Hydra dry-run v9 的 59 项 scientific/resource
   contract 全部通过。
7. 首次提交的 Job `205870` 在排队期间发现 rollout JSONL 只会保存 `acc`、无法导出
   counterfactual utility/harmful-call 证据，因此在启动前取消；`RunTime=00:00:00`，
   没有消耗 GPU。现已加入稳定 audit JSON、pair/score/utility analyzer 与 worker
   自动 gate，科学配置未改变。
8. 重提的 Job `205902` 在 2026-09-03 00:56 HKT 获得资源，但因 worker 的 jq
   object-values 断言语法错误同秒 fail closed；没有创建输出目录或产生科学结果。
   根因已由相同 jq/相同 72 行 audit report 稳定复现，并已修正两处同类断言。
9. jq 修复后的 Job `206170` 已真正进入四个 FSDP actor 初始化，但 pinned runtime
   默认强制 `flash_attention_2`，隔离环境没有 `flash_attn`，四个 rank 在权重加载前
   以相同 ImportError 退出。没有 rollout、optimizer step、checkpoint 或 H5 指标。
   进一步源码诊断发现，只覆盖为 SDPA 仍不充分：`use_remove_padding=true` 会把 Qwen
   attention forward 再替换回自定义 FlashAttention 路径。因此所有实验臂共同冻结为
   `attn_implementation=sdpa` 且 `use_remove_padding=false`；这属于首次科学结果前的
   对称运行时修复，不改变方法或比较定义。
10. commit `22b89a5ca872356c621203f7bb724042c846a091` 已加入 fail-closed 配置审计和
   单 H800 真实图片 actor-load/forward smoke。Job `206174` 随后以 `COMPLETED`、
   `ExitCode=0:0`、50 秒、零 restart 通过：完整 Qwen 权重与 verl 多模态 patch 在
   SDPA/no-remove-padding 下对 966-token 真实图片输入完成前向，6/6 checks 全真，
   峰值已分配显存约 7.42 GiB。报告已由 commit
   `0122689050a4bfc91df0a55dd154dc52d7fce83d` 绑定进 G1 launcher。下一步在最终
   clean revision 复跑 gate，再重新提交唯一 4×H800、最多 2 optimizer-step 的
   paired-signed G1。只有实际
   tool-call rate、pair validity 与训练稳定性通过冻结 stop rules，才以同一 revision
   顺序运行 zero/shuffled/outcome-only controls；失败则按预注册规则停止，不调整
   prompt、seed、temperature 或阈值追结果。
11. 四卡 Job `206179` 已越过此前 backend 阻塞，到达首个 `_update_actor`，随后因
   action-credit adapter 把 donor trajectory IDs 作为 Python list 注入而在 verl
   `DataProto.chunk()` 失败。该 runtime 要求全部 `non_tensor_batch` 值为 ndarray；
   同一 runtime 的最小复现已从断言失败变为 4-way chunk 7/7 checks 全真。commit
    `ea489f7c520880ab087af761a620b03f357b18e0` 修复类型并把 exact CPU smoke 加入
    submitter/worker。科学配置和 stop rules 未变；最终 clean Hydra gate 后允许一次重提。
12. 重提的 Job `206184` 已完成 32 行 step-1 rollout 并进入最终 checkpoint 保存，
    但原 32 GiB 空间 gate 低于实测至少 40.39 GiB 的 checkpoint shards，持久盘写满后
    无法写出 step-2 rollout、完整 checkpoint 或 analyzer。step-1 结构审计 10/10
    checks 全真、task score `0.53125`，但 parser-valid 工具调用为 0；事后 raw 文本为
    19 条 final answer 与 13 条裸 focus intent。这只提高正式 stop 风险，不能
    替代两步判定。不可恢复 checkpoint 与可重建 Arrow cache 已按用户授权清理，当前
    空间约 77.1 GiB；资源合同提高为 submitter/worker 双重 64 GiB fail-closed gate。
    当时计划是验证并 clean commit 后只重提 signed arm；该重提已完成。
13. Job `206205` 已完成两步、64 行 rollout、正式 analyzer 与约 42 GiB 的唯一
    `global_step_2` checkpoint。两步 score 为 `0.5625` / `0.53125`，但工具调用为
    `0/64`，正式 decision 为 `paired_signed_g1_stop_rule_triggered`。checkpoint 和核心
    小产物均已做内容哈希审计。
14. 按冻结 G1 规则关闭当前 on-policy H5：不运行三组 controls，不改 seed/prompt/
    temperature/threshold。下一步先完成零支持条件下新算法的一手文献碰撞与可行性审计，
    审计前不提交 GPU。
15. 事后格式诊断已证明 Job `206205` 有 16/64 裸工具意图，但 0/16 可仅补围栏执行；
    进一步发现 V1 prompt 声称可用的 `x_values_bbox/y_values_bbox` 不在实际 execution
    context；这不改变 G1 stop，却要求新路线先建立 typed-action V2 reliable baseline。
16. Prompt/SFT/forced-call、representation steering、logit bias、ordinary listwise reward
    和 crop loss-gap router 均已有直接文献覆盖，只能作 baseline。唯一暂存主方法候选是
    action-boundary interventional objective；若形式化后退化为上述目标，则在 CPU gate
    关闭。
17. Typed-action B0 完整 CPU/真实 runtime gate 已在 commit `47fde37` 通过。独立 V2
    official-train 单行 Parquet SHA-256 为 `2c6a6c9b...e8184c`，与 V1 使用同一 row/image
    以隔离 prompt 变化；旧 V1 Parquet 仍字节级复现为 `0de5b142...66199`。真实 Qwen
    processor 得到 975 prompt tokens，26/26 checks 全真；renderer-owned action 经 strict
    parser 后在 pinned runtime 显示一个像素发生变化的 PIL image。没有模型权重、optimizer、
    checkpoint 或 protected split。
18. 唯一 V2 H800 smoke（Job `206227`）已完成，0 optimizer、0 checkpoint。16 次生成中
    intent/fence/syntax/argument/parser/execution 分别为 11/7/7/0/0/0；11/11 intents
    都复制无效 `_with_MODE`。V2 不再重跑，完整负结果和 raw outputs 原样保留。
19. N0 形式化/数值 gate 已完成并关闭：直接 expected-utility gradient 在零支持处为零；
    非零替代退化为 listwise/AWR/value-router。没有提交 GPU 或打开新 outcome。
20. N1 流式机器盘点已完成：四主 bank `59,949` decisions / `299,745` rows，完整性与
    provenance 通过；stop/selection 可识别，evidence-use 不可识别，且动作族、replicate、
    同数据集主 backbone 因子不足。现有-assets benchmark 路线关闭。
21. N2 在 CPU/纸面上先形式化严格可加的 stop/selection/prefix/evidence decomposition，
    做一手文献碰撞，并计算最小 factorial augmentation 的统计功效和资源上界；未同时通过
    新颖性、识别与成本 gate 前不提交 GPU。
22. V3 如有必要只承担 concrete-template 强 baseline，不作为新颖贡献，且必须使用独立
    row/seed 预注册。其他 validation/test/reserve 继续封存；本地修改不 push GitHub。
