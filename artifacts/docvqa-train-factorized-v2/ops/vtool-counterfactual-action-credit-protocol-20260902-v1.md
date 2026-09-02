# VTool same-prefix counterfactual action-credit protocol v1

状态：2026-09-02 18:12 HKT 冻结方法与最小验证协议。本文件不授权下载数据/模型、
安装 GPU 环境、打开 validation/test/reserve 或提交 Slurm。数据 manifest、环境
image digest 与 smoke 资源必须在各自 amendment 中绑定后，才允许下一阶段。

## 科学问题与唯一主假设

VTool-R1 当前只用最终任务 outcome 训练 final-answer tokens，并屏蔽产生视觉工具
代码的 action tokens。主假设 H5 是：对同一 policy snapshot、同一 question 和同一
已生成 action prefix，比较 factual visual observation 与冻结 no-op observation 的
最终任务结果，可以得到低方差、保留 rescue/harm 的 signed action credit；把它只
施加到对应 action tokens，能比 outcome-only、zero-credit 和 shuffled-credit 更好地
学习是否及如何调用视觉工具。

本路线不能声称“首个 stepwise evidence reward”“首个 tool-benefit RL”或“首个
decoupled tool/answer objective”。可能的新颖性只限于：具体 visual action 的
same-prefix、arm-matched、cost-adjusted signed effect，以及 action/answer token
分离后的训练和 matched-budget 因果验证。

## 冻结 estimand

对 factual arm `f` 与 counterfactual arm `c`：

`A_visual = (Y_f - lambda * C_f) - (Y_c - lambda * C_c)`

- `Y` 是同一个冻结 judge 给出的最终答案 task score；VTool primary 为二值 0/1。
- `C` 是 arm-specific logical tool cost；primary 中 factual tool attempt 为 1，
  no-op arm 为 0，`lambda=0.05`。
- 因此 rescue、neutral-correct、neutral-wrong、harm 的 primary credit 分别为
  `+0.95/-0.05/-0.05/-1.05`。
- 这个定义在交换完整两臂（包括 score 与 cost）时严格反号；不得只交换 score 或
  在差值外另减一次 cost。
- Primary 不对 `A_visual` 做 batch centering、std normalization 或符号二值化；其
  已有界于 `[-1.05, 0.95]`。这样零点、harm 与稀疏绝对收益不会因 batch 组成改变。
- `beta=1.0`，不在已打开结果上搜索。其他 beta 只能在 primary 通过后作为消融。

## 两臂构造与同 prefix 合同

1. Agent 先正常生成第一段 assistant action；该 token prefix 在两臂完全相同。
2. Factual arm 执行该 action。成功时把 edited image 作为第二幅图；失败时使用原图。
3. No-op arm 始终把原图作为第二幅图。两臂必须使用相同 observation 文本模板、图像
   数量、tokenization、policy checkpoint、final-answer decoding config 与
   continuation seed；只有第二幅图内容允许不同。
4. 两个 continuation 独立 request，但必须保存一致的 `prefix_sha256`、
   `action_sha256`、`target_sha256`、`policy_sha256`、`decoding_sha256`、
   `scorer_sha256` 与 seed，保存两个 `observation_sha256`。Mismatch 直接拒绝，不
   计算 credit。
5. Tool failure 时 edited image 等于原图，因此视觉 effect 为零；logical cost 仍为
   1，primary credit 为 `-0.05`。不得丢弃失败 action。
6. Judge temperature 为 0；异常或 timeout 不得静默记为 0。两臂任一 scoring failure
   都使 pair invalid，并单独报告 failure rate。

## Token role 与 advantage 合同

每条 response 的 valid tokens 必须由三种角色无缝、互斥覆盖：

- `action_mask`：第一段 assistant tool code；
- `observation_mask`：环境插入文本/图像 placeholder；
- `answer_mask`：最终 assistant answer；
- padding 在 valid response 之外，四种 mask 每个位置恰有一个为 1。

Policy mask 是 `action_mask OR answer_mask`；observation/padding 永不训练。无工具
trajectory 的全部 valid assistant tokens 属于 answer，action/observation 均为空。
有工具但缺 final answer、role gap/overlap、越界 span 或 action/observation 单边存在
的 trajectory 一律拒绝。

Primary token-local advantage：

`A_t = A_outcome * answer_mask_t + beta * A_visual * action_mask_t`

`A_outcome` 保持 upstream GRPO 的 prompt-group normalization，只由 factual final
answer outcome 计算；`A_visual` 使用上面的 raw bounded pair effect。KL、old/ref
log-prob 与 policy loss 必须覆盖 action/answer union mask，而 final-answer judge 的
decode 仍只使用 answer mask。必须报告 action token length 与梯度/loss contribution
的关系；只有 primary 通过后才允许 role-balanced loss 作为预注册消融。

## 对照与预算匹配

单一对照无法同时匹配额外 counterfactual continuation 带来的 trajectories、steps
和 GPU-hours，因此冻结两个正交比较，不再声称一个 baseline 同时匹配所有轴：

1. `upstream_outcome_only`：原 VTool mask/reward，匹配初始 checkpoint、train rows、
   actor rollouts、optimizer steps、tool budget 与 seeds；报告额外 walltime/GPU-hours。
2. `paired_zero_credit`：执行并评分完全相同的 factual/no-op pairs、启用相同 union
   mask/KL/entropy，但 action advantage 为 0。它是 implementation/compute control。
3. `paired_shuffled_credit`：同一 pair bank，把 credit 按冻结 trajectory-id 排序后做
   无 fixed point 的循环置换；保持 credit multiset 与全部计算成本，只破坏归属。
4. `paired_signed_credit`：唯一 proposed branch。
5. 另做 `outcome_only_compute_matched` 曲线：让 outcome-only 使用与 proposed 相同的
   累计 GPU-hours；它允许更多 steps/rollouts，因此只用于 compute-efficiency 比较。

Primary 方法证据必须优于 `paired_zero_credit` 与 `paired_shuffled_credit`，并同时报告
相对 upstream outcome-only 的 step-matched 与 GPU-hour-matched 结果。若一个 batch
少于两个 valid tool pairs，则该 batch 的 shuffled action loss 对所有 paired branches
统一跳过并计数，不允许把自身 credit 当 shuffled donor。

## 数据、泄漏与 baseline gate

- 第一阶段只用 VTOOL/Refocus_Chart 的 official train；禁止使用 upstream FP8 脚本
  默认的 `test.parquet`。`trainer.val_before_train=False`、`trainer.test_freq=-1`。
- 下载后先审计 stable row ID、chart/image identity、重复图与 source grouping，再用
  hash 分配 train/curve-eval；manifest、原始/派生 SHA-256 和 split seed 必须在运行
  前写入 amendment。test、validation 与本项目 reserve 继续封存。
- 先复现 pinned upstream 3B outcome-only 的 train-only 极短曲线；若 reward、tool
  attempt/success、loss 或 checkpoint 无法重现稳定趋势，方法分支不得启动。
- Judge、model、dataset 和 image digest 必须固定；judge exception 必须 fail closed。

## 分阶段最小实验与停止规则

### G0：纯 synthetic gate

必须全部通过：role masks 无 gap/overlap；observation/padding advantage 为 0；arm swap
严格反号；no-op cost arithmetic 正确；pair provenance mismatch 被拒绝；序列化 round
trip 不改变 credit；shuffled donor 无 self-assignment 且保持 credit multiset。

### G1：CPU/import 与 4×H800 smoke

先冻结权威 container/image digest，做 import-only 检查；再用 train-only 极小真实
batch、最多 2 optimizer steps，测量显存、吞吐、pair validity、judge failure、tool-call
rate、checkpoint/resume 与每 branch 实际 GPU-hours。若 call rate `<1%`、任一 pair
mismatch、resume 改变 credit identity、judge failure `>0` 或无法在账户 4 H800 内
运行，停止并重新审计，不扩大作业。

### G2：短程 matched-control gate

数据 amendment 根据 G1 吞吐冻结 steps、3 个 seeds 与 curve-eval power；冻结后不因
结果改变。Primary 同时要求：

1. proposed 对 zero 与 shuffled 的 cost-adjusted utility 差在每个 seed 方向一致，
   且预注册 source/seed hierarchical interval lower endpoint 大于 0；
2. task accuracy 不出现预注册 non-inferiority margin 之外的下降；
3. harmful-call rate 降低，rescue rate 不下降；
4. 相对 outcome-only 的 GPU-hour curve 不被额外 counterfactual 成本完全支配。

任一 primary 条件失败则关闭该路线；不得事后改 beta、lambda、pair 构造、judge、
seed、steps、prompt、credit normalization 或挑 checkpoint 挽救。

### G3：仅在 G2 通过后

才允许完整 train、独立 calibration、sealed formal，以及至少第二 benchmark 或第二
backbone。单一 ChartQA/Refocus split、单 seed 或仅训练 reward 上升均不足以支持
ECCV/ICCV/CVPR 主张。

## 当前授权边界

本协议只授权在主仓库实现 dependency-free schema/纯函数与 G0 单测，并更新中文
实验记录。它不授权修改只读 reference clone、安装 vLLM、下载 Hugging Face 数据、
提交 GPU job 或打开任何受保护 split。
