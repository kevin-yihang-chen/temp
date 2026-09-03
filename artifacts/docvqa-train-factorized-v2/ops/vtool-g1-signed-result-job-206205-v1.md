# G1 Job 206205 paired-signed 正式停止审计

状态：2026-09-03 12:31 HKT，Job `206205` 正常完成两步训练与产物保存；冻结分析器
给出 `paired_signed_g1_stop_rule_triggered`。这是 H5 当前 on-policy 实现路线的正式
G1 负 gate，不是工程失败，也不证明“若存在工具动作，signed action credit 必然无效”。

## 假设与不可变设置

- 待检验假设：同 prefix factual/no-op 的 signed、cost-aware action credit 能只作用于
  tool/action tokens，并在真实训练中产生可观测学习信号。
- 代码 revision：`9c6bdc46f60b31d57b11d7a5c95a4712eef5fd44`，提交时工作树干净。
- 配置 SHA-256：`3f2b143850023ecb8e9e7bb79ffcbf3684e736d0f194f0f863d96fa1d055ebb3`。
- 数据：Apache-2.0 official ReFocus train 派生的冻结 72 行 paired set；train SHA-256
  `3a5be076df0df4d1aadf6841fbec40cb265387c43989eab0a46ed092ec9ced46`。
- 模型：`Qwen/Qwen2.5-VL-3B-Instruct` revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`；4×H800，最多 2 optimizer steps。
- 采样、seed、reward、`lambda=0.05`、`beta=1.0`、`<1%` tool-call 停止规则均在结果前
  冻结。validation/test/reserve 未访问。

## 执行与结果

- Slurm 已在控制器清除短期记录前观测到 `COMPLETED`、`ExitCode=0:0`、零 restart；
  12:26:27--12:31:10 HKT，worker 实际 elapsed `261.8567s`。
- 两步均完成 actor update。step 1/2 的 score mean 分别为 `0.5625` / `0.53125`；64 行
  总体 task score 和 realized cost-adjusted utility 均为 `0.546875`。
- 两步工具调用均为 `0/32`，总计 `0/64`，tool-call rate `0.0`，低于冻结的 `0.01`。
  因此没有 factual/no-op tool pair，harmful/rescue/no-effect/tool-success 与 mean signed
  action credit 均不可定义；action-credit tool trajectory count 和 applied credit 均为 0。
- 结构证据完整：64 行、两份 step 文件、10/10 analyzer checks 全真，pair mismatch 和
  judge failure 均为 0，trajectory IDs 唯一，protected split 标记为 false。
- 普通 outcome GRPO 路径执行了更新（step 1/2 actor grad norm `66.4326` / `9.9353`），
  但 H5 特有的 action-local credit 路径没有收到任何支持；不能把普通更新解释为 H5
  方法学习成功。

## Checkpoint 与复现证据

- 唯一 checkpoint 为 `global_step_2`；`latest_checkpointed_iteration.txt` 为 `2`。
- 4 个 model shards、4 个 optimizer shards、4 个 extra-state shards、`data.pt`、FSDP
  配置和 Hugging Face metadata 均存在。文件 payload 合计 45,077,408,354 bytes；目录
  apparent size 为 45,077,408,391 bytes，`du -sh` 约 42 GiB。
- 2026-09-03 已重新顺序读取全部 checkpoint 文件并计算 SHA-256。机器可校验 manifest：
  `vtool-g1-signed-checkpoint-job-206205-v1.sha256`。从仓库根目录运行：
  `sha256sum -c artifacts/docvqa-train-factorized-v2/ops/vtool-g1-signed-checkpoint-job-206205-v1.sha256`。
- 这证明文件集合当前完整且内容被绑定；本次没有另启昂贵 resume job，因此不声称执行过
  checkpoint resume。checkpoint 不加入 Git，manifest 加入 Git。

核心小产物 SHA-256：

- launch manifest：`8a667ef054163a301978de0bd715bc68d0fd0bfe6ffefce024c692a6bfb42cf5`
- execution：`5f1b28095c1e087cc49d2c06e66daa399ab3655134b410fa61e43ad7d52211f3`
- rollout analysis：`d8c4950831e38720b514ca087411d249bfcb431b39bce5c76a0d2736102b6c21`
- rollout step 1 / step 2：`04ba5634e7298186724404ae0b88b1482f9c53358941d605e20775b50a65ae6f` /
  `4b07ae08172f170565f745e13612d17ae8638aa6264bd973acaeab4b0c1ea55b`
- worker status：`22eb7f41cb39591d3ac3f45b1ce18aebd48be4cd9f9d09fe131a9b34d72dd5be`
- Slurm log：`f5def80063c840cafd66309c6febfdf431c7066ebc790218cbf67e786a8facec`

## 科学解释与决策

冻结 G1 协议要求 tool-call rate `<1%` 时停止并重新审计，不能扩大作业。Job `206205`
满足该停止条件。当前 on-policy policy 从初始 checkpoint 不产生工具动作，故 action-local
credit 没有可训练 support；继续运行 paired-zero、paired-shuffled 或 outcome-only controls
不能检验 credit 的因果效果，只会复现“没有 action token”这一边界。因此：

1. 不运行三组 G1 controls，不改 prompt、temperature、seed、threshold 或 call rate 追结果；
2. H5 的当前 on-policy sampled-action 实现路线关闭，不进入 G2、calibration 或 formal；
3. 下一步只做 pre-GPU 新颖性/可行性审计：候选必须在零 on-policy action support 下仍能
   定义可学习信号，并与 ToolVision/ToolsRL/forced-tool SFT/curriculum/off-policy supervision
   明确区分；简单强制调用、SFT 或 tool bonus 不构成新方法；
4. 只有新的 estimand 与算法通过一手文献碰撞审计，并先通过 synthetic/CPU gate，才允许
   新 GPU smoke。

## 资源状态

12:41 HKT 实时 `squeue -u yihangc` 为空。quota helper 为 222,000 GPU 分钟总额、
42,284 已用、179,716 剩余（约 2,995.27 GPU-hours）。checkpoint 落盘后持久盘可用
37,988,859,904 bytes（约 35.38 GiB，97% 已用）。Job 使用了
`--mail-user=yihangc@connect.hku.hk --mail-type=ALL`；Slurm END 属于通知范围，但这里
不声称已验证收件箱投递。
