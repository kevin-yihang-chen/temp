# G1 Job 206184 checkpoint 存储失败审计

时间：2026-09-03 12:16（Asia/Hong_Kong）

## 结论

Job `206184` 不是 H5 的正式正/负结果。它完成了可审计的 step-1 rollout，并在
第二步的 checkpoint 保存阶段耗尽持久磁盘；由于 upstream 在 checkpoint 之后才写
当步 rollout，`rollouts/2.jsonl`、完整 checkpoint、tracker 与最终 analyzer report
均未落盘。因此冻结的两步 stop rule 不可计算，禁止用单步结果替代正式判定。

## 执行与终态

- 代码 revision：`83f6e53fe1fdd4d83e4d5cd1bae7a0ab00e02b80`。
- 资源：4×H800、48 CPU、384 GiB、2 小时上限，`--mail-type=ALL`。
- 时间：2026-09-03 11:27:40--11:32:35 HKT；此前实时 Slurm 查询记录为
  `FAILED`、`ExitCode=1:0`、`RunTime=00:04:55`、零 restart。当前 controller 已
  清除该短期 job record，`scontrol show job 206184` 返回 invalid job id。
- 当前队列为空；12:09 HKT quota helper 为 GPU 222,000 分钟总额、42,268 已用、
  179,732 剩余。

## 持久证据

- `rollouts/1.jsonl`：32 行，SHA-256
  `a2118345d04633d235659812ba5319284131c3a5b782ef4575f576598ad9e75c`。
- Slurm log：SHA-256
  `7e88271055f5ead7b5ca641706f6099b81efebbd20ab23d6dd3aba376e4f6722`。
- Launch manifest：SHA-256
  `f8e6383718bb74c12f618606314e8cfce7a1f0ec828f9ade3c99062aa45d894b`。
- 当前 job 目录还保留 launch manifest、step-1 rollout 与空 Hydra log；没有
  `rollouts/2.jsonl`、`rollout-analysis.json`、`execution.json` 或 worker status。

## 磁盘根因

提交前只要求 32 GiB 空闲，但实测不完整 checkpoint 已写入：

- 四个 model shards：每个 3,755,065,659 bytes；
- 四个 optimizer shards：6,945,898,496 / 7,189,561,344 /
  7,062,945,792 / 7,148,797,952 bytes；
- 上述已知文件合计 43,367,466,220 bytes，即约 40.39 GiB；尚未写完
  extra-state、dataloader state 与 tracker。

所以 32 GiB gate 在数学上必然不足。upstream 只在最后一步或 save frequency 命中时
保存，本配置 `total_training_steps=2`、`save_freq=2`、
`max_actor_ckpt_to_keep=1`，每个实验臂只产生一个 `global_step_2` checkpoint；四个
model/optimizer shard 只是同一 distributed checkpoint 的组成部分。

用户授权后，已永久删除该不完整且不可恢复的 checkpoint 目录，并删除约 37 GiB
可重建的 Hugging Face Arrow dataset cache。Qwen 模型缓存、Hub source blobs、正式
artifacts、step-1 rollout 与日志均保留。清理后持久盘可用 80,854,016 KiB，约
77.1 GiB。

## 单步诊断与解释边界

使用现有 analyzer 的相同 schema/pair/score 合同，仅把临时诊断 config 的期望步数
设为 1，对已持久化的 32 行做只读审计：10/10 checks 全真、pair mismatch 0、judge
failure 0、task score mean `0.53125`，但 tool-call count `0`、rate `0.0`。

这说明初始策略在第一批没有产生 action-credit token，显著提高最终 `<1%` stop 的
风险；但正式协议要求 step 1/2 合计 64 行，第二步只要至少一次工具调用，aggregate
rate 即为 1.5625%。因此当前科学 decision 必须保持 `not_available`，不得把临时
analyzer 的单步 stop 输出当作 H5 结论，也不得事后修改 prompt、temperature、seed
或阈值。

## 修复与下一步

资源合同新增 `minimum_free_persistent_disk_gib=64`，submitter 在排队前、worker 在
模型加载前各自读取同一冻结字段并 fail closed。64 GiB 相对已观测 40.39 GiB 已知
shards 留约 23.6 GiB 余量；当前 77.1 GiB 可用空间通过该 gate。该 amendment 不改变
数据、模型、prompt、reward、credit、sampling、seed、训练步数、controls 或指标。

完成测试、clean commit 与最终 Hydra/DataProto gate 后，只允许重提 signed arm。
若两步正式结果触发工具调用率或其他冻结 stop rule，则关闭 G1，不运行 controls；
只有通过后才处理后续三个实验臂的独立 checkpoint 存储，不允许让多个约 41 GiB
checkpoint 无界累积。
