# Compute-budget plan — point-in-time snapshot 2026-08-28

## 2026-09-02 17:42--17:44 HKT live refresh

旧快照中的 42,000 GPU-minute 总额已经失效。当前 live helper 与 association 记录
一致：

- GPU quota：222,000 分钟；已用 42,348；剩余 179,652（2,994.2 GPU-hours），
  使用率 19.08%。
- CPU quota：2,664,000 分钟；已用 200,618；剩余 2,463,382 分钟，使用率 7.53%。
- association 上限：4 GPU、4 H800、48 CPU；查询时当前使用均为 0。
- `squeue -u yihangc` 为空。H800/H100 partitions 有 mixed/allocated nodes，但该状态
  只是一时快照，不保证提交后立即调度。

因此旧的“一次 4 卡 24 小时几乎耗尽余额”判断不再成立。当前配额足以进行有界的
3B matched-control RL、失败恢复和必要的第二次验证；仍不授权无 protocol 的多 seed/
多模型 sweep。每次提交前必须重新查询 live quota，并继续使用全状态邮件。

新的支出顺序：

1. Counterfactual visual-action credit 的 full-text novelty 与 upstream 静态
   feasibility audit 已完成，不使用 GPU；当前结论带 dependency/credit-path
   blockers，不等于训练授权。
2. Protocol 与 synthetic mask/sign tests 已在 G0 通过。下一步先冻结 train-only data
   manifest 与 vLLM 0.17 image digest；再做极小 4×H800 smoke，记录 method、
   outcome-only、zero 与 shuffled-credit 每步真实耗时和峰值显存。
3. 只有短程 matched-control 学习曲线同时改善 task score、utility 与 harmful-call
   rate 才扩完整 3B run。
4. 为 outcome-only matched control、失败恢复、独立 benchmark/backbone replication
   预留至少一半当前余额；不把配额全花在 proposed branch。

## Account limits and current usage

The following values were queried live on `cluster3` while the ChartQAPro
formal job was running. Availability and consumed quota are dynamic and must be
queried again immediately before any new submission.

- GPU quota: 42,000 GPU-minutes total; 33,813 used; 8,187 remaining before the
  current formal job completes (approximately 136.5 GPU-hours).
- CPU quota: 504,000 CPU-minutes total; 154,668 used.
- Account group limit: four GPUs, four H800s and 48 CPUs. No explicit
  association memory limit was shown; 49,152 MiB was the current memory usage.
- No explicit `MaxJobs` or `MaxSubmitJobs` limit was shown. In this Slurm
  output, `N(1)` means no configured limit with one job currently counted; it
  does not mean a one-job limit.
- Current formal allocation: one RTX 4090, four CPUs and 48 GiB.

At the snapshot time, multiple RTX 4090 nodes were idle. H800 nodes were
allocated or mixed rather than fully idle, so a four-H800 request may queue.
This is not a stable availability guarantee.

## Consequences for VTool-style post-training

The VTool-R1 reference reports a minimum four-H100 setup for its 3B training
recipe and a typical training time around 24 hours, with some runs taking one to
two days. Under the current quota:

- `4 GPUs * 24 hours = 5,760 GPU-minutes`, which is feasible once;
- `4 GPUs * 48 hours = 11,520 GPU-minutes`, which exceeds the remaining quota;
  and
- broad multi-seed or hyperparameter sweeps are not feasible.

A separate GPU judge would consume part of the four-GPU account ceiling and
therefore leave fewer accelerators for training. A bounded experiment should
prefer co-locating services, the newer CPU-judge path, or a deterministic exact
scorer where scientifically appropriate.

## Frozen spending order

1. Complete and analyze the current single-GPU ChartQAPro formal evaluation.
2. Do not spend multi-GPU quota if the when-to-call confirmation fails; develop
   and confirm a domain-robust signal first.
3. If the stopping gate passes, validate the VTool adapter with a short,
   checkpointed smoke allocation before the main run.
4. Prefer a resumable 4-H800 run capped at 12 hours initially (2,880
   GPU-minutes). Extend only if reward, tool syntax, checkpoint recovery, and
   matched-control logging all pass.
5. Reserve quota for the outcome-only matched control and final evaluation;
   never spend the whole balance on the proposed method alone.

This budget makes one carefully controlled 3B post-training comparison
plausible. It does not support the many-seed, many-model RL study that a broad
claim would require, so model-scale breadth should come from cheaper frozen
inference/value-head experiments unless additional allocation becomes
available.
