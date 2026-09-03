# Predictability 36-cell synthetic smoke v1

时间：2026-09-03 17:58（Asia/Hong_Kong）

本项只验证代码路径，不包含真实 benchmark 证据。报告明确设置
`formal_claim_eligible=false`，不会增加正式矩阵完成计数。

- synthetic benchmark：ChartQA/DocVQA/HRBench 三个名字各一套独立生成数据；
- train/validation/test：每个 benchmark 为 40/20/20；source 与 RGB SHA-256 overlap 为零；
- 矩阵：4 predictor levels × 3 targets × 3 benchmarks = 36/36 cells；
- seeds：17、29、47，共 108 个 seed-runs；
- 每个 run 均执行 train-only fit、validation-only variant/calibration/threshold、冻结 test
  evaluation、policy curve 与 paired source bootstrap；
- strict JSON 最终成功写出，报告 SHA-256
  `a458cad9f835b05577f62aec292ff5dc40df4ec0eda1a37b393d21cb7f93c0f9`。

首次完整 smoke 的训练已结束，但 strict writer 因单类 AUROC/AUPRC 为 NaN 拒绝序列化。
修复仅把不可定义的单类指标编码为 JSON `null`，未改数据、模型、seed 或其他指标；随后
原设置重跑通过。小样本 MLP 多次达到冻结的 500 iteration 上限，warnings 保留为正式运行
需要显式记录的 convergence 诊断，不允许因此事后改变网络或训练上限。
