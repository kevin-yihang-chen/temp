# InfographicVQA fixed-action signed stop OOF 真实输入 smoke v1

日期：2026-09-02（Asia/Hong_Kong）

状态：通过。该 smoke 在完整 OOF 执行前完成，只验证输入、特征与分折合同；
未拟合模型、未生成 OOF score，也未计算任何 policy metric。

- Protocol commit：`e0f95d2ff0ed01b0343530989d1a2b52242ada3d`。
- Implementation commit：`0683526db78c13cdced9eda237b833e9614f54c5`。
- 命令：`scripts/fit_infographicvqa_attention_signed_stop_oof.py --smoke-only`，
  其余参数与冻结 worker 的输入路径及 SHA-256 完全一致。
- 结果：23,946 decisions，2,204 sources，80 features，所有特征有限。
- 类别：1,023 positive-net states，22,923 negative-net states。
- 分折：5 个 whole-source folds；每折的 train/held-out source overlap 均为 0；
  每个 train fold 都包含正负两类。
- 边界：`fit_performed=false`，`policy_metrics_computed=false`，
  `validation_or_test_inputs_used=false`。
- 资源：elapsed `00:58.67`，峰值 RSS `3,678,368 KiB`，4 CPU threads。

工程 smoke 支持在 64 GiB / 45 分钟预算内提交完整 OOF 执行。本记录不包含
任何有利或不利的模型性能信息，也不授权 GitHub push。
