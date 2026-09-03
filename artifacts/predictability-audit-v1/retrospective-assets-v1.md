# 固定工具 predictability：retrospective asset audit v1

时间：2026-09-03 17:47（Asia/Hong_Kong）

## 结论

机械决定为 `retrospective_assets_only_formal_matrix_incomplete`，正式矩阵完成度 `0/36`。
旧资产足以确认二元标签与固定工具成本实现，但不具备终局审计所需的完整特征和 untouched
test，因此以下数字不能用于宣布 `GO/PIVOT/REPRESENTATION/STOP`。

| 旧 bank | decisions | always-call utility | privileged binary-oracle utility | oracle call rate |
|---|---:|---:|---:|---:|
| ChartQA opened test | 2,500 | -0.180800 | +0.028160 | 3.52% |
| DocVQA opened train | 13,580 | -0.194816 | +0.010270 | 2.38% |
| V* opened proxy | 191 | -0.184293 | +0.054450 | 6.81% |

这里的工具始终执行四个冻结 UG-grid crops，以最低 post-action entropy 选答案并支付总成本
`4 × 0.05 = 0.20`。oracle 只在 `gain - 0.20 > 0` 时调用；所有聚合先在 source 内平均，
再跨 source 平均。

## 缺失项

- 三个旧 bank 都没有完整的 max probability、top1-top2 margin 与 L3 frozen-Qwen states；
- ChartQA 旧 2,500 bank 已打开且旧 source ID 不能保证相同像素跨角色隔离；
- DocVQA 是开发/训练资产，不是 untouched test；
- V* 只作小样本 proxy，正式第三 benchmark 已冻结为 HRBench；
- 尚未拟合任何本审计的 predictor，因此 36 个正式 cell 全部未完成。

## 不可变输入与复现

输入哈希由 `configs/predictability_retrospective_assets_v1.json` 绑定。机器报告为
`retrospective-assets-v1.json`，其 SHA-256 是
`14c111b160cdb771e41f55bff93f94fe0e14cf48f49ff89e42976976dd830c0f`。

```bash
PYTHONPATH=src python -m beyond_entropy.predictability_asset_audit \
  --config configs/predictability_retrospective_assets_v1.json \
  --repository-root . \
  --output /tmp/retrospective-assets-v1.json
```

下一步只做冻结矩阵所需的数据/feature/runner，不再从旧 opened outcomes 选择新方法或门槛。
