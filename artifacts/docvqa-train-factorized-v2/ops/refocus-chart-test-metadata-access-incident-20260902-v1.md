# Refocus_Chart test metadata access incident v1

状态：2026-09-02 18:41 HKT 发现并立即隔离。该事件必须作为 protocol violation
保留，不能被解释为正式 test evaluation 或用于支持方法选择。

## 发生了什么

在 G0 完成后的 public dataset identity audit 中，新写的初版
`scripts/audit_refocus_chart_metadata.py` 为计算 train/test overlap，同时对 pinned
`VTOOL/Refocus_Chart` 的 `train.parquet` 和 `test.parquet` 做了 HTTP range 读取。
这违反了已冻结 protocol 中“第一阶段只用 official train；test 保持封存”的规则。

访问发生在 dataset revision
`00f10ecc5b25d94fd66e14c3671af9fb0f088989`。Test 文件 LFS SHA-256 为
`f2055cd5dd667cfb3c313f22905adb1f536e41e0433839e832f638478ba0c1c3`，包含 826 rows。

## 暴露范围

脚本只读取以下非图像 top-level columns：`id`、`source`、`split`、`data_source`、
`ability`、`agent_name`、`prompt`、`reward_model`、`extra_info`。因此 test question、
ground-truth answer 与 tool metadata 在进程内被读取并转换成 SHA-256 manifest。

明确未读取的 top-level columns 包括 `images`、`edited_image`、`thoughts` 及独立 bbox/
axis columns。终端没有打印 test 问题或答案明文，产物也只保存哈希与 aggregate；但
ground truth 被程序访问这一事实本身已经构成封存违例，不能因“只存哈希”而豁免。

没有进行模型加载、拟合、reward comparison、checkpoint 选择、方法选择、GPU 或
Slurm job。该访问没有影响本项目此前保持封存的 InfographicVQA/DocVQA/
ScreenQA validation、test 或 reserve。

随后一次 one-off original ChartQA lineage 命令还枚举了固定 Git tree 中 train/val/test
的 PNG path IDs，用来检查 Refocus train row IDs 是否跨原始 split；它没有读取图片、
问题、答案或标签，也没有用于方法选择，但属于额外的 protected-split filename metadata
暴露。后续可复现 runner 已改成逐级只遍历 `ChartQA Dataset/train/png`。因此本次
corrected lineage report 的 `protected_split_contents_accessed=false` 只描述该次 runner，
不能抹去更早的一次性 filename access；original ChartQA test 同样不得作为 sealed
formal split。

## 隔离与不可使用边界

初版报告已从正常 dataset-audit 目录移到：

`artifacts/docvqa-train-factorized-v2/incidents/refocus-chart-test-metadata-access-20260902/report.json.quarantine`

- 大小约 11 MiB；SHA-256
  `70475749243d69c5b651fb1882bd7f3ec8f9ff18f81698ee607baa03ccd46625`。
- 该文件保持 ignored/untracked，不加入 commit、不用于任何分析或方法决定。
- 从现在起，Refocus_Chart 官方 test 在本项目中标记为 **contaminated public
  reference**，不得称为 sealed calibration/formal/test，不得用于调参、停止或主结果。
- 后续正式证据必须使用从未打开的独立 benchmark/split，并在读取前冻结协议；不能
  通过重新下载或删除本地报告恢复“sealed”资格。

## 修复

1. Runner 已删除所有 test 参数与读取分支，只能读取固定 `train.parquet`，并在输出
   写入 `test_accessed=false`。
2. 新增 source-level regression test，要求脚本存在 train call、绝不能出现 test
   call 或 `--test-sha256` 参数。
3. 重新运行 corrected train-only audit；正常 report 只包含 official-train manifest。
4. PROJECT_STATUS、EXPERIMENTS 与后续 paper claims 必须披露该 incident；任何
   Refocus_Chart test overlap aggregate 也不得进入候选选择或结果叙事。

事件决定码：`refocus_chart_test_metadata_contaminated_and_quarantined`。
