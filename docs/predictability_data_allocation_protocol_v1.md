# Predictability data allocation v1

冻结时间：2026-09-03 18:15（Asia/Hong_Kong）

该分配在任何新 test rollout、feature fitting 或指标计算之前冻结。选择器只能读取 dataset
identity、question type、source/document ID 与解码 RGB hash，不读取模型回答、correctness、
gain、rescue 或 harm。

- ChartQA：已打开的 4,500 official-train states 仅分成 3,600 train 与 900 validation；
  从相同固定 revision 的其余 rows 选择 1,000 个 human/augmented 平衡、RGB 唯一且与所有
  历史 manifest 不重合的 test states。
- DocVQA：已打开的 3,500 train source groups 仅分成 2,800 train 与 700 validation；从
  固定 official validation 选择 500 个未出现在任何历史 manifest 的完整 `docId` groups
  作为 test。一个 document 的全部问题必须在同一 role。
- HRBench：固定使用 8K split 的全部 800 rows，按 source/RGB connected components 以
  60%/20%/20% 分配 train/validation/test。

所有 role 在输出后再次按 source ID 与 canonical decoded-RGB digest 做 pairwise audit；任一
overlap 非零则整个分配失败。test manifests 在此时固定，但在全部模型、阈值和 calibration
冻结之前不得运行或读取 correctness。
