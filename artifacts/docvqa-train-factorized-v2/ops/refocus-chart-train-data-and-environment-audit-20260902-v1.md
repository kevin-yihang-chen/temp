# Refocus_Chart train 数据与 vLLM 环境审计 v1

状态：2026-09-02 19:03 HKT 完成 CPU/网络只读审计；**G1 尚未授权**。本记录不把
metadata/schema 可读取误写为完整数据可合法使用，也不把镜像存在误写为本集群已能
运行。

决定码：`g1_not_authorized_pending_dataset_license_pixel_identity_and_runtime`。

## 审计边界

- VTOOL/Refocus_Chart 固定 revision：
  `00f10ecc5b25d94fd66e14c3671af9fb0f088989`。
- VTool training-v2 固定 commit：
  `d2aa28353ec10c7f91b39f502925003a81d6982d`。
- 本次 corrected audit 只读取 public `train.parquet` 的指定非图像 columns；不读取
  image/edited-image/thoughts columns，不加载模型，不安装依赖，不提交 GPU/Slurm。
- 原 ChartQA lineage runner 只对 ancestor trees 做非递归遍历，再递归读取
  `ChartQA Dataset/train/png`；不遍历 validation/test subtree 内容。
- 先前误读 Refocus_Chart test metadata 的事件单独保留在
  `refocus-chart-test-metadata-access-incident-20260902-v1.md`。该 test 已永久降格为
  contaminated public reference；one-off lineage 命令还曾枚举 original ChartQA
  val/test PNG path IDs。Corrected runner 的 train-only flag 只描述本次执行，不恢复这些
  split 的 sealed 资格。

## Pinned 数据身份

Hugging Face dataset API 在审计时返回：

- repo public、not gated；dataset card/license 字段为空，repository tree 只有
  `.gitattributes`、`train.parquet`、`test.parquet`，没有 README/LICENSE；
- `train.parquet`：1,095,790,683 bytes，14,344 rows，LFS SHA-256
  `d7972ca232aa9c0646af387f7dffb987528b99b3d9693ccd58bbef0463f2d4e1`；
- `test.parquet`：29,191,788 bytes，826 rows，LFS SHA-256
  `f2055cd5dd667cfb3c313f22905adb1f536e41e0433839e832f638478ba0c1c3`。

Corrected train-only report：

- 路径：
  `artifacts/docvqa-train-factorized-v2/dataset-audit/refocus-chart-metadata-v1/report.json`；
  report SHA-256
  `0e62d74948a6e8bedb19a52c814f7ee3aa0afac53311ef9736b23c2b469742ba`；
- schema `refocus_chart_train_metadata_report_v1`，`test_accessed=false`；
- 14,344 rows，14,344 unique row IDs，0 duplicate row-ID rows；
- 10,806 unique structural signatures，3,538 rows 落在重复 signature，最大 group 59；
- exact question duplicate rows 141，question-answer duplicate rows 23，prompt duplicate
  rows 141；source 为 `h_bar=4,722`、`v_bar=9,622`；全部 tool name 为 `refocus`；
- manifest SHA-256
  `a034f7fd1d3492950faa0d079a6b2da58e86742bee3ed3696a8c657b0c19677f`。

Structural signature 只绑定 source、axis values/bboxes 与 figure bbox，并排除
question-specific focus bbox。相同 signature 按潜在同 chart 保守分组；不同 signature
**不能**证明像素不同。因此它可用于避免明显 group leakage，但不是 pixel-identity
certificate。

## Original ChartQA train lineage

原仓库 `vis-nlp/ChartQA` 固定 root tree
`044eabfc306abfe9340c5741f0093aefc5973d06`。安全逐级解析得到：

- `ChartQA Dataset` tree：`0a4076a03da688e572a70af7eeac9681455deceb`；
- `train` tree：`1caf945c1d48b91c0a571e4dad18dbaadc7673fc`；
- `png` tree：`1869eec4dabbe2dc9d582809fb80f5c1d70a9af9`。

Lineage report：

- 路径：
  `artifacts/docvqa-train-factorized-v2/dataset-audit/refocus-chart-metadata-v1/lineage.json`；
  SHA-256 `15189ebb6128900c684ffc3cd7b07838a802a4eba88a42353b3ddb3b9dca0f6c`；
- original train tree 有 18,317 个 unique PNG stems；Refocus train 的 14,344 个 unique
  row IDs 全部一一命中，missing=0；
- 决定：
  `all_refocus_train_row_ids_match_pinned_chartqa_train_png_stems`。

这支持“Refocus train 的 row-ID lineage 来自 original ChartQA train”，但不证明 Parquet
中的 image/edited-image bytes 与原图相同，也不自动把原数据许可传递为派生数据的明确
许可。

## License 判定

- VTool-R1 code repository 声明 Apache-2.0；该 code license 不能自动视为 hosted
  `VTOOL/Refocus_Chart` dataset license。
- Refocus_Chart repo 没有 dataset card/license，故当前无法从一手发布物确认派生
  Parquet 的下载、训练与再分发条款。
- Original `vis-nlp/ChartQA` repository 带 GPL-3.0 LICENSE；它能说明原发布物的条款，
  但不能补写 Refocus_Chart 发布者没有声明的派生数据许可。

所以当前不下载完整 1.10 GB train Parquet、不开始训练。解除方法只有两类：获得发布者
明确的数据许可说明；或从条款明确的 original ChartQA 与可审计代码重新生成所需训练
输入，并记录生成链和文件 SHA。不能用“公开可访问”替代 license evidence。

## vLLM 0.17 环境身份与本集群可执行性

Pinned upstream 的 `docker/Dockerfile.stable.vllm` SHA-256 为
`98197da9628e5ad2c886b0a89aa4e3442d5bdd7e3c2785f3b3daa139bff61ce9`。它声明 CUDA
12.9.1、Python 3.12、PyTorch 2.10.0、vLLM 0.17.0，但 base image 使用 mutable tag，
且 Apex、MBridge 等多项 Git dependency 没有固定 commit。与此同时：

- `setup.py` SHA-256
  `55c2b29c939b702d272a4a4d92780b49ebbd9335d4314f35b43428452f98c43f`，要求
  `vllm>=0.8.5,<=0.12.0`、`torch==2.9.1`；
- `requirements.txt` SHA-256
  `8a181e893340a23a41d4d982419a134873fa23d58b31f9178094f881bc61a453`，只保留
  `# vllm==0.8.4` 注释；
- FP16 recipe SHA-256
  `20e8f4114c7a9f7b31de3cdc86fccccb513f7f9d2b12319ed80b8bbc599a7365`；FP8 recipe
  SHA-256
  `30378ff5b09987800d44e4069625cefa5877a9989736c6fc4b618d26d4dae75d`。

官方 Docker Hub 的 `verlai/verl:vllm017.latest` 在审计时解析到 linux/amd64
manifest digest
`sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6`，compressed
size 14,356,458,058 bytes，last pushed 2026-03-12。后续若使用它，只允许引用
`verlai/verl@sha256:...`，禁止再用 mutable `latest` tag。

但是当前本集群还不能把“已找到 digest”当作 import gate 通过：

- login environment 没有 docker、podman、apptainer、singularity、enroot、skopeo、
  umoci、runc/crun；Slurm CLI 虽显示 `--container` OCI bundle 参数，但 live config 的
  `JobContainerType=(null)`，本地也没有已展开 bundle；
- 现有 base/hulumed/llava-med/qwen-vl/qwen-vl-dola Conda 环境没有 vLLM 或 Ray；四个
  named env 只有 PyTorch 2.4.0+cu121，Transformers 分别为 4.37.2、4.51.x 或 5.4.0；
- upstream 默认 `/verifier-agent/models/Qwen2.5-VL-3B-Instruct` 与 judge
  `/vllm-openai_v0.17.1.sif` 均不存在；
- home filesystem 只剩约 50 GiB、使用率 95%。14.36 GB compressed image 解包后会
  显著逼近剩余空间，未经独立空间规划不能下载或展开。

## G1 准入与下一步

当前已通过：train schema/manifest、row-ID uniqueness、original-train row lineage、
published image digest identity。当前未通过：derivative dataset license、pixel-level
identity、cluster runtime/import、model/judge digests。

因此不能提交 4×H800 G1。信息价值最高的下一步是：

1. 优先解决数据许可；若无明确答复，改走 original ChartQA 可审计再生成路线；
2. 在不碰 Refocus test 的前提下，为 train bytes 做 pixel hash/group manifest；
3. 与集群已有 OCI/SIF 设施对接，或在有足够 scratch 的位置从上述 immutable digest
   构建只读 runtime；先 import-only，再做单 GPU/单 batch，无条件通过后才扩大到
   protocol 规定的最多 2 optimizer steps；
4. G1 amendment 必须在结果读取前固定 model/tokenizer/judge digest、source-grouped
   train/curve-eval manifest、seeds、命令、邮件参数、checkpoint/resume 与停止规则。

本审计没有生成任何效果指标，因此既不是方法成功，也不是方法失败。它把当前最大失败
风险从模糊的“可能跑不起来”收敛为三个可检查条件，避免直接花费 GPU-hours 得到不可
发表或不可复现的结果。
