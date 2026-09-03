# N3 公开 tool checkpoint 与独立新颖性联合 gate

时间：2026-09-03 15:04（Asia/Hong_Kong）

## 结论

决定为 `n3_public_initializer_exists_but_joint_gate_failed_before_download`。公开且许可
清晰的 tool-trained VTool 权重确实存在；若只考虑下载成本与模型族兼容性，唯一优先候选
是 `VTOOL/VTool-Qwen2.5-3B`，固定 revision
`0ca11e812287b5c024c7277db71859da5bda17ac`，远端占用 `8,143,089,840` bytes。
但是，当前 artifact 没有给出与现行 `training-v2` runtime 对应的精确 checkpoint 映射、
prompt/parser contract 或可逐条审计的 parser-valid execution trace。更重要的是，H5 拟
主张的四个核心组成已分别被 TACO、TAPO 与 The Illusion 覆盖；ToolVision 又覆盖了
with/without-tool benefit supervision。联合 gate 因而失败，不下载权重、不提交 GPU，
新增 checkpoint 数为 0。

这不是说公开 checkpoint 不可用。它很可能是一个值得复现的强 baseline；结论只是：在
当前顶会目标下，单独花资源证明它会调用工具，不能区分一个仍然新颖的训练假设，因此
不能越过“baseline 与 novelty 必须同时通过”的预注册边界。

## 公开权重盘点

2026-09-03 的 Hugging Face API 快照显示：

- `VTOOL/VTool-Qwen2.5-3B`：public、ungated、MIT，
  `Qwen2_5_VLForConditionalGeneration`，4,065,787,904 parameters，远端占用
  8,143,089,840 bytes，full revision 为
  `0ca11e812287b5c024c7277db71859da5bda17ac`；仓库文件页只提供 24-byte README，
  没有 tool prompt、parser 或 evaluation trace；
- `VTOOL/VTool-Qwen2.5-7B`：public、ungated、MIT，8,292,166,656 parameters，远端
  占用 16,595,836,368 bytes，full revision 为
  `b5c901087a12796ab1a783520e1098a194eaa540`。其 model metadata 把 base model 标成
  `Qwen/Qwen2.5-VL-3B-Instruct`，与 7B 名称/参数量不一致，且 model card 为空；在没有
 额外澄清时不能优先于 3B。

两个模型均不在本地 Hugging Face cache。3B 是尺寸最小且与当前 Qwen2.5-VL runtime
family 相容的候选，因而机器 gate 会把它列为“若科学上获授权则选择”的 artifact，但
`selected` 不等于已下载或已验证。

远端证据：

- [VTool 3B 文件与许可](https://huggingface.co/VTOOL/VTool-Qwen2.5-3B/tree/main)
- [VTool 7B model metadata](https://huggingface.co/VTOOL/VTool-Qwen2.5-7B)
- [VTool Hugging Face organization](https://huggingface.co/VTOOL)

## 代码、prompt 与 artifact 对接边界

本地两个公开代码快照都为 Apache-2.0：旧仓库 revision
`ae93c5457f7b5a397e3e70dda967a51b91a9a361`，`training-v2` revision
`d2aa28353ec10c7f91b39f502925003a81d6982d`。模型架构能被现有 Qwen2.5-VL loader
读取，属于工程上的正证据。

但静态检查也发现三处不可跳过的 provenance 缺口：

1. 旧 eval 脚本引用的是 `VTOOL/VTOOL-R1-3B-V3-F`、
   `VTOOL/VTOOL-R1-32B-F` 等旧 model IDs，而不是当前公开的
   `VTool-Qwen2.5-3B/7B`；
2. 新 `training-v2` launch recipe 默认从 base
   `Qwen2.5-VL-3B-Instruct` 开始训练，工具 prompt 由 Parquet 数据提供，没有把当前
   VTool checkpoint ID、训练数据 prompt hash 与 model revision 绑定；
3. 新 parser 只在完整 `python` fence 出现时认作 tool call，执行 context 暴露
   `image_1`、`columns_bbox`、`rows_bbox` 与 `focus_on_*`。模型卡没有说明当前权重学习的
   是哪一版 fence、函数名与参数合同，也没有提供 exact artifact 的 raw response → parse →
   execution trace。

因此 baseline 七项检查通过四项：公开、full revision、许可和 model family 通过；精确
checkpoint-to-code mapping、prompt/parser contract 与 parser-valid execution evidence
失败。一次下载后的 no-training smoke 可以补这三项中的后两项，但不能修复下面的新颖性
gate。

## 独立新颖性 gate

H5/N3 进入本轮前的核心候选由五项构成：signed tool value、tool-token responsibility
routing、fixed-prefix factual/counterfactual observation contrast、action-level
counterfactual credit，以及 with/without-tool benefit supervision。一手文献逐项给出：

- [TACO](https://arxiv.org/abs/2606.30251) 在同一问题、图像与 pre-tool reasoning 下读取
  tool-off/tool-on answer，以 outcome reward 差定义 `Delta`；有用为正、误导为负、不改变
  为零，并通过双 advantage channel 与 outcome-gated token routing 训练视觉工具 agent。
  这已经覆盖 signed tool value 与 responsibility routing 的中心主张。
- [TAPO](https://arxiv.org/abs/2606.05784) 为确定性信息获取工具在 batch 内构造
  counterfactual witnesses，并把 action-level credit 转移到被 GRPO 错误归因的调用。
- [The Illusion of Visual Tool-Use](https://arxiv.org/abs/2608.06270) 已在固定 prefix/action
  下替换单个 observation，用 real-vs-counterfactual difference 隔离 visual evidence
  effect。
- [ToolVision](https://arxiv.org/abs/2608.08907) 已用 stepwise evidence gain 选择 SFT
  trajectory，并在 RL 前比较 learner 的 with/without-tool benefit 来构造监督。

H5 的 observation-only paired contrast、显式调用成本和“只把 signed credit 给 action
tokens”在实现细节上并不完全等同 TACO。但在 The Illusion 已给出干预 estimand、TACO 已
把 signed before/after contribution 与 token routing 用于 RL 的情况下，这些差异目前只是
相邻工作的组合/约束变化，没有一个不可由现有组件直接合成的新原理。五项 core claim 均
被注册一手文献覆盖，六项 novelty checks 为 0/6；不允许仅以“公式略有不同”启动昂贵实验
追求事后新颖性。

## 机器 gate、资源与复现

- N2 输入 SHA-256：
  `60b398454f6a495c4fbcb337a0c1eae075cc1536ea09f2f78b2f0a2c2ac99404`；
- N3 registry SHA-256：
  `26e11938323078005d47053f25fe2b3909bc1f0ef2a62ce0b8e3344f4110ab2e`；
- N3 机器报告 SHA-256：
  `6d145cba4846ff608788b5dc8791d7fabcd0cdd1380ff9f2907bea5be3394f5c`；
- module/runner/test SHA-256：
  `be8a6334ad6d93bf1de6e855af70815ba22e739162efef9f5c4fcbce50cea842` /
  `5811bce80d3c4dc0d11dc3fb506583d1ef5d1a18e5de7ae69c032b5e5583010d` /
  `30d9f10e5fb41e35500329c9ab80da7d562e77c990a7ac6a5340a73fe4eedc86`；
- baseline checks 4/7，novelty checks 0/6；最终 joint gate 为 false；
- `downloaded_checkpoint_bytes=0`、`authorized_new_gpu_jobs=0`、
  `authorized_new_checkpoints=0`。

复现命令：

`PYTHONPATH=.:src python scripts/audit_tool_checkpoint_novelty.py --registry configs/n3_tool_checkpoint_novelty_audit_v1.json --n2-report artifacts/docvqa-train-factorized-v2/ops/n2-causal-regret-decomposition-audit-v1.json --expected-n2-sha256 60b398454f6a495c4fbcb337a0c1eae075cc1536ea09f2f78b2f0a2c2ac99404 --hf-cache /userhome/cs3/yihangc/.cache/huggingface/hub --old-vtool-repo /userhome/cs3/yihangc/Documents/references/vtool-r1-old --training-v2-repo /userhome/cs3/yihangc/Documents/references/vtool-training-v2 --output artifacts/docvqa-train-factorized-v2/ops/n3-tool-checkpoint-novelty-audit-v1.json`

相关 18 tests、三文件 mypy、compileall、Black check、两次机器报告 byte comparison、
N2 hash gate 和 `git diff --check` 通过。本项只读本地代码/cache 并获取公开元数据；没有
模型权重下载、Slurm、GPU、optimizer、protected split 或新增 checkpoint，因此没有触发
计算任务状态邮件。

## 下一步边界

关闭“先换成 VTool checkpoint，再验证当前 H5 credit”的路线。公开 checkpoint 仍保留为
未来强 baseline 候选，但只有出现独立新方法假设，或论文问题改为对 checkpoint 本身有
必要的可复现实证时，才值得下载并做一次 no-training support smoke。

下一候选不得只是把 The Illusion 的 estimand、TACO 的 signed credit、TAPO 的 witness 或
ToolVision 的 benefit filter 重新组合。应回到当前最坚实的负结果：工具调用的价值高度
稀疏且 outcome-free router 不可预测，寻找能在**不依赖答案标签和工具 rollout**时改变
信息获取决策的独立归纳偏置；在写出与上述工作不可约的机制前继续保持 GPU gate 关闭。
