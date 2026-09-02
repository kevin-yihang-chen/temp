# VTool counterfactual action-credit upstream feasibility audit v1

状态：2026-09-02 17:53 HKT 完成静态 gate。结论为
`upstream_static_feasibility_supported_with_dependency_and_credit_path_blockers`。
本审计不授权安装环境、下载模型/数据、提交 Slurm 或训练；validation、test 与
reserve 继续封存。

## 冻结对象

- 上游：[VTool-R1 training-v2](https://github.com/VTool-R1/training-v2)。
- 本地只读浅克隆：
  `/userhome/cs3/yihangc/Documents/references/vtool-training-v2`。
- pinned commit：`d2aa28353ec10c7f91b39f502925003a81d6982d`
  （`2026-03-20 12:34:48 -0500`, `Update README.md`）。
- License：Apache-2.0；浅克隆占用约 25 MiB；worktree clean。

## 当前训练通路的事实

1. `recipe/vtool/vtool.py` 首先把生成的 assistant chunk 标为
   `response_mask=1`；一旦该 chunk 被解析为工具调用，又把整段 action tokens
   改成 `response_mask=0`，对应 rollout log-prob 也置零。
2. 工具 observation tokens 追加到 prompt，但同样为 `response_mask=0`。工具后的
   final-answer chunk 才保留为可训练 response；`vtool_final_response_text` 也通过
   该 mask 解码。
3. `turn_scores` 与 `tool_rewards` 目前只是 `extra_fields` 中的空列表，并没有进入
   可训练的 token-level reward/advantage tensor。
4. `compute_grpo_outcome_advantage` 先把 token rewards 求和成每条 rollout 的标量，
   做 group normalization 后，再乘单一 `response_mask` 广播到 token 维度。因此
   当前 architecture 不能给 action tokens 和 final-answer tokens 分配不同来源的
   advantage。

所以本方法不是“给 reward manager 加一个分数”即可实现。若保持当前 mask，
`A_visual` 对 action policy 的梯度严格为零。

## 最小必要实现面

1. Agent loop 额外输出互斥的 `action_mask` 与 `answer_mask`；observation tokens
   继续全零，union 才作为 policy/log-prob 的可训练 response mask。
2. 保留现有 `vtool_final_response_text` 的 answer-only 解码语义，不能因 union mask
   把 tool code 送给最终 answer judge。
3. 新增显式 advantage estimator，例如
   `A = A_outcome * answer_mask + beta * normalize(A_visual) * action_mask`；两项分别
   normalization，并审计 loss、old/ref log-prob 与 KL 是否真正覆盖 action tokens。
4. 对同一 action prefix 生成 factual edited-image continuation 与预先冻结的 no-op
   或 fixed-alternative continuation；固定 decoding 配置与可控的 paired seed，交换
   factual/counterfactual 两臂时 signed score 必须严格变号。
5. 保存 branch identity、action cost、两个 final scores、signed effect、masks 与
   scorer provenance；shuffled/sign-randomized credit 必须复用完全相同的 rollout
   budget。

这些改动集中在 VTool agent loop、trajectory batch contract、advantage registry 与
单元测试，静态上可实现；但在 synthetic sign/mask tests 之前不能推断训练正确。

## 环境与恢复风险

- 3B recipe 是单节点 4 GPU，默认 batch 32、rollout `n=8`、prompt/response 各
  8192 tokens、15 epochs，标准脚本 `save_freq=20`；这与账户最多 4 H800 的资源
  上限形状相符，但不证明显存与吞吐已通过。
- PPO 配置有 `resume_mode=auto`/`resume_from_path`，trainer 有 checkpoint load，
  因此具备恢复入口；精确 data-sampler、paired-branch 与 RNG 恢复仍需中断 smoke
  验证。
- 上游依赖定义不一致：`docker/Dockerfile.stable.vllm` 安装 PyTorch `2.10.0`、
  CUDA `12.9.1` 与 vLLM `0.17.0`；`setup.py` 的 vLLM extra 却限制
  `vllm<=0.12.0`；`requirements.txt` 还保留注释 `vllm==0.8.4`。在确定 recipe
  对应的权威 image/digest 并完成 import-only CPU audit 前，不能直接污染现有环境。
- paired counterfactual 至少增加 tool-use 样本的 final continuation/scoring 成本；
  实际额外 GPU-hours 只能由小真实 batch smoke 测量，不能由 tool-call rate 直接
  外推。

## Gate 决定

上游静态结构没有否定该方法，且 4×H800 资源形状原则上适配 3B recipe；因此允许
进入 protocol 与 synthetic implementation 阶段。但以下任一未解决都禁止训练：

1. 不能冻结唯一依赖 image/digest，或 import-only audit 失败；
2. action/answer/observation mask 不能在逐 token 单测中完全分离；
3. paired scorer 不能保证同 prefix、同 decoding 条件和 arm-swap antisymmetry；
4. checkpoint/resume 改变 paired identity、mask 或 credit；
5. full-text novelty audit 无法把本方法与 ToolVision、AdaTooler-V、AdaptVision 的
   stepwise/question-level benefit 和 decoupled objective 清楚区分。

下一项只写一个 matched-control protocol 与纯 synthetic tests；没有通过上述 gate
前，不安装 GPU 环境、不提交 Slurm、不读取新 outcomes。
