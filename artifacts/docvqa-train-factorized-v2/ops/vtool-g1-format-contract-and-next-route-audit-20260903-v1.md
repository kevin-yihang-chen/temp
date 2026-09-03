# G1 工具意图、格式合同与下一路线审计

状态：2026-09-03 13:11 HKT。本文是 Job `206205` 的只读事后诊断，不改变冻结
G1 指标、停止阈值或正式 decision，也不把事后发现重算为方法结果。

## 结论

Job `206205` 的正式结论仍是 parser-valid tool call `0/64`，因此
`paired_signed_g1_stop_rule_triggered` 不变。但“64 条都是自然语言 direct answer”
并不准确：原始 response 中有 48 条以 `FINAL ANSWER:` 开头，另有 16 条以允许的
`focus_on_*` 函数名开头。后者表明模型存在裸工具意图，却没有形成可执行动作。

这 16 条中：

- 15 条是可解析的单个 Python focus-call expression；
- 1 条把裸调用与 `FINAL ANSWER` 混在同一响应，Python 语法不成立；
- 16 条均缺少要求的 ` ```python ... ``` ` 围栏；
- 15 条可解析调用均缺少 `display(...)`，且 **0 条**满足运行时真实三参数签名
  `(image_1, [axis labels], columns_bbox/rows_bbox)`；
- 因此 `fence_only_repair_executable=0/16`，仅补围栏不能恢复任何工具调用。

决定码为
`g1_zero_parser_valid_support_with_malformed_bare_tool_intent`。这把失败位置从笼统的
“模型完全不想调用工具”收敛为“存在 intent，但 syntax/API-argument support 为零”。

## 不可变证据

- 正式 rollout analysis SHA-256：
  `d8c4950831e38720b514ca087411d249bfcb431b39bce5c76a0d2736102b6c21`。
- Step 1/2 rollout SHA-256：
  `04ba5634e7298186724404ae0b88b1482f9c53358941d605e20775b50a65ae6f` /
  `4b07ae08172f170565f745e13612d17ae8638aa6264bd973acaeab4b0c1ea55b`。
- 事后诊断 JSON：`vtool-g1-intent-format-posthoc-job-206205-v1.json`；SHA-256
  `df19920bd426e62d1d2152d85bf2afce596c7b111c247e5807e4a5ba17a44160`。
- 诊断器 SHA-256：
  `1a83b250a402f1e6fd285a419fba9c99bf28f11c6585c1baf6deb9b7025fa851`；实际 agent
  `vtool.py` SHA-256：
  `e75f54b9792c2349fa21ec684233d58b63da909a0d18c4c085923393ab799215`。
- 可复现命令：

  ```bash
  PYTHONPATH=.:src python scripts/analyze_vtool_g1_intent_format.py \
    --rollout-dir artifacts/docvqa-train-factorized-v2/g1-runs/paired-signed-v1/job-206205/rollouts \
    --official-analysis artifacts/docvqa-train-factorized-v2/g1-runs/paired-signed-v1/job-206205/rollout-analysis.json \
    --runtime-vtool /userhome/cs3/yihangc/Documents/runtime/vtool-action-credit-g1/recipe/vtool/vtool.py \
    --output /tmp/vtool-g1-intent-format-posthoc-job-206205-v1.json
  ```

诊断器逐行绑定 trajectory ID、raw output 与 audit payload，验证两步各 32 行、正式
parser-valid count 与原报告一致、protected split 仍关闭。它不加载模型、不执行工具、
不修改输出，也不读取 validation/test/reserve。

## 分步现象与解释边界

| Step | `FINAL ANSWER` | 裸 focus intent | parser-valid | 签名合法 | 仅补围栏可执行 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 20/32 | 12/32（37.5%） | 0 | 0 | 0 |
| 2 | 28/32 | 4/32（12.5%） | 0 | 0 | 0 |
| 总计 | 48/64 | 16/64（25.0%） | 0 | 0 | 0 |

Step 2 的裸意图率低于 Step 1，与 ordinary outcome GRPO 对无效响应施压的解释相容，
但两步使用不同 mini-batch，且只有一个 run，因此不能把下降因果归因于 optimizer。
它只作为后续预注册实验的方向性证据。

运行时 parser 在没有 ` ```python ` 起始围栏时无条件返回 `NOTOOL`；而真实六个
axis-focus 函数都要求 image、label sequence 和对应 bbox mapping 三个位置参数。冻结
V1 prompt 只列出函数名与变量，没有给出签名或一个可执行模板。诊断器进一步对实际
agent `vtool.py` 做 AST 审计：execution context 只赋值 `image_1`、`columns_bbox`、
`rows_bbox` 与 `display`，并未注入 prompt 所称可用的 `x_values_bbox/y_values_bbox`；
用后两者的形式会真实触发 `NameError`。因此当前证据支持“prompt/action API 合同不足”
是 baseline 缺陷，但尚不证明加模板后会产生有用调用。

Job `206184` 的 32 条 step-1 response 也需同样纠正：严格 parser 记为 `0/32`，但原始
文本实际为 19 条 `FINAL ANSWER` 与 13 条裸 focus intent。它没有完整 step 2，仍不能
替代 Job `206205` 的正式判定。

## 一手文献碰撞

以下方向不能作为本项目的新方法主张：

1. **只修 prompt、加格式模板或强制合法调用。** 这是建立可靠 baseline 的必要工程，
   不是贡献。Tool-RL collapse 已把 control-token/格式失效和 SFT/hint/off-policy
   supervision 作为核心问题；ToolVision 也明确用 capability-aligned SFT 教 how-to-use。
   参考：[Tool-RL collapse](https://arxiv.org/abs/2606.26027)、
   [ToolVision](https://arxiv.org/abs/2608.08907)。
2. **对 call token 做 representation steering 或固定 logit bias。** 前者已能把 call
   rate 从近 0 单调推到 90% 以上并报告多模型、多模态和 live-tool frontier；后者已从
   KL-regularized RL 推导 rollout/reward/token-probability 的 inverse-propensity
   estimator。参考：[Tunable Tool-Call Rates](https://arxiv.org/abs/2608.25198)、
   [Black-Box Logit Bias](https://arxiv.org/abs/2607.22837)。
3. **把全部候选 response/action 的 reward 做普通 listwise optimization。** LIRE 已用
   多条 offline response likelihood 与 reward 直接做 listwise reward optimization，
   LiPO 已系统化 ranked-list preference objective，ToolPrefer-LLaMA 已从 inference tree
   构造 step-wise tool preference 并做 SFT+DPO。参考：
   [LIRE](https://arxiv.org/abs/2405.13516)、
   [LiPO](https://arxiv.org/abs/2402.01878)、
   [ToolPrefer-LLaMA](https://arxiv.org/abs/2406.07115)。
4. **只从候选 crop 的 loss/utility 学一个 whether/where router。** GapSight 已比较
   global-only 与 candidate crop views 的 target loss/option margin，并训练预测 whether、
   expected utility 和 continuous crop box 的 router。参考：
   [GapSight](https://arxiv.org/abs/2608.21762)。

因此，forced-call、SFT curriculum、tool bonus、call-rate steering、普通 logit bias、
全 response listwise reward 或 crop loss-gap router 均只允许作为 comparator/基础设施，
不能作为论文主方法。

## 冻结的下一步

下一步分成不可混淆的两个 gate：

### B0：typed-action baseline 修复

这是 baseline correctness，不是新颖性实验。新增 V2 prompt/action schema 时必须：

1. 保留 V1 数据、Job `206205` 产物和 decision，不覆盖、不重算；
2. 给出唯一 canonical grammar：
   x-axis 使用
   `display(focus_on_x_values_with_{mode}(image_1, [labels], columns_bbox))`，y-axis 使用
   `display(focus_on_y_values_with_{mode}(image_1, [labels], rows_bbox))`；
3. 用纯函数 renderer/parser 做 round-trip、非法轴、非法 label、错误 bbox、额外语句和
   direct-answer 分支测试；
4. 重新生成独立版本号和 hash 的 official-train-only development data；
5. GPU 前先在真实单行 processor/fake executor 上证明 exact prompt 可执行；随后最多
   一次无 checkpoint 的 1×H800 generation smoke。预注册报告 intent、syntax、argument、
   execution 四层 rate，不以单一 `tool_call_rate` 混合它们；
6. B0 只回答底座能否可靠表达动作，不允许据此宣称 action credit 有效。

B0 的纯 CPU 第一阶段已由 commit
`150803ac113008f9ad5555f00f743aa17df9746c` 完成：V1 prompt hash 保持不变；V2
canonical renderer/strict parser 对全部 axis/mode 做 round-trip，并拒绝错误 bbox、
kwargs、越界 label、额外语句和混合答案。生成的 x/columns 与 y/rows 两类动作均已在
pinned `refocus_tools.py` context 中执行并返回唯一 PIL image。全仓为 532 passed、35
expected skips。该证据仍未覆盖真实模型 generation，下一 gate 是独立 V2 单行 official-
train processor/fake executor，而不是直接提交训练。

### N0：action-boundary interventional objective 新颖性 gate

唯一保留、但尚未获实验授权的主方法候选，是把有限 typed visual macro-action 的
**边界概率**与同 prefix、同 continuation seed 的 action intervention outcomes 结合，
直接优化 action-selection regret，同时把 answer-token learning 分离。它必须在写 GPU
protocol 前证明：

1. estimand 与整段 response likelihood 的 LIRE/LiPO 不同；
2. learning signal 与 ToolVision 的 beneficial-question MUT reward、GapSight 的
   loss-gap router 及 ToolPrefer 的 step-wise DPO 有不可约区别；
3. 在 sampled valid action support 为零时仍有可计算、无偏或明确有界偏差的梯度；
4. 用 synthetic finite-action environment 验证符号、归一化、zero-support、harm/cost 与
   answer/action gradient isolation；
5. 若形式化后退化为 contextual-bandit full-information/listwise loss，则在 CPU gate
   关闭，不消耗 GPU。

在 N0 通过前不重跑 G1、不启动 controls、不提交 GPU。当前 35.38 GiB 空间也低于原
64 GiB checkpoint gate；B0/N0 的 CPU 工作无需删除已哈希 checkpoint。
