# ReFocus typed-action B0 generation 结果审计（Job 206227）

审计时间：2026-09-03 14:18（Asia/Hong_Kong）

## 冻结问题

独立 typed-action V2 在真实 Qwen2.5-VL-3B 首轮生成中，是否能产生足够的工具意图，
并使其中至少 80% 依次通过完整 Python 围栏、语法、参数合同、strict parser 与真实
executor？本实验只评价 baseline correctness，不训练、不写 checkpoint，不重解释
Job `206205`，也不是 N0 方法证据。

## 运行与终态

- Git revision：`96cd1665c72913b971ebb7c5aaa87f2bee15bbc0`（clean submission）。
- Slurm Job：`206227`；`q-h800`，1×H800、8 CPU、64 GiB、30 分钟上限。
- Slurm 终态：`COMPLETED`，`ExitCode=0:0`，零 restart；
  2026-09-03 14:08:12--14:09:33 HKT，运行 81 秒。
- 通知：`MailUser=yihangc@connect.hku.hk`，BEGIN/END/FAIL/REQUEUE 等状态事件已配置；
  Slurm 配置不等于邮件客户端送达证明。
- 模型：`Qwen/Qwen2.5-VL-3B-Instruct` revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`；vLLM `0.12.0`、Torch
  `2.9.0+cu128`、CUDA runtime `12.8`、bfloat16。
- 模型加载 16.2395 秒，16 次生成 20.3844 秒；加载模型占用约 7.16 GiB。
- optimizer step `0`，checkpoint `0`，reward target 未使用，protected split 未访问，
  raw model text 从未直接执行。输出目录只有 `report.json`。

## 预注册分层结果

16 个唯一种子固定为 `2026090300..2026090315`；temperature `0.7`、top-p `0.9`、
max tokens `128`。21/21 输入与 provenance checks 全真。

| 层级 | 计数 | 全部生成率 | 有意图条件率 |
| --- | ---: | ---: | ---: |
| tool intent | 11/16 | 68.75% | 100% |
| 完整 Python fence | 7/16 | 43.75% | 63.64% |
| Python syntax valid | 7/16 | 43.75% | 63.64% |
| argument contract valid | 0/16 | 0% | 0% |
| strict parser valid | 0/16 | 0% | 0% |
| execution success | 0/16 | 0% | 0% |

冻结 gate 要求至少 2 个 tool intents 且 intent-conditional execution rate 不低于 80%。
前者通过，后者为 0%，机械 scientific decision 为
`typed_action_b0_malformed_tool_intent`。

## 失败分解

- 5/16 直接回答 `FINAL ANSWER: 2024 TERMINATE`，无工具意图。
- 7/16 是完整且语法合法的 Python fence，但全部调用无效函数
  `focus_on_x_values_with_MODE`，因此在 allowed-function 检查处失败。
- 另外 4/16 有工具意图但没有完整 fence；它们也全部使用同一无效 `_with_MODE`
  函数名，其中 3 条还混入 final answer，1 条追加 `TERMINATE`。
- 因而 11/11 有工具意图的输出都逐字保留了 prompt 示例中的元变量 `MODE`，没有按文字
  指令替换为 `draw`、`highlight` 或 `mask`。这直接解释了 0/11 参数合同通过率。
- 7 条完整 fence 中，4 条只选择正确标签 `"2024"`；2 条错误扩张为全部 x labels；
  另 1 条除扩张标签外还把 x-axis 函数配成 `rows_bbox`。由于函数名检查已先失败，
  后两类是可见的附加合同错误，不能当作独立 parser 计数。

最窄的机制判断是：V2 把元变量嵌入“唯一合法形式”，Qwen 具有明显工具意图且经常
遵守 fence，但把抽象模板当作可复制代码。该证据否定当前 V2 prompt 的可靠 baseline
资格；它不证明 concrete-function prompt、结构化解码或更大模型必然成功，也不支持
事后修复这些 16 条输出。

## 不可变边界与下一步

1. 不改 V2 prompt、种子、temperature、top-p 或阈值后重跑，不把其失败改写为 G1 结果。
2. V2 报告与 raw completions 原样保留；不执行任何失败输出。
3. 若建立 V3，只能作为新版本 baseline correction：先在 CPU 上证明所有展示模板本身
   parser-valid，并预注册一个未用于 V1/V2 的 official-train structural group、新种子和
   单次停止规则；不能在已打开的当前 row/seed 上选择模板。
4. V3 仍只是强 baseline。主方法 N0 继续要求新 estimand、新颖性审计和 zero-support
   synthetic gradient gate，不能把 prompt correction 包装成论文贡献。

## 内容绑定

- protocol config SHA-256：
  `05a09c1aba64bb685d2808a4c447441f319aac7ecbd8646af34d9ce35d3214e2`
- dataset SHA-256：
  `2c6a6c9b0a2329199ca750ad6489d3b1fafdf17b91a106ab426c634510e8184c`
- report SHA-256：
  `9c6c763e10b0e005b605331bc2e3570fb3d841b6b36fc63f362c5f3dd741e875`
- worker status SHA-256：
  `c4731fc7d95fc8ed9957a9745f20f30985b6476dd6e49d54f60b6208ca2b461e`
- Slurm log SHA-256：
  `eaea14cbfb0636728d6a0c0c6f8c6736c6fe8ee750d2c2a9c24db46f829286d8`
- runner / worker / submitter SHA-256：
  `4eb30ae82ec27cb3b5cddbf2f28dda68ab6cdbb4efcaab02d16f8cd459ece77a` /
  `89b4a261e940f3e196185d10fe601a0b11bcbf1753c608e4765c2255cb21682e` /
  `7c92e8c709aa59a365a9d7c25d74db56328ba08fa2a78555ec0df321f6dccc9b`
