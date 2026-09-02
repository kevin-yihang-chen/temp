# InfographicVQA fixed-action stop 因子化诊断结果 v1

状态：Job `203290` 于 2026-09-02 完成，机械执行通过。该实验使用已打开的
official-train outcomes，是 post-hoc 诊断，不是 deployable 候选或正式结果。

## 绑定执行

- 假设：固定 raw-attention action 后，剩余负 utility 主要来自 stopping；
  attention max 或 top-two margin 可能比 entropy 更好地识别值得执行的状态。
- 代码 commit：`91a0359a5bafdd086b22cba153077177438952d0`。
- 数据：InfographicVQA official-train，23,946 decisions，2,204 sources，4,406
  images；绑定 raw feature SHA-256
  `009cdbfa1761f9b53b791a50f70b7e98bdf275eec8743d8bcaf078a52ded8ce8`。
- 配置：固定 `argmax(question_region_attention)` action，`lambda=0.05`；
  call rates 0.5/1/2/5/10%；20,000 次 whole-source bootstrap，沿用 raw gate
  的冻结 bootstrap indices。
- 命令：`scripts/submit_infographicvqa_attention_stop_factorization.sh`。
- 环境/资源：Slurm Job `203290`，1×RTX 4090 预留但 evaluator 隐藏 GPU，
  4 CPU，64 GiB，runtime `00:18:09`，exit `0:0`，全状态邮件。
- 日志：`slurm-infovqa-stop-diag-203290.out`。
- 产物：diagnostic SHA-256
  `f07eddb658444cd11ab67a62b53143c90ebf81a07026f00c7bba1411a3ad8e1a`；
  complete SHA-256
  `0160654dd9173192409b434728c3a654c76a275dd55220e6ecd6ab74d50ef068`；
  execution SHA-256
  `03cfc69868333a9613c4a0e65fd01d20cda4763b970e1e9ec7c9ce4627b584c9`。

## 结果

Raw fixed action 有 1,023 个 positive-net states，来自 483 个 sources。

| Nominal rate | Entropy utility | Attention-max utility | Margin utility | Budget oracle utility |
|---:|---:|---:|---:|---:|
| 0.5% | -0.0000975 | -0.0004001 | -0.0003839 | +0.0044968 |
| 1% | -0.0000413 | -0.0008041 | -0.0010019 | +0.0102111 |
| 2% | -0.0005847 | -0.0015571 | -0.0016535 | +0.0162879 |
| 5% | -0.0004104 | -0.0023299 | -0.0029065 | +0.0213175 |
| 10% | -0.0026846 | -0.0047354 | -0.0060091 | +0.0213175 |

Unrestricted fixed-action positive-net oracle 的 source-balanced utility 为
`+0.0213175`，95% CI `[0.0184472, 0.0244436]`；调用率约 3.93%，
positive-net precision 为 1，无 harm。Full task-action positive-net ceiling 为
`+0.0338476`，95% CI `[0.0301742, 0.0377351]`。

## 结论与解释边界

Fixed raw action 的可学 stopping headroom 明显，因此 raw where 的负 gate 不等于
action 本身无效。但 attention max/margin 的 positive-net precision 在每个注册预算
都低于 entropy，简单 confidence stopping 已被否定。

下一步只允许一个预先固定的低容量 whole-source OOF signed-value stop
候选；不搜索 attention max/margin 变体、模型族、特征子集或注册 call rate。
Validation/test 仍封存。本结果未授权 GitHub push。
