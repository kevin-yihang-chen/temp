# InfographicVQA fixed-action signed stop OOF 结果 v1

状态：Job `203330` 于 2026-09-02 正常完成，机械决策为
`fixed_action_signed_stop_train_not_supported`。该结果不授权 calibration、
validation/test 开放或 GitHub push。

## 绑定执行

- 协议 commit：`e0f95d2ff0ed01b0343530989d1a2b52242ada3d`；协议 SHA-256
  `32e6080399153b91f1584d068590826fb105451c52c0a8944f0ce62ba6dac74a`。
- 提交 commit：`7b5f5ea2500cd49ad101c3dd11422f32d8e5bb98`。
- 数据：InfographicVQA official-train，23,946 decisions，2,204 sources，4,406
  images；1,023 positive-net fixed-action states。所有输入哈希由冻结协议绑定。
- 配置：fixed raw-attention argmax action；80 维 pre-action features；L2
  logistic `C=0.01`；绝对 net-utility / equal-source 权重；5 个 whole-source
  OOF folds，seed `20260918`；唯一 primary 为 2% / 479 calls；20,000
  次冻结 whole-source bootstrap。
- 命令：`scripts/submit_infographicvqa_attention_signed_stop_oof.sh`。
- 资源：RTX 4090 预留但隐藏，4 CPU，64 GiB，runtime `00:07:00`，
  queue wait 11 秒，exit `0:0`，全状态邮件。
- 日志：`slurm-infovqa-signed-stop-203330.out`。

## 唯一 primary 结果

| Metric | Signed-value stop | Entropy stop |
|---|---:|---:|
| Pooled calls | 479 | 479 |
| Source-balanced utility | -0.0000626 | -0.0005847 |
| Positive-net calls | 90 | 77 |
| Positive-net precision | 18.79% | 16.08% |
| Source-balanced induced harm | 0.0008267 | 0.0013701 |
| Source-balanced negative-utility magnitude | 0.0013452 | 0.0019664 |

Candidate utility 的 95% CI 为 `[-0.0007393, 0.0006553]`。Candidate-minus-
entropy utility 为 `+0.0005221`，paired 95% CI
`[-0.0003039, 0.0014439]`。Positive-net precision 条款通过，但 candidate
utility 过零和 paired improvement 两条都失败。

## 次要曲线与边界

0.5% 点的 candidate utility 为 `+0.0001598`，95% CI
`[-0.0001800, 0.0005725]`；相对 entropy 的 paired lower endpoint 为
`-0.0000158`。这是次要描述性 near miss，不能挽救预注册 2% primary。
1%、5%、10% 的 candidate utility 均为负；10% 的上端仍为负。

所有分折 source overlap 为 0，模型在 6--7 iterations 收敛；输入、有限值、
OOF coverage、matched call、无泄漏与禁止数据角色审计全部通过。

## 产物与结论

- Report SHA-256:
  `aa5de1fa1d9891d8425d192e7ed03782c003491d28c435dcf22abc69711e51ad`.
- Model SHA-256:
  `a053a47c5914d96423906abdd2d09500d3e2e193bb66826436a3149c0290be5e`.
- Scores SHA-256:
  `9bf4ad6a895864811427e9c37aeadf4844a8b5345165babb545db7fc9cc5f945`.
- Completion SHA-256:
  `47e5cf8eb5ae89ed3834042492122844ebf236058e9b093efd2b2b7fc7b1d62a`.
- Execution SHA-256:
  `aff57cd082b644a957ebd3a45442e636f463ecd94a4aaaca939234811924c7c4`.

结论是“存在弱排序信号，但不足以产生稳定正 utility 或统计显著的
entropy 改善”。按协议停止该模型族，不在已打开的 train outcomes 上调整
C、特征、权重、seed、classifier family 或 primary call rate。下一主决策等待
冻结 literature-attention where 强基线。
