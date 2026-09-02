# InfographicVQA literature-attention where 结果 v1

状态：特征 Job `203273` 与评估 Job `203340` 均正常完成；冻结机械决策为
`literature_attention_where_train_not_supported`。ViCrop 与 LASER 的所有注册
operating point 都未通过，validation/test/reserve 保持封存。本结果不授权
calibration、事后调参或 GitHub push。

## 绑定执行与审计

- 协议 SHA-256：
  `a86c4327a5e7ea8f5787b95883240149835e52a603266715900b5fddf8d682b1`；
  blind audit SHA-256：
  `1731fe8cf14568bb92ec8878477fa1f47dbb102f06953e84490afaa356cd7993`。
- 特征代码 revision：`940ee8603f8b84bb7e107be4ecbd21cf9698d2b8`；
  evaluator revision：`96508310366e5327c80094c3016a67561ec882c9`；模型 revision：
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`。
- 数据：InfographicVQA official-train，23,946 decisions、2,204 whole-source
  groups、4,406 images；四个 extraction shards 完整且 source-disjoint。
- 特征 Job `203273`：2xH800，16 CPU，192 GiB；2026-09-02 11:56:03--17:00:07
  HKT；三次 prefill/decision；checkpoint/resume 启用；exit `0:0`。
- 评估 Job `203340`：CPU evaluator，RTX 4090 仅为 QOS admission 预留并对进程
  隐藏，4 CPU，64 GiB；17:00:23--17:09:52 HKT；runtime `00:09:45`，
  exit `0:0`。
- 所有计算任务均绑定 `--mail-type=ALL` 到 `yihangc@connect.hku.hk`。
- 特征中无 outcomes，候选 crop 未在特征抽取时执行；privileged teacher 仅用于
  evaluation；validation/test 未读。Raw-attention 与四个冻结 comparator 均精确
  复现。
- 复用 formal `int32 [20000, 2204]` whole-source bootstrap，seed `20260917`；
  两个候选使用 Bonferroni-corrected central 97.5% intervals。

## 冻结 operating points

下表均为 source-balanced cost-adjusted utility；括号为校正 97.5% interval。
`Delta raw` 是候选减 raw-attention where 的 paired utility。

| Nominal rate | Calls | LASER utility | LASER Delta raw | ViCrop utility | ViCrop Delta raw |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5% | 120 | -0.000039 [-0.000370, 0.000346] | +0.000059 [-0.000197, 0.000419] | -0.000067 [-0.000415, 0.000327] | +0.000031 [-0.000330, 0.000443] |
| 1% | 240 | -0.000082 [-0.000923, 0.000662] | -0.000040 [-0.000571, 0.000436] | -0.000194 [-0.001019, 0.000500] | -0.000153 [-0.000757, 0.000391] |
| 2% | 479 | -0.000476 [-0.001749, 0.000728] | +0.000109 [-0.000701, 0.000984] | -0.000653 [-0.001832, 0.000420] | -0.000069 [-0.000854, 0.000650] |
| 5% | 1,198 | -0.000373 [-0.002217, 0.001445] | +0.000038 [-0.002145, 0.002075] | -0.000833 [-0.002532, 0.000838] | -0.000423 [-0.002633, 0.001576] |
| 10% | 2,395 | -0.002808 [-0.005990, 0.000367] | -0.000124 [-0.003200, 0.002962] | -0.002981 [-0.006050, 0.000123] | -0.000296 [-0.003231, 0.002687] |

两个候选的所有 utility 点估计都为负，所有校正下界都不大于零，且没有一个
operating point 对包括 raw attention 在内的所有注册 deployable comparator
满足非劣条件。LASER 在 5% 对四个旧 where comparator 显著更好，但对 raw
attention 的 paired interval 跨零；这不能转化为方法通过。两个 10% 点还违反
induced-harm tolerance。

## Localization 与 stopping 解释

在全部状态上，LASER 的 exact NLL-teacher agreement 为 `33.60%`、exact
task-oracle agreement 为 `24.14%`、helpful-state rescue 为 `72.15%`；ViCrop
分别为 `33.18%`、`20.28%`、`70.00%`。LASER 动态层在 23,946 个状态中的
16,767 个选择 layer 22，且两个方法都没有 zero-map fallback。

这些数值说明 literature attention 含有 localization 信号，但没有比 raw
attention 产生稳定的净收益。ENCORE early-attention entropy 与 helpful-crop
存在性的 Spearman 相关在 layer 0/1 分别只有 `0.0177` 与 `-0.0144`，与所选动作
NLL regret 的相关绝对值也不超过 `0.0111`；它不能提供新的 stopping 信号。

结合 Job `203290` 的 fixed raw-action privileged stop ceiling `+0.021318`
（95% CI `[0.018447, 0.024444]`）与 Job `203330` 的线性 signed-value stop 失败，
当前结论是：有用工具调用确实存在，但现有 pre-action representation 无法以足够
精度识别；继续修改 attention layer/head/ratio、call rate、entropy threshold 或
线性 classifier family 没有合理的信息价值。

## 不可变产物

```text
ffec54e5c48ee9711bccde13a53f9ee4c9e6b85a2453eadcbe8ddde3236bec02  merged literature features
53775f83b9a0231c0104b1b0fe69fedab6d1ffbeecaf0c28d5696c7bd0bfca9b  feature audit
0f8ab5fefe1f2974775261d84b5b56a93ec7fb94691a55018eaed1ac345128c0  merge report
8e059abdddd712ef7e230a32987b42064bd99ef3dbf137fc4ada52747d089e7e  feature completion
dfdb7e33dc975d24faad7d97241dbc80070633af2a1863813ea537ec7c844db1  feature execution
560b47edfa6cf2465d40e4138a4e5b5133898a2437af864a6bd32f1599d264ee  evaluation
0e6c092e05d7a26da61ba77a54cba76f674599ab4f5e9b486740f027b9d58b91  decision
10c1f24c69c4deef611f1b3a74dc46fe05d9e1c2497af01639235a8026a3d61f  evaluation completion
f1f4a222f11db0dfb61e7c7aaa4a3d9fb44ed7590249b45b805170d080dffc6d  evaluation execution
```

## 路线关闭与下一步

关闭当前 fixed four-box attention-localization / simple-confidence family。
不得在已打开的 official-train outcomes 上调 layer、head、ratio、threshold、
call rate、C、feature weights、seed 或 classifier family。下一步只能是具有新
信息来源的实质 pivot（例如新的 action proposer/representation 与显式 stop/where
因子化），或转向以完整 sibling bank、prospective risk 与跨域失败机制为核心的
严谨 empirical audit；二者都必须另行冻结并保留 validation/test 封存。
