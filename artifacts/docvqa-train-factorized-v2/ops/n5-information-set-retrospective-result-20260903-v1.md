# N5 信息集效应回顾性否证结果

时间：2026-09-03 16:05（Asia/Hong_Kong）

## 结论

机械决定为
`n5_current_information_boundary_candidate_not_supported_before_calibration`。
N5 的 8 项科学条件全部失败，因此当前 N4 information-boundary benchmark/evaluation
候选关闭，不打开 ScreenQA `risk_calibration`，不提交 GPU，不生成 checkpoint。

这不是证明“更丰富视觉信息永远无用”。本项证明的是：当前仓库中可获得的 nested-
information learned routers 没有给出足够强、稳健、实质的现实效应，无法支撑为 N4 再
生成 49,755 条受保护 calibration action 记录。N4 的 selector ledger 仍可作为评测卫生
原则保留，但不足以作为 ECCV/ICCV/CVPR 主贡献。

## 协议诚信

协议和配置先于本次逐 decision 配对结果，以本地 commit
`9e674abb6ca08ab21266f5ddc308579cfa9f0dff` 冻结。协议明确披露两个 DocVQA 旧
question-weighted aggregate 和 ScreenQA OOF aggregate 已知，因此本项只称
retrospective route-falsification，不冒充盲测、预注册 formal 或新 benchmark result。

实现绑定 clean commit `2df1ad20e05740b34a5d32ce761f1175891173ba`。10/10 artifact
checks 全真：所有输入哈希、同一 DocVQA bank、相同训练协议/seed/folds/alpha/lambda、
label-free semantic feature、既有 frozen evaluation 复现、outcome-blind matched call set、
成本单调性和 ScreenQA role 封存均通过。

## DocVQA 同样本配对结果

数据为此前已打开的 1,608 个 decisions、400 个 sources；每个 decision 都有同一
`ANSWER_NOW + 4 UG-grid ZOOM` action bank。Primary 调用预算固定为 5%，实际恰好选择
80/1,608（4.9751%）个 decision。两个 learned router 的调用集合重叠 60/80，Jaccard
为 `0.60`；crop action agreement 只有 `30.10%`。

Source-balanced primary：

- 较低信息 `context-geometry` utility：`-0.00274318`；
- 较高信息 `semantic-context` utility：`-0.00296742`，97.5% CI
  `[-0.00757542, 0.00096483]`；
- higher-minus-lower：`-0.00022424`，paired 97.5% CI
  `[-0.00528886, 0.00430109]`。

所以高信息策略自身为负，且相对低信息模型既未达到 `+0.001` 最小效应，也没有正的
paired 下界。这里已经把高信息 feature acquisition cost 设为 0，给了它最乐观上界；
每 decision 再扣 `0.001/0.005/0.01` utility 时，source-balanced utility 单调降到
`-0.003967/-0.007967/-0.012967`，任何非负真实 feature 成本都不会改变失败方向。

旧 frozen threshold 也不支持高信息优势：source-balanced context/semantic utility 为
`-0.00165723/-0.00456720`，higher-minus-lower 为 `-0.00290997`。本项同时精确复现旧
question-weighted utility/call rate，排除了评估器或模型载入漂移。

## 一个重要但不能救活路线的聚合反转

同一 5% matched call 下，question-weighted context/semantic utility 为
`-0.00509079/-0.00395807`，差为 `+0.00113272`；但 source-balanced 差为
`-0.00022424`。也就是说，高信息模型的表面改善集中在 QA 数较多的 sources，换成每个
source 等权后符号反转。

这说明后续评测必须同时报告 question-weighted 与 source-balanced 结果，不能用前者覆盖
跨 source 失败。但“样本权重会改变结论”本身是常见统计问题，不等于 N4 所需的 selector
information-set rank reversal，也不能单独构成顶会新颖贡献。

## 强基线与 headroom

共同 5% 预算下的 source-balanced deployable 排名中：

1. entropy gate + fixed `ug-grid-01`：`+0.00125919`，但 97.5% CI
   `[-0.00357436, 0.00715268]`，不稳健；
2. `ANSWER_NOW`：`0`；
3. fixed `ug-grid-02`：`-0.00191399`；
4. random gate + random crop expectation：`-0.00273516`；
5. entropy gate + random crop expectation：`-0.00274215`；
6. context learned router：`-0.00274318`；
7. semantic learned router：`-0.00296742`；
8. fixed `ug-grid-03`：`-0.00461585`；
9. fixed `ug-grid-00`：`-0.00569795`；
10. 扣除四份候选执行成本的 exhaustive UG-style：`-0.00838393`。

Post-action entropy 只扣一次 crop 成本的乐观 idealized comparator 为 `+0.00047844`，
CI 仍跨零。相反，matched privileged oracle utility 为 `+0.03117658`，97.5% CI
`[0.02145013, 0.04238910]`。这再次证明有真实 action headroom，但当前 outcome-free
信息与模型无法可靠提取；不能把 oracle ceiling 解释为 deployable 正结果。

## ScreenQA 开发证据与成本止损

ScreenQA 已打开的 14,511-decision ranker-training OOF 中：

- context utility：`0.00006547`；
- hybrid semantic utility：`0.00011371`；
- 差：`0.00004824`，约比 `0.001` 门槛小 20.7 倍；
- 两者 tail-risk status 均为 `no_non_degenerate_safe_threshold`。

因此独立数据域上的现有开发证据也不支持晋级。ScreenQA 的 4,001 张 calibration 图像、
9,951 个 QA decisions 保持未打开；避免生成 49,755 条 action 记录。formal-test 与 reserve
继续封存。

## 识别边界

当前资产缺少 question-only selector、统一低分辨率 preview selector，以及同一方法族在
各信息集下的 factorial refit；两个模型的容量也不严格匹配。因此 strict N4
cross-information-set rank reversal 不可识别。负结论只关闭“用当前候选继续 N4”的路线，
不能外推为视觉获取的一般不可能性。

## 复现与哈希

- 配置 SHA-256：`c661c7fd4362cb2abf62058a769b1d5f557ae6132abdf85b8e20639ced1f0b00`；
- 协议 SHA-256：`276b936309efbf814d3b2467526e75385bad929849bf2b66864d1bb6acc44d74`；
- module/runner/tests SHA-256：
  `16a4fc85c8f8f82ff0152320bbe3dd7f73dd2d6118e2cb40a6fe7488893fd003` /
  `97eaab357255277d543b7262806eaafaa943e1f758c039202a0fbea38e4a61b0` /
  `6085976bdc0386ab53264bbfe6992b3e93ac6ba95dec069e5b3c00cff5a5e7a5`；
- 结果 SHA-256：`ed657489ee63950c73ec685ce24d023ac873f09d4252a752b70faacb752bad0a`。

复现命令：

`PYTHONPATH=.:src /userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python scripts/audit_n5_information_set_retrospective.py --config configs/n5_information_set_retrospective_v1.json --output <new-output.json>`

独立第二输出与正式 report 字节级相同。第一次复现尝试把 `mktemp` 已创建的空文件直接作为
output，按预期触发“不覆盖既有文件”的 fail-closed `FileExistsError`；改用新临时目录内
尚不存在的路径后成功，结果哈希完全一致。正式结果未受影响。

## 下一步

不继续 N4，不为它打开 calibration 或增加算力。下一个候选必须重新回到 problem selection：
只接受能解释并利用“privileged oracle 很高、所有现有 outcome-free router 跨 source 失败”
这一残差结构的新机制，而且先通过一手文献碰撞和已有数据上的零成本可证伪 gate。不得把
source weighting、fixed crop 偶然正点、更多 feature、阈值或模型容量变化包装为主贡献。
