# Counterfactual Utility SFT: GO / NO-GO

1. **Utility-SFT 是否超过 Frozen VOI？**

   **否，未被稳定证明。** Primary `lambda=0.05` 下，ChartQA 差值为 `0`；DocVQA 为
   `+0.000565`，95% CI `[-0.002073, 0.003363]`；HRBench 为 `+0.010156`，95% CI
   `[-0.056250, 0.079687]`。没有一个 domain 的 paired lower endpoint 大于 0。

2. **Utility-SFT 是否超过 Best-Action SFT？**

   **否。** ChartQA 和 DocVQA 差值均为 `0`，两者都退化为全选 `ANSWER`；HRBench
   为 `+0.035156`，95% CI `[-0.009375, 0.093750]`，不足以证明 soft counterfactual
   utility supervision 比 one-hot imitation 有额外价值。

3. **是否至少两个 domain 成立？**

   **否。** ChartQA、DocVQA 均不成立；只有 HRBench 有方向性改善且置信区间跨 0。
   语义消融也没有呈现跨域一致的 image-question-region 依赖。

4. **是否值得进入 RL？**

   **否，当前路线为 NO-GO。** 已触发预注册 Stop 1、Stop 2、Stop 3。正式 test 未打开；
   不进入 GRPO/PPO、7B、multi-turn 或 continuous bbox，也不继续搜索 loss、seed、threshold
   来寻找偶然正结果。
