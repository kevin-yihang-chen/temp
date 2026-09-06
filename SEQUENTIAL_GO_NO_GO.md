# Sequential Acquisition GO / NO-GO

Current verdict: **PENDING — tiny development smoke passed; powered evidence is absent.**

The held-out test is unopened. This file must not claim GO, PARTIAL GO, or NO-GO until the
fixed protocol has completed train/validation freezing and its one-shot independent test.

1. Does a second fixed visual acquisition have beneficial and harmful counterfactual
   support? **Yes in a 144-state engineering smoke, but rates are sparse and not powered.**
2. Does the learned gain critic beat entropy/confidence/margin and matched random at the
   same acquisition rate? **No in the 16-state ChartQA smoke; a larger pilot is required.**
3. Does risk-plus-gain improve the accuracy-cost frontier with positive paired CI? **Pending.**
4. Does the result hold on at least two of ChartQA, HRBench, and DocVQA? **Pending.**
5. Is the 5--80% nontrivial acquisition-rate condition met? **Pending.**
6. Is a more complex/RL stage justified? **No authorization unless all GO conditions pass.**
