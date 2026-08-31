# Qwen2.5-VL-7B backbone replication decision

Decision: **strong_backbone_replication**.

This mechanically applies the four conditions frozen before the full
512-state Qwen2.5-VL-7B result was observed. It selects no threshold
or call rate and does not authorize opening a protected role.

## Conditions

- `answer_loss_spearman_ci_low_above_zero`: **PASS** - ci_low=0.04467775
- `answer_loss_top_one_gain_ci_low_above_zero`: **PASS** - ci_low=0.00390625
- `answer_loss_top_one_gain_exceeds_entropy_and_random`: **PASS** - answer=0.01757812, entropy=0.00390625, random=-0.00390625
- `answer_loss_top_one_harm_below_entropy_and_random`: **PASS** - answer=0.00390625, entropy=0.01562500, random=0.02197266

## Boundary

The frozen Qwen2.5-VL-7B mechanism gate passed. This supports only persistence of the proxy hierarchy across the Qwen 3B/7B scale on opened ScreenQA development sources.
