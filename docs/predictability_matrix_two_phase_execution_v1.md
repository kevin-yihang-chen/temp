# Predictability matrix 两阶段执行合同

本文件记录正式 36-cell 审计如何在代码层证明“先冻结、后且仅后打开 test”。它是
`configs/predictability_audit_v1.json` 的运行说明；不能修改研究问题、模型、阈值或终局
规则。

## Phase A：development-only fit 与 freeze

`scripts/freeze_predictability_matrix.py` 只接受 schema 为
`predictability_matrix_development_inputs_v1` 的 JSON。顶层及每层 key 必须精确匹配，任何
额外 `test` 字段都会被拒绝。每个 benchmark 必须提供 train 和 validation 的 manifest、
rollout、rollout provenance 与 feature 文件及其 SHA-256：

```json
{
  "schema": "predictability_matrix_development_inputs_v1",
  "code_revision": "<40-hex clean HEAD>",
  "protocol": {"path": "<absolute path>", "sha256": "<64-hex>"},
  "benchmarks": {
    "chartqa": {
      "train": {
        "manifest": {"path": "<path>", "sha256": "<sha>"},
        "rollouts": {"path": "<path>", "sha256": "<sha>"},
        "rollout_provenance": {"path": "<path>", "sha256": "<sha>"},
        "features": {"path": "<path>", "sha256": "<sha>"}
      },
      "validation": {"manifest": {}, "rollouts": {}, "rollout_provenance": {}, "features": {}}
    },
    "docvqa": {"train": {}, "validation": {}},
    "hrbench": {"train": {}, "validation": {}}
  }
}
```

空对象只表示文档省略了重复字段；真实 spec 中每个 role 必须完整。loader 会重新计算所有
hash，验证 feature metadata、rollout provenance、manifest/feature state coverage、固定
四-crop label、pre/post feature 维度、clean Git revision 和协议中的固定 Qwen execution
contract。通过后，fit 完成三 benchmark × 四 levels × 三 targets × 三 seeds、每 benchmark
三个 post-action probes 与六个强基线，并原子保存：

- 一个仅可信本地加载的 pickle bundle，必须以记录的 SHA-256 加载；
- 一个严格 JSON freeze report v2，明确 `test_data_present=false`，并列出所有 validation
  selection、每-seed primary/L3 cell、三个 seed 的多数投票 call mask 与 development
  identity digest。

调用示例：

```bash
PYTHONPATH=src /userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python \
  scripts/freeze_predictability_matrix.py \
  --input-spec <development-inputs.json> \
  --input-spec-sha256 <sha256> \
  --repo-root . \
  --model-output <new-frozen-model.pkl> \
  --report-output <new-freeze-report.json>
```

## Phase B：held-out test 一次性应用

正式 test rollouts/features 只能在 Phase A artifact 已存在并固定 hash 后生成。首先由
`scripts/build_predictability_test_transaction_plan.py` 建立不读取 test artifact 的 hash-bound
plan。唯一 Slurm worker 在验证 frozen model/report、协议、clean code revision、GPU 和离线
权重后，调用 `scripts/start_predictability_test_transaction.py` exclusive-create access ledger；
只有此后才能检查 allocation report 和 test manifests。worker 随后在同一不可自动重提的
job 内顺序生成三个 benchmark 的 rollouts/features，并分别封存 role completion。

最终 evaluator `scripts/evaluate_predictability_matrix_test_once.py` 的 test spec 使用 schema
`predictability_matrix_test_inputs_v1`，包含相同 protocol/code revision、上述 frozen model/
report、allocation report、三个 benchmark 的 test role artifacts、已有 access ledger 的
path/hash、transaction-plan hash 及唯一 output 路径。它不暴露 fit/selection 参数。

evaluator 首先验证已有 ledger 与 transaction/freeze/protocol/code/allocation hashes 一致，
随后才加载 test 文件；再检查 bundle 中保存的全部 development source/RGB identities 与
test 零重叠，执行冻结推断和 20,000 次 paired source bootstrap。旧的“先生成 test、再由
evaluator 建 ledger”顺序已被禁止。任何异常都会留下 ledger，要求人工审计，而不是悄悄
重试或换参数。

```bash
PYTHONPATH=src /userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python \
  scripts/evaluate_predictability_matrix_test_once.py \
  --input-spec <test-inputs.json> \
  --input-spec-sha256 <sha256> \
  --repo-root .
```

`run_predictability_matrix(...)` 只服务于 synthetic smoke，并在
`formal_claim_eligible=true` 时无条件拒绝一键执行。

若 machine report 满足冻结的四类规则之一，`scripts/render_predictability_audit.py` 才允许
exclusive-create 名为 `PREDICTABILITY_AUDIT.md` 的终局文件；matrix 不完整、非 formal、
缺少 one-shot ledger、split audit 失败或证据 inconclusive 时均拒绝生成。
