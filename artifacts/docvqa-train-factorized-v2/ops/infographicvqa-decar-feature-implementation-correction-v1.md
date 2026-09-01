# InfographicVQA DECAR inference-feature implementation correction v1

Status: frozen on 2026-09-01 before the full rollout and before any pilot task
endpoint was read for method selection. This corrects an implementation
omission; it does not change the registered feature family or method.

## Omission

The registered protocol requires four answer-now generation summaries:
generated-token count, mean and maximum normalized token entropy, and mean
selected-token log probability. Pilot-v1 rollout metadata serialized the
first three quantities but did not serialize selected-token log
probabilities. Consequently the frozen 16-scalar DECAR input could not be
constructed exactly from that pilot bank.

The omission was found by inspecting schemas and tensor shapes only. No pilot
task outcome, policy endpoint, or advancement statistic was used. The DECAR
join fails closed when any per-token log probability or its mean is absent,
misaligned, non-finite, or inconsistent.

## Correction

`Qwen25VLBackend` now computes, from the same generation-step logits already
used for entropy, the log-softmax value of each actually generated token. It
serializes the aligned per-token list and its arithmetic mean. Generation,
model, prompt, image/action family, entropy calculation, targets, cost,
hardware class, folds, architectures, losses, and advancement rule are
unchanged.

```text
937dcd29deed4e671b4969a30b8521b685c326619fbf907f673240853b25ac3d  src/beyond_entropy/qwen_backend.py
ede9e208b7d7b56ead155fcd389e47a5f81d332f6863f8c4b739cb7e207007ef  tests/test_qwen_runtime.py
```

The qwen-vl runtime smoke passed under PyTorch `2.4.0+cu121`. After formatting,
the exact hashes above were revalidated in the full nested OOF engineering
smoke on H800 as Slurm job `200068` in five seconds, with zero restarts and
exit code zero. That synthetic smoke read no InfographicVQA task endpoint.

Pilot-v1 remains a valid throughput measurement, but its rollout bytes are not
valid DECAR scientific inputs. Before the full rollout, rerun the registered
512-source pilot with this corrected metadata and require exact schema,
checkpoint/resume, NLL, and feature-join audits. Pilot endpoints remain
implementation-only and cannot change any frozen scientific choice.
