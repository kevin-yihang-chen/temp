# Cross-benchmark scorer protocol

This protocol prepares independent evaluation on document, scene-text, and
high-resolution perception tasks without using their labels to change the
frozen ChartQAPro experiment.

## Pinned reference

- Repository: `ExplainableML/ug-framework`
- Commit: `13050ee49865e4330519108f42d1ccfccff1aee1`
- Inspected task adapters: `docvqa`, `textvqa`, and `hrbench`
- The local compatibility implementation has no runtime dependency on the
  reference repository.

## Frozen semantics

- **DocVQA:** append the released short-answer suffix and score the maximum
  answer normalized Levenshtein similarity. Preserve the released boundary
  rule (similarity `0.5` is retained) and its raw-string length denominator.
- **TextVQA:** expose the released OCR-token reference prompt, normalize both
  prediction and ten human answers with the pinned `EvalAIAnswerProcessor`
  semantics, and use leave-one-annotator-out soft accuracy.
- **HRBench:** present lettered options and request the option letter directly.
  Use the released rule-based A--D extraction. Report both micro accuracy and
  the benchmark's cycle-category macro average in downstream analysis.

Targets are stored only in `GroundTruth`; prompt builders accept no answer
argument. Dataset manifests must keep the gate-visible core question separate
from the backend-only formatted prompt.

The exporter pins public dataset revisions and groups related questions by
document/image source. HRBench 4K and 8K rows share `source_id = hrbench:<index>`
so paired resolutions cannot leak across a source-disjoint split. Selection
strata use only pre-outcome fields: DocVQA question types, TextVQA OCR-token
count buckets, and HRBench category plus cycle category.

| Task | Dataset/config | Split | Pinned revision |
| --- | --- | --- | --- |
| DocVQA | `lmms-lab/DocVQA` / `DocVQA` | `validation` | `539088ef8a8ada01ac8e2e6d4e372586748a265e` |
| TextVQA | `lmms-lab/textvqa` | `validation` | `9c0699cd19768ac5ab97568f6b3cbac4c0062884` |
| HRBench 4K | `DreamMr/HR-Bench` / `hrbench_version_split` | `hrbench_4k` | `83b9013d6293b85dc507e87199ca52517536939c` |
| HRBench 8K | `DreamMr/HR-Bench` / `hrbench_version_split` | `hrbench_8k` | `83b9013d6293b85dc507e87199ca52517536939c` |

## Compatibility audit

Run:

```bash
PYTHONPATH=src python scripts/audit_cross_benchmark_scorers.py \
  --reference-root data/external/ug-framework-13050ee
```

The frozen audit in `artifacts/cross-benchmark-scorer-compat-v1/report.json`
passes 8 DocVQA ANLS cases, 154 TextVQA normalization cases, all 11 possible
TextVQA annotator-match counts, and 9 HRBench extraction formats. The report
also binds the SHA-256 digest of every inspected reference file.

## Scientific use

These adapters are infrastructure, not benchmark evidence. Dataset selection,
development/formal partitioning, candidate actions, cost, and success criteria
must be registered before any new labels or rollout outcomes are inspected.
