# InfographicVQA DECAR corrected Qwen-7B pilot freeze v2

Status: frozen on 2026-09-01 after the answer-now selected-token log-prob
implementation correction and before rerunning the registered 512-source
pilot. This is an implementation-only rerun. Its endpoints cannot select or
change any scientific choice.

## Bound scientific inputs

The population, source selection, actor, prompts, four UG actions, targets,
cost, folds, NLL definition, semantic embeddings, hardware class, optimizer,
baselines, and advancement rule are identical to pilot v1.

```text
d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342  infographicvqa-decar-method-protocol-v1.md
42f671576b4ff1dd91f09fa67627984ffa2758d84bc58803fd94c7cf27804142  infographicvqa-decar-pilot-implementation-freeze-v1.md
22a7e9046dcd7e949aee2d725a068f7b9cd0b5a3476130e8e9c50818bf158d46  infographicvqa-decar-feature-implementation-correction-v1.md
80067cc1446782f458665d8ddfa98745bda73b03b9eb96da3528f82f22158d29  pilot task-manifest.jsonl
9b28285892d43290b898eefa9bca3abef79f40a248323c84c4bce0df5b52562a  pilot materialization complete.json
```

The output root is new and empty:
`artifacts/infographicvqa-train-v1/decar-v1/pilot-qwen7b-v2`. No pilot-v1
rollout, NLL, feature, or diagnostic output may be resumed or copied into v2.

## Frozen delta implementation

```text
937dcd29deed4e671b4969a30b8521b685c326619fbf907f673240853b25ac3d  src/beyond_entropy/qwen_backend.py
5729228f02ac5fa316f9a8549acedec0643c14ef555455a8fa61b20b79c260ce  src/beyond_entropy/infographicvqa_decar.py
32a1b54de5dd31aade3d118e3ef348d0e9a1668d0bcbff6adaaad8fe39d61d6a  scripts/audit_infographicvqa_decar_inputs.py
0783e8ef2a27e800b8d3a3f07a057e03526fc429aadbebf119e13f74c6e2a4d7  scripts/slurm_infographicvqa_decar_pilot_h800.sh
912a552b14ff5782cfe2172017a74a2e881e4ca955b1888ea5266cedd93f1a79  scripts/submit_infographicvqa_decar_pilot_h800.sh
7dabd9cf738eb461b8a08d90263e124a5b59250e46eb8e8b65c682f533529bcc  tests/test_infographicvqa_decar_pilot_contract.py
```

Unchanged rollout merge, NLL scoring/merge, semantic extraction/merge, and
label-free audit implementations remain bound through the v1 freeze. Their
hashes are rechecked by the worker or by their own provenance contracts.

## Additional v2 acceptance condition

After the canonical rollout, NLL, and label-free feature merges, the worker
runs the strict DECAR join over all 512 decisions. It requires exactly one
ANSWER and four registered UG actions per decision, exact NLL and semantic
coverage, the 16 registered inference scalars, aligned finite per-token
entropies/log probabilities, outcome-free semantic storage, and the expected
question/global/ROI tensor shapes. The audit reports schema, counts, names,
and dimensions only; it reports no task/NLL/policy endpoint.

Applying this strict join to pilot v1 fails closed with
`DECAR generated-token statistics are incomplete`, confirming that v1 cannot
silently enter a scientific fit. The v2 pilot passes only if the joined-input
audit succeeds and its hash is stored in the execution record.

## Execution and security

Run the same four-H800 offline job, complete-state checkpoint/resume tests,
and runtime hardware audits as pilot v1. The submitter requires a clean
tracked revision, at least 720 remaining GPU-minutes, Slurm admission success,
an empty v2 output root, and `--export=NONE`. Slurm sends all supported state
emails to `yihangc@connect.hku.hk`.

No credential is exported. Validation/test remain sealed. No GitHub push is
authorized.
