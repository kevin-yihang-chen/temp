# InfographicVQA DECAR Qwen-7B pilot implementation freeze v1

Status: frozen on 2026-09-01 after the registered 512-source pilot was
materialized, but before any Qwen task answer, entropy, ANLS, correctness,
teacher likelihood, or pre-action semantic feature was computed on it.

This is an implementation-only pilot. Its endpoint values may not select or
change a model, hardware class, method, population, feature, hyperparameter,
threshold, action, cost, baseline, or advancement rule.

## Frozen scientific inputs

- DECAR method protocol SHA-256:
  `d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342`.
- Allocation-result SHA-256:
  `3d0948cc6840b008cd4b19408ff002ed0756bb0d9f7f5e6b8cdb6d0af5a4da60`.
- Pilot-materialization result SHA-256:
  `7b5a73fa8fad96eae542c74a3abf4a8c5687e4b0edb68a376a5a554956358345`.
- Pilot task manifest SHA-256:
  `80067cc1446782f458665d8ddfa98745bda73b03b9eb96da3528f82f22158d29`.
- Pilot materialization completion SHA-256:
  `9b28285892d43290b898eefa9bca3abef79f40a248323c84c4bce0df5b52562a`.
- Population: exactly 512 questions, 512 source components, 512 images, one
  ANSWER plus four deterministic UG-grid ZOOM siblings per question.

## Frozen execution code

- Four-H800 worker SHA-256:
  `1cfc8c85f84b146e3708f1679918f4526cfbf99b0f3e44ab908fbe28231d1642`.
- Quota-gated submitter SHA-256:
  `8ef4824f175e7a4d66d0308db10aa5832e08e2a4ec64b3c27ea1c1c0845ae8ac`.
- Static contract test SHA-256:
  `0d3ec3e55ff85b7a4441fdf2f3dc585448db88dc6e541d21f75025fdc5cc594f`.
- CLI SHA-256:
  `6512131e7a9bbe55b65f9229a044df43e0fa9c4564e4c20fca060a2a17059346`.
- Qwen backend SHA-256:
  `5ee063fb3d8abe3461186e7185960afd002848f1f31aad7b1fdbc1fc53840acb`.
- Rollout module SHA-256:
  `b4e30265e3b0d9bd69119ffd32901679ccd2b59140d7785c299e14465deff455`.
- Crop proposer SHA-256:
  `ddbd23e1f3e7930f1ae187aa325f0a26406e8cfa78fbf65a525ea43a22b138bf`.
- Benchmark loader/scorer SHA-256:
  `d96d95f3814209822f724f131175058dda7044f6fb70b402aae27a806d1a30fc`.
- DocVQA prompt/ANLS SHA-256:
  `8022db85d17d99246a1058bc2bf49153c478f827f44eda2221bca99d26410e69`.
- Answer-likelihood module SHA-256:
  `afcf8ec83e513d855532bf64b7ecc61911a21776b005220d4ec2f8a64e18f470`.
- Semantic extractor/module SHA-256:
  `6e94320c8bf54c982072e06b11950a3758b7b39fcd3975e619729caf3860b3b6` /
  `fa4a1aafe9f8e4eafa635c311bae7620311036af3e1c18cdf382f1bdac7d7606`.
- Rollout merger module/runner SHA-256:
  `b480e939017774dcd5dab483eeb5864425b046468dbe2356d006408063d347b5` /
  `5ddd3fcbff9d21f036c75efa8591ab70e3cd9a311e7bd6d679dafcb251061744`.
- NLL scorer/merger SHA-256:
  `230e1cf2d8e264d9092c0b1c390dbd29029049635911455757d52f3ad9062be4` /
  `4e5c8f2a97e9bdfed835f592e6cc9e52138134e4b6d1cfcd855c013b05f5974d`.
- Semantic merger/label-free auditor SHA-256:
  `3b1051ea28b07a5aefd70c4c347c43410c1023cc35eed739216dc0d0d1d3ff30` /
  `7c3e84b13962e61dcf0a3182c1c262e6b01f67bac4e13fe88f3ac22ef3f6be30`.

Both shell scripts pass `bash -n`; two pilot-specific contract tests pass; the
full repository suite passes (`410 passed, 19 skipped`). The submitter requires
a clean tracked tree, records the exact execution revision, verifies at least
720 remaining GPU-minutes, runs a Slurm `--test-only` admission check, and
passes only five non-secret positional arguments under `--export=NONE`.

## Frozen actor and stages

All model stages use the locally cached
`Qwen/Qwen2.5-VL-7B-Instruct` revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, bfloat16, SDPA, deterministic
seed zero, `min_pixels=200704`, `max_pixels=602112`, no quantization, no
offload, and no network access.

On four H800s the worker performs:

1. four source-disjoint rollout shards, with one ANSWER and four UG crops,
   complete-state checkpoints every eight decisions, and a byte-identical
   complete-resume audit;
2. a canonical 2,560-record rollout merge;
3. four 128-decision teacher-NLL shards, with no raw target serialization and a
   byte-identical complete-resume audit;
4. a canonical 2,560-record NLL merge;
5. four label-free original-image feature shards using
   `contextual_text_mean`, ROI pooling from the original-image token grid, and
   no outcome fields; and
6. a canonical 512-decision feature merge and label-free audit.

The rollout merger requires 100 engineering bootstrap resamples because its
implementation rejects zero. This correction was made during static preflight,
before any endpoint existed. The resulting diagnostic file must not be opened
or used for any setting decision. The pilot execution record exposes only
dimensions, hashes, runtime, hardware, and selection-use flags.

## Failure and security behavior

Shard failures retain complete-state checkpoints and require an explicit
`--resume`; completed outputs are not silently overwritten. The worker verifies
H800 identity, compute capability, bfloat16/SDPA, peak allocation, source and
record coverage, raw-target exclusion, and feature outcome exclusion. A failed
contract stops before declaring pilot completion.

The job uses `HF_HUB_OFFLINE=1`, unsets Hugging Face and proxy variables, and
does not receive the submitter environment. Slurm email is configured to
`yihangc@connect.hku.hk` for all state changes. No credential appears in a
script, log, manifest, or artifact. No GitHub push is authorized.
