# InfographicVQA DECAR full materialization freeze v1

Status: frozen on 2026-09-01 after the registered 512-source engineering
pilot passed, but before reading the remaining full-train question text,
answers, or model endpoints. Official validation and test remain absent and
sealed.

## Bound inputs

```text
ecc46c6a073ebd89fc114cba6fee5c711c8600e596b5a785bec981d98b168f13  download-manifest.json
fc577513dd8f9993f40d14454c7ec4ecf48897ff0d1660479fb5c49d3ae9512a  source-manifest.jsonl
7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6a232b0837a  outer-folds.jsonl
8b4977776841794185eedd8cbbfdb8e23ec2c44cc0be9931cc2e672b232f344c  inner-folds.jsonl
d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342  infographicvqa-decar-method-protocol-v1.md
3d0948cc6840b008cd4b19408ff002ed0756bb0d9f7f5e6b8cdb6d0af5a4da60  infographicvqa-decar-allocation-result-v1.md
d91f756e82ee2ce58edf4a66b3fde3433d0f2466cc72f04994d758dc1c23f697  infographicvqa-decar-pilot-result-v1.md
```

The only transport files are the 24 already pinned official-train parquets at
revision `539088ef8a8ada01ac8e2e6d4e372586748a265e`, with aggregate size
1,981,251,656 bytes. The materializer fails if any file, footer row count,
encoded image, decoded RGB image, source assignment, fold count, or bound hash
changes.

## Frozen implementation

```text
6974d3a2a157e04935c80a9bd9344bb9c2c88fe4291fb13b321364f9e513b639  src/beyond_entropy/infographicvqa_decar_manifest.py
7051b2c68a18caab112e519cee708d26f74cac6d3f05b73c53ded20985f4be7f  scripts/materialize_infographicvqa_decar_full.py
33d1f5213c5864370027f2c1f488a8f0470c72a2ee3a8ee9bcaac0efb2887a0e  tests/test_infographicvqa_decar_manifest.py
808892a99db7ec4c69e5c29bcd0a1c8c0ed9398f963648030e539ab7bd24fe5f  scripts/slurm_infographicvqa_decar_full_materialize.sh
f9b6a86fc547b4cd610a5c90bc04d81c7d42938ad3a3b7daac4c66396d0c4010  scripts/submit_infographicvqa_decar_full_materialize.sh
```

The focused materializer tests, focused mypy, Python compilation, shell
syntax, whitespace audit, and the complete repository regression suite passed
before this freeze.

## Output and leakage contract

The atomic output directory is
`artifacts/infographicvqa-train-v1/decar-v1/full-manifest-v1`. It must contain:

- exactly 23,946 task rows with unique state/question IDs;
- exactly 2,204 source components and 4,406 decoded-RGB image IDs;
- exactly 4,406 hash-addressed encoded image files and image-manifest rows;
- a report proving exact five-outer/four-inner source exclusion over all 8,816
  registered inner source-context rows; and
- a completion manifest and Slurm execution record with immutable hashes.

The task manifest contains only `state_id`, `question_id`, `image_id`,
`source_id`, `image_path`, `question`, `model_prompt`, and `target`. It must not
contain hostname, transport filename/row, outer/inner fold IDs, answer type,
OCR, or operation/reasoning fields. Fold identities remain in the separately
hashed allocation files and cannot become inference inputs.

Only `questionId`, `question`, `answers`, `image`, and `data_split` are read
from the train parquets. Materialization computes no task outcome, teacher
likelihood, rollout, feature, threshold, policy, or diagnostic. It may not read
or download validation or test rows.

## Execution

Run one 30-minute `q-h800` job with one H800, 12 CPUs, and 96 GiB. The GPU is
requested only to use the cluster's submitted compute path; this stage loads no
Qwen parameters. The submitter requires a clean tracked revision, a free sole
job slot, at least 30 remaining GPU-minutes, a successful Slurm admission test,
and all-state email to `yihangc@connect.hku.hk`.

No credential is exported to the job. No GitHub push is authorized.
