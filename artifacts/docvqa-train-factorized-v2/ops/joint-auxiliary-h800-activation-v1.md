# Joint auxiliary proposer H800 activation v1

Status: prepared on 2026-09-01 after the implementation-only 1-epoch smoke and
before the registered 200-epoch result existed.

The smoke exercised the complete 3,500-source join, five OOF folds, incumbent
reconstruction, metrics, and serialization using one epoch and 100 bootstrap
resamples. Its metrics are non-scientific and were not used to modify the
registered architecture, targets, optimizer, hyperparameters, pass rule, or
protected-role boundary. The only post-smoke change removed the outcome-derived
oracle action from serialized prediction rows and made the formal runner fail
closed unless it sees H800 and 20,000 resamples.

## Bound implementation

- local commit: `6431c65a6f0ace4b53af35d575ce6ea5214340b0`;
- fit runner SHA-256:
  `a0f2d54a5285369a427466ee4613442ee157b63a7d627ae1e133ad872699bec7`;
- model/analysis module SHA-256:
  `0cd055874be436cecc06052916c7b3d0bc4b145d2712aa66dd05bdd7438bd30c`;
- focused test SHA-256:
  `8cf005aba4d2d08bca5ce230ec3894416d7344f5963d690f0eb9ff15f02d71b3`;
- protocol SHA-256:
  `8a6e8914c9d71e90c86c74079022983c215fa70fb477071a7acd54a204425cea`.

The full existing test suite passed at this commit, and all three Torch-focused
tests passed in the frozen Qwen environment.

## Live resource decision

At activation, the account had used 39,656 of 42,000 GPU-minutes, leaving
2,344. There were no user jobs. The live `q-h800` partition reported three
eight-H800 nodes and 168 idle CPUs. The request is one H800, eight CPUs, 64 GiB
memory, and two hours, reserving at most 120 GPU-minutes and leaving more than
19 times the requested reserve.

The worker verifies the clean tracked commit and every code/protocol/input
hash, reruns compile and focused tests, records the accelerator identity, and
refuses a non-H800 GPU or existing output directory. It uses
`--mail-type=ALL` for `yihangc@connect.hku.hk`.

No GitHub push is authorized by this activation.

## Pre-result interruption and recovery

Job `199847` started immediately on one H800 and passed all hash, revision,
accelerator, compile, and focused-test checks. It then failed on the first CUDA
linear operation, before any fold result or output directory existed. PyTorch
deterministic mode correctly refused cuBLAS because CUDA requires
`CUBLAS_WORKSPACE_CONFIG=:4096:8` or `:16:8` to be set before the process.

Recovery adds only `CUBLAS_WORKSPACE_CONFIG=:4096:8` to the ignored operational
worker and a deterministic CUDA forward/backward preflight. It changes no
tracked code, input, model, target, seed, fold, epoch, optimizer, bootstrap, or
advancement condition. The failed job used `--mail-type=ALL` and its terminal
state was `FAILED`, `ExitCode=1:0` after 41 seconds.
