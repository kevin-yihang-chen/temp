# InfographicVQA DECAR identity-allocation result v1

Status: accepted and frozen on 2026-09-01 before any InfographicVQA question
text, answer, task outcome, teacher likelihood, or Qwen task endpoint was read
or computed.

## Bound implementation and inputs

- Code revision: `3393f493e3274f88fdc1fe41ac95caa41ddcbf18`.
- Allocation module SHA-256:
  `acaa4d09a10675ca51663b16cbf8e4bc57710100e40526e947d762fd2f431407`.
- Runner SHA-256:
  `35c54ecfb13f5a35d02d08ebf9de31dd1da67a43572dfbec61ce66e89d3e60fe`.
- Test SHA-256:
  `7e17f670f9617f1e74e31048b7b727925531042926e6e9ba868288d72cc78e9b`.
- Method protocol SHA-256:
  `d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342`.
- Source manifest SHA-256:
  `fc577513dd8f9993f40d14454c7ec4ecf48897ff0d1660479fb5c49d3ae9512a`.
- Pilot-source manifest SHA-256:
  `75f20c141ccc273dcc36a4527ec7697826e3fea4b2bfc110754027ad9bb9ffe3`.

Targeted allocation/source-audit tests passed (`5 passed`), focused mypy
passed, and the complete repository suite passed (`406 passed, 19 skipped`)
before publication.

The first direct invocation stopped at Python import before opening either
input because the login-node base environment did not have the repository
`src` path installed. No output directory was created. The identical committed
runner then completed with the repository `src` path exposed, as every existing
Slurm worker does. No algorithm, input, or scientific setting changed.

## Frozen outputs

- Allocation report SHA-256:
  `fdf5e5139dcab4b04f824805d1d2989cb6c61ea5d4317d3fa6fe647942e1886c`.
- Outer-fold manifest SHA-256:
  `7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6a232b0837a`.
- Inner-fold manifest SHA-256:
  `8b4977776841794185eedd8cbbfdb8e23ec2c44cc0be9931cc2e672b232f344c`.
- Pilot-question manifest SHA-256:
  `fd5de6036f685b4b03d560fc59959196b9e9b96aba2fce70a80353b5adfc2388`.
- Completion SHA-256:
  `aa9c4be09d58c7a997ec3937ceb2f5f9389a1b1c4d6aab0c2b89a1c55041617b`.

The outer folds contain 2,204 unique source components and all 23,946
questions. Their question/source counts are:

| Outer fold | Questions | Sources |
|---:|---:|---:|
| 0 | 4,790 | 435 |
| 1 | 4,789 | 441 |
| 2 | 4,789 | 442 |
| 3 | 4,789 | 442 |
| 4 | 4,789 | 444 |

For every outer-test fold, the four inner folds contain only the other sources.
Their question counts are 4,789 or 4,790, with 8,816 total context-specific
source assignments, exactly `2,204 * 4`. No source is divided to improve this
balance.

The engineering pilot contains exactly one identity-only question selection
from each of the frozen 512 pilot sources. The manifest contains transport
coordinates, dimensions, image hashes, and identifiers only. The three output
manifests contain no question text, answer, target, answer type, OCR,
operation/reasoning, correctness, reward, likelihood, entropy, or Qwen output.
The report records all protected-field access flags as false.

## Next authorized access

The committed method protocol and frozen identity allocation now authorize
materializing only the 512 registered train-pilot questions and answers for
engineering validation. Pilot endpoint values may not change any method or
hardware choice. Full train materialization and model endpoints remain a later
step; official validation/test and ScreenQA protected roles remain sealed. No
GitHub push is authorized.
