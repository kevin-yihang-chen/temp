# InfographicVQA DECAR OOF startup-hash correction

Status: deterministic startup correction after failed Slurm job `202915` and
before any official-train OOF fit, prediction, or scientific endpoint
evaluation. Validation and test remain sealed.

Slurm authoritatively records job `202915` as `FAILED`, `ExitCode=2:0`, with an
11-second runtime and zero restarts. It allocated one H800, 12 CPUs, and 192
GiB, then stopped at the pre-input `outer-folds` SHA-256 check. Its one-line
log is `DECAR OOF outer-folds SHA-256 mismatch`; the log SHA-256 is
`afadd2d4e3432cadf3cd1360baf74d0b4914eeb0eae213b059bdc42a69aaf8e7`.
No fit, evaluation, or OOF execution output directory was created.

The evaluation freeze and OOF worker contain the 71-character literal
`7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6b8cdb6d0af5a4da60`,
which cannot be a SHA-256. The immutable allocation completion record binds
`outer-folds.jsonl` to the actual 64-character SHA-256
`7f0f23e65d155e728b592a96b5d5a463d67cfed9742977532535e6a232b0837a`.
The successful full-materialization worker also uses that exact value. Current
file hashing reproduces it.

Replace only the two malformed outer-fold SHA literals in the OOF worker: the
startup `require_hash` value and the fit runner's
`--expected-outer-folds-sha256` value. Do not modify the allocation file,
source folds, input population, model, training schedule, comparator set,
bootstrap, advancement rule, validation boundary, or test boundary.

Bound evidence before correction:

```text
aa9c4be09d58c7a997ec3937ceb2f5f9389a1b1c4d6aab0c2b89a1c55041617b  allocation-v1/complete.json
fdf5e5139dcab4b04f824805d1d2989cb6c61ea5d4317d3fa6fe647942e1886c  allocation-v1/report.json
1eaa1e329a7de5a55881f4031bdfa02641bbc149fd68c104630bbdf9d4fe75af  full-materialization worker
6e1efc836d8b766f8d05b9e3462fad3ade379d8d1297df6c5696456bbb442ce1  evaluation freeze
1d17d734a0582c35f55fd95ac08967d90092b2f929dc0aa1408b224a0f41ff23  failed OOF worker
```

The submitter passes this correction's SHA-256 to the worker, and the worker
verifies it before reading task inputs. The original evaluation freeze remains
the immutable scientific record; this document corrects only an invalid hash
transcription. All state-change emails remain enabled. No GitHub push is
authorized.
