# InfographicVQA DECAR OOF Slurm resource amendment

Status: operational correction before any official-train OOF fit, prediction,
or scientific endpoint evaluation. Validation and test remain sealed.

On 2026-09-01, the frozen submitter's Slurm admission test predicted job ID
`202901`, but the immediately following real submission was rejected with
`Requested node configuration is not available`. A first diagnosis based on
the truncated `sinfo` memory column led to an unexecuted 192 GiB amendment;
that admission test predicted `202903`, and real submission was rejected with
the same message. Slurm confirms that neither predicted ID exists, so no OOF
job started and no OOF output was created.

The authoritative full node records show three H800 nodes, each with 96
schedulable CPU cores, `2064037 MiB` of configured memory, eight H800s, and
`RestrictedCoresPerGPU=12`. Thus requesting 32 CPUs with one H800 is
physically unsatisfiable under the GPU-core binding. The partition also sets
`MaxMemPerCPU=20480 MiB`, so 12 CPUs can request at most 240 GiB. A second
correction that restored 384 GiB while lowering CPUs to 12 was therefore also
invalid; its admission prediction was `202912`, and Slurm confirms that this
ID does not exist either. The partition itself is up and permits the requested
four-hour limit.

Replace the worker's impossible 32-CPU, 384-GiB combination with 12 CPUs and
192 GiB. This is 16 GiB per CPU, below the partition maximum, and exactly
preserves the successful full-shape benchmark's eight-CPU, 128-GiB ratio while
adding a 50-percent reserve to both resources. The worker itself caps OMP and
MKL at eight threads. Keep one H800, four hours, all state-change emails, 65
deterministic fits, 200 epochs, source folds, input
hashes, prediction-without-outcomes boundary, 20,000 source-bootstrap draws,
comparators, and the advancement rule unchanged.

The submitter passes this amendment's SHA-256 to the worker, and the worker
verifies it before reading task inputs. The original evaluation freeze remains
the immutable scientific contract; this amendment supersedes only its
impossible CPU-memory resource combination and historical worker-file hash.
No GitHub push is authorized.
