# InfographicVQA DECAR OOF Slurm resource amendment

Status: operational correction before any official-train OOF fit, prediction,
or scientific endpoint evaluation. Validation and test remain sealed.

On 2026-09-01, the frozen submitter's Slurm admission test predicted job ID
`202901`, but the immediately following real submission was rejected with
`Requested node configuration is not available`. Slurm confirms that `202901`
does not exist, so no OOF job started and no OOF output was created.

The live `q-h800` inventory has three H800 nodes, each with 112 logical CPUs
and `206403 MiB` of configured memory. The frozen worker requested 384 GiB on
one node, which is physically unsatisfiable. The partition itself is up and
permits the requested four-hour limit and 32 CPUs.

Replace only the worker memory request from 384 GiB to 192 GiB. This fits below
the node limit and retains a 50-percent reserve above the successful 128 GiB
full-shape runtime benchmark request. Keep one H800, 32 CPUs, four hours, all
state-change emails, 65 deterministic fits, 200 epochs, source folds, input
hashes, prediction-without-outcomes boundary, 20,000 source-bootstrap draws,
comparators, and the advancement rule unchanged.

The submitter passes this amendment's SHA-256 to the worker, and the worker
verifies it before reading task inputs. The original evaluation freeze remains
the immutable scientific contract; this amendment supersedes only its
impossible 384 GiB Slurm request and historical worker-file hash. No GitHub
push is authorized.
