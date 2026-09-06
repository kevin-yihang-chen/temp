# CV Method Phase A Result — Job 209085

Decision: **PHASE_A_PASS (engineering only)**.

Job `209085` ran the frozen commit `32bd83c` on 2×RTX 4090 from
2026-09-06 14:22:25 to 14:24:55 HKT. Slurm recorded `COMPLETED`,
`ExitCode=0:0`, no restart, and 2m30s wall time. Each matched arm used one GPU;
the measured per-arm peak allocation was 9.51 GiB and elapsed training plus
validation was about 133 seconds.

Both Outcome-only and Counterfactual Utility passed all nine runtime checks:
paired reward/gain consistency, valid binary support, finite trace, lower loss
on a fixed train audit, nonzero gradient and parameter updates for the head,
vision merger and last language block, no proposed-crop execution, finite
nonconstant validation scores, and no complete natural-action collapse. Their
deterministic training schedule hashes matched and neither report accessed test.

The Outcome-only fixed audit loss changed from `0.685516` to `0.661889`; the
Counterfactual Utility loss changed from `0.687914` to `0.469614`. Natural
CONTINUE rates were `22/24` and `23/24`, respectively. The high rates are a
diagnostic warning but not complete action collapse; scientific evaluation uses
an exact top-count matched call rate.

The hash-selected validation smoke has only 12 states per domain and ChartQA's
subset is entirely neutral, so its accuracy values are deliberately not used as
method evidence. Phase A authorizes the pre-registered Phase B pilot and nothing
else.

Artifact hashes:

- Outcome report: `5817e115ac909d973f4484fd7b48edf5ec2e2d3b94b6423d83752c21625ce887`
- Outcome selector: `0ca1c5dff5c0026247aa8bd6352067f2d035a1f89b02b971a228793f1c88a380`
- Counterfactual report: `7a16afce9ca50533a347fcf0bda1136f06e89be2d4a5c898afcc729bc3c21fda`
- Counterfactual selector: `e4c189e7414b8d6784a9a10a0c8fadff2f3cd70c99658f44f066df4738629281`
- Evaluation report: `71eef3c8e3ab4054b07bcb191e9c62b2b0421299ebf9106ff2aabca22a8cdfdf`

The first 2×H800 request, Job `209083`, was canceled with zero runtime after the
scheduler estimated a 2026-09-08 start. A first 4090 submission attempt was
rejected before job creation because its CPU/memory shape was unavailable. No
model or data was opened in either case.
