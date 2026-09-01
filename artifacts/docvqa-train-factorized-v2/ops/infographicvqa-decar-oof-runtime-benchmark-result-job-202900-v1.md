# InfographicVQA DECAR full-shape OOF runtime benchmark result

Status: passed on 2026-09-01 before official-train OOF fitting, prediction, or
scientific endpoint evaluation.  This result may affect only the registered
OOF Slurm wall-time.

## Execution contract

Slurm authoritatively recorded job `202900` as `COMPLETED`, `ExitCode=0:0`,
with zero restarts.  It ran for 11 seconds on one NVIDIA H800 with eight CPUs,
128 GiB requested memory, and a 45-minute limit.  Queue wait was one second,
and all supported state-change emails were bound to
`yihangc@connect.hku.hk`.

The job used code revision
`78f9a3fad31178c99838e576d611215cdda4b28f`.  The report has exactly the
frozen full shape: 23,946 decisions, 2,204 sources, embedding dimension 3,584,
four candidates, 16 scalar features, five measured epochs, and the registered
65-fit schedule at 200 epochs.

The persisted report is semantically identical to the JSON in the Slurm log,
and the worker-recorded report SHA-256 matches the file.  Every measured fit
time and peak-memory value is positive.  The report asserts synthetic inputs
only, no task outcome read, no scientific endpoint computed, no validation or
test input used, and no credential present.

Bound evidence:

```text
c3af4bb9befc666a880c26f24bc26029a8f2d6c4e816beebfb6899ce469da2ae  full-shape report.json
8cdf64d2f4a99489a9de0fd64b3ffa5c1809ff2473488bb3a57a27829c8d81b6  slurm-infovqa-decar-fitbench-202900.out
c88bddebd701ff0fbed7b6fed03d3b094f5de489ba4d38bb305b30f90829bfcb  corrected benchmark runner
cb1673ad57440788936f34890e669c66483f43fcd8b2e782451fc586dd83205f  H800 CUDA smoke result
635b31b897c85907c3602204bea4dcad60344e16d891eb2aff30f3675fbff837  CUDA-init correction v2
e8d00817171c96d27410db0b6069839e8417740c865dc1b87c3be9ad920bf30f  runtime benchmark freeze v1
```

## Projection

Measured five-epoch fit times were:

```text
where_inner         2.2949864328838885 s
where_outer         1.4712000079452991 s
when_ternary_outer  0.7350894210394472 s
when_binary_outer   0.7022224839311093 s
```

Scaling those measurements by `200/5` and the exact registered fit counts
projects `4694.938560994342` seconds, or `1.304149600276206` hours.  Applying
the frozen 25-percent reserve gives `5868.6732012429275` seconds, or
`1.6301870003452577` hours.

## Wall-time decision

The existing OOF worker requests four hours.  Because the reserved projection
is below four hours, retain the four-hour request unchanged.  The benchmark
protocol permits only retaining or increasing this limit; it does not permit
shrinking the scientific fit schedule or using the short benchmark wall time
as if it were an OOF result.

The unchanged one-H800 nested-OOF job is now authorized.  It remains bound to
the frozen 65 fits, 200 epochs, source folds, prediction-without-outcomes
contract, 20,000 source bootstrap draws, comparators, and advancement rule.
Validation and test remain sealed.
