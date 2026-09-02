# InfographicVQA relative-where evaluation recovery v1

Status: frozen after job 203099 failed and before any recovery submission.

Job 203099 completed all 20 source-OOF fits, then failed closed while checking
that recomputed comparator aggregates exactly matched the frozen hybrid and
oracle evaluations. The failure was not an OOM, timeout, model-fit failure, or
scientific gate result. Validation and test data were not opened.

Bound failure evidence:

```text
job:                    203099
state:                  FAILED (NonZeroExitCode, ExitCode 1:0)
log SHA-256:            1277c06f98a14ffbd8cfddb4a833c87a1a2a38de149cf786c87c6621e6f00def
predictions SHA-256:    94e292d33fe9e55e9d50d53eef5f2590c4d5c751567feb2fe736867fc7dd6b6b
fit audit SHA-256:      256c34ad9d370107950f9edf915c4a65337bf35df0b198fcb6bccf02d56319af
fit report SHA-256:     f164e1481e09f3bc9be7450b7fc82fd682e0b1177b3c696db264c366c2d0202a
fit complete SHA-256:   700170914af0e5721479fdd5594696cd872ac4f49ed5fcd5b6bd14649410b677
prediction rows:        23,946
prediction outcomes:    absent
validation/test inputs: absent
```

Recovery changes only evaluator control flow and diagnostics:

1. Recomputed frozen comparator aggregates are checked immediately after
   aggregation, before bootstrap.
2. A mismatch still aborts; its first exact field path and values are reported.
   No tolerance is added and no comparator is removed.
3. If every comparator matches, the evaluator runs the same frozen 20,000
   source-bootstrap resamples, endpoints, qualification rules, and selection
   rule from the relative-where OOF protocol.
4. The existing fixed predictions are reused byte-for-byte. No model is refit,
   no hyperparameter is selected, and no failed output directory is overwritten.
5. Output is written atomically to `evaluation-recovery-v1` only after the full
   evaluator succeeds. A positive train decision still authorizes only a newly
   frozen validation protocol; it does not open validation automatically.

Resources: `debug`, one reserved RTX 4090 hidden from the CPU evaluator, four
CPUs, 64 GiB RAM, 45 minutes. Slurm mail type is `ALL` for
`yihangc@connect.hku.hk`. No GitHub push is authorized.
