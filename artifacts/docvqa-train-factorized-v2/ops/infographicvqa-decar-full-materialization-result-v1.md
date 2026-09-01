# InfographicVQA DECAR full materialization result v1

Status: passed on 2026-09-01. This is an input/fold engineering result, not a
scientific endpoint. Official validation and test remain absent and sealed.

## Execution

- Successful Slurm job: `200049` on one H800 (`q-h800`).
- Runtime: 852 seconds; queue wait: 2 seconds; restarts: 0; exit code: 0.
- Tracked revision: `cc6faa9979edf8f2c97d49e9b61b24b05957dfc8`.
- Decoder: Pillow `12.1.1`, matching the frozen source audit and pilot.
- All-state email was configured for
  `yihangc@connect.hku.hk`.

Initial job `200046` failed closed after five seconds and published no output.
Its decoder-version cause and the outcome-blind correction are recorded in
`infographicvqa-decar-full-materialization-freeze-v1.md`.

## Verified population and exclusions

- 23,946 task rows, 4,406 image-manifest rows/files, 4,406 decoded-RGB image
  identities, and 2,204 source components.
- Outer question counts: `4790, 4789, 4789, 4789, 4789`.
- Outer source counts: `435, 441, 442, 442, 444`.
- Exactly 8,816 inner source-context rows; every outer-test source is absent
  from its inner context.
- Encoded and decoded image hashes, fold identity coverage, and source
  disjointness all passed exactly.
- No task outcome or teacher likelihood was computed. No validation or test
  row was read. Fold IDs and registered forbidden fields are absent from the
  task manifest.

## Immutable outputs

```text
b78a024cb623b17bb8cb73416b3c62f78b140e2e3c3b9737e1dde38bdfe3d254  task-manifest.jsonl
0916a6b5a32e15c4f5b3bf920e1ecd4f304aeb97ae186e0e3e846391e2304203  image-manifest.jsonl
c3af91895419ffb6134bcbb37662197c5cfd63d77cc6e12b77044aa9b6e0281f  report.json
b873b5bffc3ebf2f64e353afbfdd058608165069cab6d0387412f56e20be921b  complete.json
334c7d51b85cea6d440605afd67b8db6468d07db90be6bbde4976eb9c5c9d638  execution.json
```

The materialized bank is therefore accepted as the registered official-train
input for the full DECAR sibling rollout. It does not establish that DECAR is
scientifically successful.
