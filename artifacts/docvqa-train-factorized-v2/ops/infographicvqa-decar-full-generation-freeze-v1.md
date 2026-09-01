# InfographicVQA DECAR full Qwen-7B generation freeze v1

Status: frozen on 2026-09-01 after the corrected 512-source engineering
pilot passed and before full official-train rollout generation. No pilot task,
teacher-NLL, policy, validation, or test endpoint was inspected or used to
select this execution plan.

## Bound scientific inputs

```text
d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342  infographicvqa-decar-method-protocol-v1.md
b78a024cb623b17bb8cb73416b3c62f78b140e2e3c3b9737e1dde38bdfe3d254  full task-manifest.jsonl
0916a6b5a32e15c4f5b3bf920e1ecd4f304aeb97ae186e0e3e846391e2304203  full image-manifest.jsonl
b873b5bffc3ebf2f64e353afbfdd058608165069cab6d0387412f56e20be921b  full materialization complete.json
86d943966e9e0e43ce50338483f85155ebbe493e7108725ec2ad3d1fcda75a94  infographicvqa-decar-full-materialization-result-v1.md
827fa8b15510cbdf2f1b9925978f866b582e2cc3ac9d333b0fc4fe6c21e89b8a  infographicvqa-decar-pilot-result-v2.md
22a7e9046dcd7e949aee2d725a068f7b9cd0b5a3476130e8e9c50818bf158d46  infographicvqa-decar-feature-implementation-correction-v1.md
```

The population is exactly 23,946 official-train questions, 2,204
source-connected components, 4,406 images, and five actions per question.
Official validation and test remain absent and sealed. The output root is
`artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1` and must be empty
for a first submission; an existing root requires explicit resume.

## Source-indivisible balanced sharding

Rollout generation, target-answer NLL, and label-free feature extraction use
the same four source-aligned shards. The frozen namespace is
`infovqa-decar-full-shard-v1-06817`. It was selected before generation by
enumerating the 10,000 strings
`infovqa-decar-full-shard-v1-00000` through
`infovqa-decar-full-shard-v1-09999` and minimizing, in order: question-count
range, maximum question count, squared deviation from the four-way mean, and
candidate index. This computation used only source IDs and question counts;
it did not read questions, answers, predictions, likelihoods, or outcomes.

The resulting frozen populations are:

```text
shard                 0     1     2     3
questions          6014  6036  5910  5986
source components   538   597   547   522
```

The prior empty-namespace question counts were
`5281, 5781, 5445, 7439`. The namespace reduces the maximum load from 7,439
to 6,036 and the range from 2,158 to 126 without splitting a source. The
worker recomputes and requires the frozen question and source counts before
loading the model. Shard provenance, merges, resume checks, and the final
execution record all bind the shard key and namespace.

## Frozen implementation

```text
7461ab5585b242bb1e0a78c2a0ebf793cc2e29ed41ac43f23836b05160fe2d82  src/beyond_entropy/sharding.py
ec1ac3710214812afaf717bf36a0382e0108f61ef02c9462c0df7a8997ef5d4f  src/beyond_entropy/cli.py
9f35495f5ac10b528e98bbaf4bedafc7e4861514a67c6c48c5f959642e8bb551  src/beyond_entropy/rollout_shards.py
3f7207f66685b90bcdff24a20f0bf95eab2b77b1a70f4d4db7416bcaf096e536  src/beyond_entropy/answer_likelihood.py
1fa0f6430072fd385fd26b4d24e1ed2be42d85a516a6e28500c5d14922061788  src/beyond_entropy/proxy_outcome_audit.py
b7f580af93d1bb43a8a4c30bc7f9a6458b4a1847849f294da525164ce3fc0ef0  scripts/merge_qwen_rollout_shards.py
22ecf5b02f080824f3f96892e28b7c42bf8f0a8128190e5aebae45f279b8cf3f  scripts/score_visual_action_answer_nll.py
4bb26a8977de2f1838b9cd2838cedbe6d11d6b3c9157df3c70deb17dc94acc86  scripts/slurm_infographicvqa_decar_full_h800.sh
2e0ace6888f24e2564e188f76dd41e0938e205bb2f174995d2d4bb8f4afbce  scripts/submit_infographicvqa_decar_full_h800.sh
7faf2579fa3d9a64baddad86a972c8981ec27446de85695b445b2bee6c805115  tests/test_sharding.py
bbef104b150d59eba9c641419e1c13e52829bedea6c0cbf3843cea961bf1fc56  tests/test_rollout_shards.py
247eff05775c73f7017be9bb302afc37528dd23687c5a93e999fe8c4cac937b5  tests/test_answer_likelihood.py
2b68611329b450ea347fa21aeacff730b34cf97682a4860d1b88216c6e4af48a  tests/test_proxy_outcome_audit.py
20b1a574df633d59a5190c5e7dfa8deb45250899396751f76a20af0294f797ed  tests/test_infographicvqa_decar_full_contract.py
```

The submitter passes the committed revision and exact worker/freeze hashes to
the job. The worker fails closed if the revision, tracked worktree, bound
documents, manifests, model snapshot, hardware, runtime configuration,
resume hashes, source partition, output coverage, label-free contract, or
strict DECAR join changes.

## Runtime and acceptance

The corrected pilot completed in 528 seconds; its slowest shard contained 142
questions. Scaling by the frozen maximum of 6,036 projects 22,444 seconds
(6.23 hours) and 1,496 GPU-minutes. A 20% reserve is about 7.48 hours and
1,796 GPU-minutes. The job requests four H800s for 8 hours 15 minutes, and the
submitter requires at least 1,980 remaining GPU-minutes plus a successful
Slurm admission test.

The run produces exactly 119,730 rollout rows, 119,730 target-answer NLL rows,
one merged label-free semantic tensor, and a strict joined-input audit for all
23,946 decisions. First-pass and resume outputs must be byte-identical. It
does not fit DECAR, compute OOF predictions, inspect scientific endpoints, or
open validation/test. All supported Slurm state emails go to
`yihangc@connect.hku.hk`; no credential is exported and no GitHub push is
authorized.

## Verification

- Complete repository regression: 421 passed, 22 skipped.
- Focused source-sharding and full-worker regression: 23 passed.
- Focused mypy: no issues in six source files.
- Python compilation, Black formatting, shell syntax, and whitespace checks
  passed.
