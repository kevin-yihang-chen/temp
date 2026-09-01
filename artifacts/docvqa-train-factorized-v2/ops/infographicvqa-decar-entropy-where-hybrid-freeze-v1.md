# InfographicVQA entropy-when / OOF-where hybrid diagnostic freeze

Status: frozen after the registered DECAR v1 train result and before computing
any hybrid endpoint. This is an official-train-only diagnostic. Validation and
test remain sealed regardless of its result.

## Motivation and single hypothesis

DECAR v1 failed because its OOF `when` ranking did not enrich helpful states.
The already-registered entropy-gated exhaustive-UG baseline produced positive
ANLS gain and much higher helpful-call precision, but remained utility-negative
because it executed four crops per selected state.

Test one derived hypothesis: use baseline answer entropy only for `when`, use
the already-frozen source-OOF DECAR action prediction only for `where`, and
execute exactly one crop. No model is refit and no outcome is used to select a
state or crop.

## Bound inputs

```text
9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e  merged rollouts
c8338b1960ca223c892c3f992b0ccc3027d543113f67c82a734eccd7e3699c4b  OOF predictions
dc3193dfc626a3df50321f4d92a336ce784aee34f1f2e91c57cf87d1f8085537  OOF audit
ebc936739e970fbfda25ebe02ef71d6b7f46674f9d00092011b5665c2daa9bf0  OOF report
8de073870fcade5ac111d59de81e9c70dc567c9900e0b223cdebca6a8318f31f  OOF completion
17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6  formal bootstrap indices
5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0  formal bootstrap source order
ee5f9972e1d897c7fb833208a5722ee3a0313a05f0217f921966b3e0e1978df9  formal evaluation
d0443614c286349b7e360d646fef960816aba47a614bc20960c119d5e0ddeb79  formal evaluation completion
bdf2ee531c76743fccdffde0873380640f0cf8cdd16ee31ef71d4d23e386143a  formal result record
d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342  method protocol
```

The evaluator must verify all hashes before reading task outcomes. It must
also verify the formal decision is `decar_not_advanced`, validation/test were
not opened, OOF coverage is 23,946 decisions from 2,204 sources, prediction
rows are outcome-free, and every formal fit/bootstrap audit passed.

## Frozen policies and budgets

Use the five registered nominal rates: 0.5%, 1%, 2%, 5%, and 10%. At each
rate, take the exact DECAR primary actual-call budget recorded in the formal
evaluation: 120, 240, 479, 1,198, and 2,395 calls. Rank all official-train
states by baseline answer entropy and use the existing complete-tie rule to
target that budget. The resulting threshold, actual calls, and selected
identities must exactly reproduce the formal
`entropy_gate_random_and_fixed` audit.

The primary policy, `entropy_when_decar_where`, calls those entropy-selected
states, executes exactly one crop, and uses the OOF DECAR
`selected_action_id`, ignoring the DECAR `when` eligibility bit. Evaluate these
fixed comparators:

- `answer_now`;
- original registered `decar` at the same actual-call budget;
- one-crop `entropy_random` and `entropy_fixed_ug_grid_00` on the identical
  entropy-selected states;
- registered four-crop `entropy_gated_ug` at `floor(actual_calls / 4)` states;
- `entropy_when_task_value_where`, using the same states and the OOF
  task-value-only action.

The DECAR, loss-only, and no-harm-head actions are known before this diagnostic
to agree on all 23,946 OOF rows, so duplicate hybrid copies are prohibited.
The task-value-only action differs on 17,446 rows and is the single meaningful
where ablation. No threshold, rate, action, cost, or comparator may be added
after endpoint computation.

## Metrics, inference, and support rule

Reuse the exact formal `int32 [20000, 2204]` paired whole-source bootstrap
matrix, source order, seed, source-balanced metrics, and cost
`lambda = 0.05`. Report question- and source-balanced ANLS gain, utility,
executed crops, call rate, helpful-call precision, induced harm, negative
utility calls, action regret, missed positive utility, all 95% percentile
intervals, and paired source-utility differences.

An operating point supports the hybrid only if all conditions hold:

1. at least 100 calls and 50 called sources;
2. the primary source-utility 95% lower endpoint is strictly positive;
3. primary source utility is strictly above every listed feasible non-oracle
   comparator, including `answer_now` and original DECAR;
4. primary source utility is strictly above
   `entropy_when_task_value_where`;
5. primary induced harm and negative-utility-call mass are each no greater
   than both one-crop entropy-random and entropy-fixed baselines;
6. every input, identity, complete-tie, budget, join, OOF, cost, and bootstrap
   audit passes.

If multiple points qualify, select higher source utility, then lower induced
harm, then lower nominal rate. Emit `hybrid_train_supported` if any point
qualifies and `hybrid_train_not_supported` otherwise. This decision controls
only whether to invest in a separately frozen next method; it must never open
validation or test automatically.

Write atomically under
`artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1/entropy-where-hybrid-v1`.
All submitted-task state changes must email `yihangc@connect.hku.hk`. No GitHub
push is authorized.
