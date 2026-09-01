# InfographicVQA entropy-when / oracle-where factorization freeze

Status: frozen after the terminal job-203059 hybrid result and before computing
any oracle-where bootstrap endpoint. This is an outcome-dependent,
official-train-only diagnostic. It is not a deployable policy, cannot count as
method evidence, and cannot open validation or test.

## Question and interpretation boundary

The frozen entropy-when / OOF-where hybrid was utility-negative at every
registered budget. Its measured action-selection regret is larger than the
utility deficit at every point, so one unresolved factorization question
remains:

> Holding the exact entropy-selected state identities and one-crop cost fixed,
> is utility significantly positive when crop choice is replaced by the
> outcome oracle, and is its paired advantage over both frozen OOF crop
> selectors significantly positive?

A positive answer supports only the diagnosis that crop/action selection is a
material bottleneck and authorizes investment in a separately frozen,
outcome-free OOF crop ranker. A negative answer closes this entropy-when line.
Neither outcome is validation or test evidence.

## Bound inputs

```text
9b2313ed122df26f75e8d27326bb695d469f0b1afad0921afb3676c040d3287e  merged rollouts
c8338b1960ca223c892c3f992b0ccc3027d543113f67c82a734eccd7e3699c4b  nested OOF predictions
dc3193dfc626a3df50321f4d92a336ce784aee34f1f2e91c57cf87d1f8085537  nested OOF audit
8de073870fcade5ac111d59de81e9c70dc567c9900e0b223cdebca6a8318f31f  nested OOF completion
ee5f9972e1d897c7fb833208a5722ee3a0313a05f0217f921966b3e0e1978df9  formal evaluation
d0443614c286349b7e360d646fef960816aba47a614bc20960c119d5e0ddeb79  formal evaluation completion
17d3d5e2354710a019b662f4261a12fec65e6b096e6095aa6db3a2955effdba6  formal bootstrap indices
5acaca75d9feb37fe1ad16155e29c7e2fd92972b34898785eed1444d02ab40a0  formal bootstrap source order
ab4a179c2141de60c9a1d173e34a7166d075935e5283569439e3d93424344a62  frozen hybrid evaluation
0597526725eac7efed05392fb652b04798de26b71d6de6d063303b49ec114d42  frozen hybrid decision
4e3b3c2b2b2e1698fcf9bb3c9e71881b11e1357982bb1b6634a12af6a7aa03ac  frozen hybrid completion
16ed848ee49702d1f1c41e9f59b2245585dc03a918e3ffacaf3520fc2fafefab  job-203059 result record
86e61bb0c7a4ad5a259077314be3a83c6c95284d2b219c414016b2280292a8bb  hybrid freeze
d8651f9c235be4da8883df10a692a5171b9ca902b42bc1864b69008655688342  DECAR method protocol
```

The evaluator must verify every hash before reading task outcomes. It must also
verify the formal and hybrid decisions are respectively `decar_not_advanced`
and `hybrid_train_not_supported`, both selected operating points are null,
validation/test were never opened, all 23,946 prediction rows are outcome-free,
all 55 source-overlap checks are zero, and the hybrid contract/action-family
audits passed.

## Frozen identities and policies

Reuse the exact five rates, actual-call budgets, complete-tie entropy
thresholds, and selected decision identities from the frozen hybrid:

```text
nominal rates: 0.5%, 1%, 2%, 5%, 10%
actual calls:  120,  240, 479, 1,198, 2,395
cost:          lambda = 0.05 per executed crop
```

For each selected state, `entropy_when_task_oracle_where` executes exactly one
of the same four registered crop actions. It selects the action with maximum
observed ANLS delta; ties use ascending action ID. This is the only oracle
operation. It must not stop a selected state, change the gate, change the cost,
or inspect validation/test.

Evaluate these fixed same-state comparators:

- `entropy_when_decar_where`, using the frozen OOF DECAR action;
- `entropy_when_task_value_where`, using the frozen OOF task-value action;
- `entropy_random` and `entropy_fixed_ug_grid_00`;
- `answer_now`.

The oracle point estimate and per-state utility difference versus
`entropy_when_decar_where` must exactly equal the primary hybrid utility plus
its already-recorded action-selection regret. No new rate, threshold, action,
cost, policy, or subgroup may be introduced after endpoint computation.

## Metrics and inference

Reuse the exact formal `int32 [20000, 2204]` paired IID whole-source bootstrap,
source order, seed, and 95% percentile intervals. Report the full existing
question- and source-balanced metric family for every policy, including ANLS
gain, utility, calls, executed crops, helpful precision, induced harm,
negative-utility calls, action-selection regret, and oracle-stop regret.
Report paired source-utility differences from the oracle policy to every
comparator.

An operating point diagnoses a supported where bottleneck only if all of the
following hold:

1. it contains at least 100 calls and 50 called sources;
2. oracle-where source-utility 95% lower endpoint is strictly positive;
3. the paired oracle-minus-DECAR-where utility 95% lower endpoint is strictly
   positive;
4. the paired oracle-minus-task-value-where utility 95% lower endpoint is
   strictly positive;
5. oracle utility, ANLS gain, calls, and per-state differences pass the exact
   arithmetic consistency audits against the frozen hybrid;
6. every hash, population, join, selection-identity, OOF, action, cost,
   bootstrap, and seal audit passes.

If multiple points qualify, select higher oracle source utility, then lower
nominal rate. Emit `where_bottleneck_supported` if any point qualifies and
`where_bottleneck_not_supported` otherwise. The decision controls only the
next official-train research branch. Validation and test remain sealed in both
cases.

Write atomically under
`artifacts/infographicvqa-train-v1/decar-v1/full-qwen7b-v1/entropy-oracle-where-factorization-v1`.
All submitted-task state changes must email `yihangc@connect.hku.hk`. No GitHub
push is authorized.
