# Adaptive proposed-checkpoint shield preflight (v8)

## Status and scope

This is a diagnostic exploratory preflight. It is not confirmatory, does not
authorize model promotion, and cannot by itself support an IEEE
Transactions-level paper claim. Every decision and manifest records
`paper_claim_allowed=false` and `promotion_decision_allowed=false`.

No policy is trained. All actor weights come from the frozen `proposed` and
`with_congestion_context` checkpoints in
`experiments/archive/congestion-context-screen-20k-v1`.

## Frozen design handoff

The preflight accepts parameters only from
`experiments/adaptive-shield-design-v8/design_selection.json`. The CLI has no
stay-bonus or relief override. The design loader must replay all of the
following before returning a selection:

- the immutable selection self-hash;
- the exact 18-candidate design grid;
- all candidate results against the bound `design_cell_gates.csv` hash;
- each candidate's eight cells and five gates, buffer gate, eligibility,
  worst-case slacks, and frozen lexicographic selection key;
- the eligible set and unique lexicographic argmax;
- equality of that unique result and `selected_parameters`;
- the design spec, source training freeze, and v7 validation freeze hashes.

The selected object has exactly `stay_bonus`, `urgency_relief`, and
`class_2_relief`. A malformed, ambiguous, non-unique, hash-mismatched, or
non-passing selection aborts before any evaluation.

## Frozen method

The three arms are evaluated for both scenarios and all four policy seeds:

1. `proposed`: proposed checkpoint with the raw greedy actor action.
2. `raw_context`: congestion-context checkpoint with the raw greedy actor
   action.
3. `shield`: the same proposed checkpoint as arm 1, loaded through
   `AdaptiveActorScoreHysteresisPolicy` with the frozen selected parameters.

The adaptive shield retains a cached feasible route unless its actor-score
margin clears the state-dependent bonus. Urgency and class-2 relief can lower
that bonus. First-use states and forced reroutes remain raw.

The proposed/shield contract is the canonical 26-feature schema:

- schema id: `leo_multi_candidate_features_v1_dim_26`;
- schema SHA-256:
  `be660bb34d6d8579773b643f2824e6dac5069fe67cfd8a8d71a36b1b147f9f70`;
- route-switch index: `17`;
- urgency index: `20`;
- class-2 index: `23`.

Every spec, shard, decision, and manifest binds the design selection hash,
source training-freeze hash, and runtime code-fingerprint hash. Every shard
also binds its checkpoint SHA, full workload list, policy schema, row count,
CSV hash, and validation freeze hash when applicable. A partial or mismatched
existing shard is an error and is never silently overwritten.

Before a spec can be built, the runner fully replays
`experiments/source-runtime-equivalence-v8/source_runtime_equivalence.json`
under the contract in `docs/SOURCE_RUNTIME_EQUIVALENCE_V8.md`. That replay
binds the exact archived training runtime, both frozen specs and checkpoints,
the declared AST method relations, the canonical feature schema and indices,
and exact archived/current trajectory signatures on already exposed workload
seed `60001`. The spec and all downstream artifacts bind the equivalence
self-hash. A missing, stale, hash-mismatched, structurally drifting, or
behaviorally drifting record aborts before source/design replay, output
directory creation, plan, invocation, or fresh-workload access. Whole-file
equality with the archived environment is neither assumed nor accepted as a
substitute for this replay.

## Workload isolation

- Validation: `71001..71010`.
- Test: `72001..72025`.

A field-aware audit found no prior workload, training, validation, test, or
reserved use of either panel. The v7 test panel `70001..70025` remains reserved
even though v7 validation failed before test opened it. The v7 validation panel
`60001..60010` and the v2 smoke training workloads `61001..61004` are exposed
and must not be reused.

Writing a protocol or immutable spec permanently reserves its workload panels.
Changing the method, parameters, gates, seeds, or panels requires a new
protocol version and new untouched panels.

## Frozen cell gates

All eight scenario by policy-seed cells must pass all five v7 gates:

- shield delivery minus proposed is at least `-0.003`;
- shield class-2 delivery minus proposed is at least `-0.010`;
- shield avoidable-switch micro-rate is strictly below proposed;
- shield mean routing switches are no greater than proposed;
- shield mean routing switches are no greater than raw context.

The avoidable-switch micro-rate is exactly

```text
sum(avoidable_routing_switches) / sum(switch_opportunities)
```

inside each scenario by policy-seed by arm cell. It is not the mean of episode
rates. A zero opportunity denominator is an error.

## Phase isolation

Validation and test are separate invocations. Test requires a passing
validation freeze and replays its self-hash, spec, source, selection, code,
parameters, workload list, individual shards, combined episode artifact,
aggregate artifact, cell-gate artifact, and all eight gate decisions.

For `--phase test`, this replay completes before `mkdir`, plan, invocation, or
shard writes. Missing, failing, stale, or non-replaying validation state leaves
the test panel unopened.

## Commands

Read-only audit and plan:

```powershell
F:\leo-venv\Scripts\python.exe src\run_adaptive_shield_preflight.py --dry-run --device cuda
```

Validation only:

```powershell
F:\leo-venv\Scripts\python.exe src\run_adaptive_shield_preflight.py --phase validation --device cuda
```

Test only after validation passes and replays:

```powershell
F:\leo-venv\Scripts\python.exe src\run_adaptive_shield_preflight.py --phase test --device cuda
```

Passing v8 would show that this fixed adaptive shield survived one diagnostic
four-policy-seed holdout. It would not establish confirmatory evidence,
cross-topology generality, robustness to distribution drift, or journal-ready
final data.
