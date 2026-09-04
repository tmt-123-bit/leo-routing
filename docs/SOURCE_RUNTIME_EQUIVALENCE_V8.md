# Source/runtime semantic-equivalence audit (v8)

## Purpose and limits

This audit establishes a narrow runtime bridge between the exact source used
by the frozen congestion-context checkpoints and the current diagnostic
runtime used by adaptive-shield preflight v8. It replaces an invalid
whole-file-equality requirement with explicit structural and behavioral
contracts.

The audit is not a new evaluation, model-selection event, or confirmatory
experiment. It opens no fresh workload panel, trains no policy, permits no
promotion decision, and cannot support a paper claim by itself. The generated
artifact records `paper_claim_allowed=false`,
`promotion_decision_allowed=false`, and
`new_evaluation_panel_accessed=false`.

## Frozen inputs

The source archive is:

```text
experiments/archive/source-snapshots/actor-score-hysteresis-c3d05253666b.zip
```

Its archive SHA-256 and the SHA-256 plus byte length of every one of its 12
entries are constants in `src/run_source_runtime_equivalence_audit.py`. The
audit also replays the self-hashes and file hashes of:

- `experiments/archive/congestion-context-screen-20k-v1/screen_spec.json`;
- `experiments/archive/congestion-hysteresis-screen-v1/hysteresis_spec.json`.

The source-screen and hysteresis-screen fingerprints must resolve to the
corresponding archive entries. The two real proposed checkpoints used by the
dynamic replay are independently bound by SHA-256 and byte length.

## Structural contract

The audit parses the archived and current environment and wrapper with
Python's AST parser. For `SynchronousLeoMultiAgentEnv`, all 27 unchanged
methods must remain AST-identical. Exactly six existing methods may differ:
`__init__`, `reset`, `step`, `_forward_reward`,
`_global_reward_components`, and `validate_invariants`. Exactly three methods
may be added: `_route_switch_context`, `_switch_cost_applies`, and
`class_delivery_ratios`.

For `CleanMARLLeoMultiAgentWrapper`, all 15 archived methods must remain
AST-identical. Only the four candidate-feature schema/index getters may be
added. Any missing, additional, or differently classified method fails the
audit.

The current proposed feature contract is independently checked as dimension
26, action size 7, schema
`leo_multi_candidate_features_v1_dim_26`, and route-switch, urgency, and
class-2 indices `17`, `20`, and `23`.

## Behavioral contract

Archived and current modules execute in separate subprocesses so Python's
module cache cannot mix runtimes. Each subprocess runs both frozen scenarios
with the already exposed v7 workload seed `60001`; no fresh validation or test
seed is accessed.

Two policies are replayed:

1. Deterministic first-feasible action selection, which isolates environment
   transitions from checkpoint inference.
2. The real frozen proposed checkpoint for policy seed `1710210210`, loaded
   through the archived/current raw actor-score hysteresis policy with
   `stay_bonus=0.40`.

For every step, the audit hashes raw observation, mask, and action bytes,
followed by reward, termination flags, and canonical core environment
metrics. Both executions must end after the expected 30 steps, match one
another exactly, and match frozen trajectory digests plus final observation
and action-mask hashes.

This establishes equivalence only for the bound proposed runtime, scenarios,
checkpoint pair, policies, seed, and trajectory length. It does not prove
general program equivalence or validate untouched workload panels.

## Artifact and fail-closed replay

The output is:

```text
experiments/source-runtime-equivalence-v8/source_runtime_equivalence.json
```

It is written immutably and contains a canonical self-hash. Replay rebuilds
the complete artifact, including archive/spec/checkpoint bindings, AST
relations, dynamic signatures, current runtime hashes, audit-script hash, and
this protocol hash. A byte-level semantic record difference fails closed.

Adaptive-shield preflight v8 must load and fully replay this artifact before
it creates an output directory, writes a plan or invocation, or evaluates a
fresh workload. Its own spec binds the equivalence artifact file hash and
self-hash.

## Commands

Build without writing:

```powershell
F:\leo-venv\Scripts\python.exe src\run_source_runtime_equivalence_audit.py --dry-run
```

Create the immutable artifact:

```powershell
F:\leo-venv\Scripts\python.exe src\run_source_runtime_equivalence_audit.py
```

Replay the artifact:

```powershell
F:\leo-venv\Scripts\python.exe src\run_source_runtime_equivalence_audit.py --replay experiments\source-runtime-equivalence-v8\source_runtime_equivalence.json --dry-run
```
