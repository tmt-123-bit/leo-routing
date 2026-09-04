# Proposed-checkpoint route shield preflight (v7)

## Status and scope

This protocol is a diagnostic exploratory preflight. It is not confirmatory,
does not authorize promotion, and cannot by itself support an IEEE
Transactions-level paper claim. The runner must record
`paper_claim_allowed=false` and `promotion_decision_allowed=false` in every
decision and manifest.

No policy is trained in v7. The only candidate is a deterministic post-hoc
controller applied to the frozen `proposed` checkpoints from
`CONGESTION-CONTEXT-SCREEN-v1`.

## Frozen method

- Policy seeds: `1710210210`, `2078783072`, `1047581915`, `1245825580`.
- Scenarios: `medium_load`, `hotspot_high_load`.
- Candidate scorer and actor weights: frozen `proposed` checkpoint for the
  matching scenario and policy seed.
- Controller: existing `ActorScoreHysteresisPolicy`.
- Fixed `stay_bonus`: `0.40`.
- Rule: retain the cached feasible route unless the raw switch-logit advantage
  is at least `0.40`.
- First-use states and states in which the cached route is unavailable are not
  changed by the controller.
- `stay_bonus=0.40` is a mechanism-motivated heuristic fixed after an
  unarchived manual probe on the already exposed `51001..51025` panel. There is
  no immutable beta-0.40 selection artifact. This provenance limitation is
  recorded explicitly; the value is frozen before v7 validation and must not
  be changed after validation or test outcomes are observed.

The three evaluation arms are:

1. `proposed`: frozen proposed checkpoint, `stay_bonus=0`.
2. `raw_context`: frozen `with_congestion_context` checkpoint,
   `stay_bonus=0`, used only as the routing-switch reference.
3. `shield`: the same frozen proposed checkpoint as arm 1,
   `stay_bonus=0.40`.

## Workload isolation

- Validation panel: `60001..60010`.
- Test panel: `70001..70025`.

A structured scan of local routing source files and experiment artifacts found
no prior seed/workload use of either panel. The initially considered test panel
`61001..61025` is rejected: a v2 smoke run used `train_seed_start=61001` and
`train_seed_count=4`, so `61001..61004` are exposed training workloads.

Validation and test are separate runner phases. Validation never evaluates or
writes a test shard. Test execution requires a passing validation freeze whose
self-hash, source hash, combined artifacts, individual shard metadata, shard
hashes, and all eight recomputed gate cells replay exactly. A missing, failing,
or non-replaying freeze blocks test execution before any test result is opened.

## Frozen cell gates

Every one of the 2 scenario by 4 policy-seed cells must pass all five gates:

- Mean delivery difference, shield minus proposed: at least `-0.003`.
- Mean class-2 delivery difference, shield minus proposed: at least `-0.010`.
- Avoidable-switch micro-rate: shield strictly less than proposed.
- Mean routing switches: shield less than or equal to proposed.
- Mean routing switches: shield less than or equal to raw context.

The avoidable-switch estimand is exactly:

```text
sum(avoidable_routing_switches) / sum(switch_opportunities)
```

It is computed separately inside each scenario by policy-seed by arm cell. A
mean of per-episode rates is not valid for a gate. A zero total opportunity
denominator is an error, not a zero rate.

Validation is pass/fail only. A validation failure blocks the test panel. Any
change to the controller, bonus, gates, policy seeds, or workload panel after a
validation failure is a new protocol version and requires another untouched
test panel.

## Execution

Dry-run, which must not write the output directory:

```powershell
F:\leo-venv\Scripts\python.exe src\run_proposed_shield_preflight.py --dry-run --device cuda
```

Validation only:

```powershell
F:\leo-venv\Scripts\python.exe src\run_proposed_shield_preflight.py --phase validation --device cuda
```

Test only, and only after independent validation replay succeeds:

```powershell
F:\leo-venv\Scripts\python.exe src\run_proposed_shield_preflight.py --phase test --device cuda
```

Evaluation shards are resumable only when the full expected metadata, source
checkpoint hash, workload list, CSV hash, row count, and policy schema match.
Test shard metadata additionally binds the validation freeze self-hash.

## Interpretation

Passing v7 would establish that the fixed shield survived this diagnostic
four-policy-seed holdout. It would not turn an adaptively developed method into
confirmatory evidence, establish generality across topologies or traffic
families, or make the result ready for direct use as final journal data.
