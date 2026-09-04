# Calm-selective proposed-checkpoint shield design (v9)

## Scope

This is an exposed-panel engineering design stage. It is exploratory, is not
validation or test evidence, and cannot support a paper claim or model
promotion. It does not train or alter any actor weight. It may use only the
already exposed v7 workloads `60001..60010`; every fresh panel remains closed
regardless of the design result.

The v9 design is admissible only after independently replaying the complete v7
reference freeze and the complete failed v8 selection. The v8 selection must
remain `no_eligible_candidate`, with no selected candidate and no validation or
test permission. The v9 spec binds both the internal v8 selection hash and the
selection-file hash.

## Exposed-panel mechanism

All 18 v8 candidates reduced the avoidable micro-rate and never exceeded the
proposed checkpoint's routing-switch mean. The common failure was the
`hotspot_high_load/1245825580` raw-context switch gate. For v8 candidate 12,
the 998 observed switches over ten episodes were 660 allowed cached-feasible
actor switches plus 338 forced switches. The largest controllable stratum was
327 allowed non-class-2, non-urgent choices. That candidate also missed the
`medium_load/1245825580` delivery gate by only `0.000125`.

The v9 intervention therefore leaves urgent and class-2 packets free of the
new term and adds resistance only to calm, non-class-2 cached-route changes.
This is a falsifiable mechanism hypothesis, not evidence that the mechanism
will generalize.

## Frozen formula

For clipped waiting age `a` and class-2 indicator `z`, every candidate uses

```text
beta_eff = (1 - u * a) * (1 - 0.75 * z)
         + k * (1 - z) * max(0, 1 - 2 * a)
```

The base stay bonus is fixed at `1.0` and class-2 relief is fixed at `0.75`.
The additive term is exactly zero for class-2 packets and for `a >= 0.5`.
Only cached-feasible states in which the raw actor requests a route change can
be suppressed. First-use and forced-switch behavior remains exactly the raw
actor behavior.

## Frozen candidate grid

Candidate indices are fixed in this order:

| Index | urgency relief `u` | calm bonus `k` | Execution source |
|---:|---:|---:|---|
| 0 | 0.500 | 0.000 | v8 candidate 12 reuse control |
| 1 | 0.625 | 0.000 | new evaluation control |
| 2 | 0.750 | 0.000 | v8 candidate 14 reuse control |
| 3 | 0.625 | 0.250 | new evaluation |
| 4 | 0.625 | 0.500 | new evaluation |
| 5 | 0.625 | 0.750 | new evaluation |
| 6 | 0.750 | 0.250 | new evaluation |
| 7 | 0.750 | 0.500 | new evaluation |
| 8 | 0.750 | 0.750 | new evaluation |

There are 72 logical scenario by policy-seed jobs. The two pre-existing
controls account for 16 logical jobs and 160 reused episode rows. Their source
shard CSV and metadata hashes are checked, each shard is validated, and every
row must equal its v8 combined-artifact row before the policy label is mapped
to v9. They are never evaluated again. The remaining 56 jobs produce 560 new
episode rows if the non-dry-run command is explicitly invoked.

## Frozen diagnostics

Every cached-feasible raw-switch choice is counted by outcome (`allowed` or
`suppressed`) in age bins with edges

```text
[0, 0.125, 0.25, 0.375, 0.5, 1]
```

It is also binned by the relative anchor margin

```text
raw_switch_advantage - (1 - 0.5 * a) * (1 - 0.75 * z)
```

using bins `(-inf, 0)`, `[0, 0.125)`, `[0.125, 0.25)`,
`[0.25, 0.375)`, `[0.375, 0.5)`, `[0.5, 0.75)`, and
`[0.75, +inf)`. Both partitions must reconcile exactly to the raw-switch
choice total. Calm non-class-2 outcome counts and addon min, max, mean, and sum
are recorded separately.

These new bins are recorded for the 56 newly evaluated jobs, including the
`u=0.625, k=0` instrumentation control. The two hash-reused v8 controls retain
their original v8 diagnostics and are not retrofitted with bins that were not
recorded at evaluation time.

## Frozen gates and design buffers

Every one of a candidate's eight cells must pass the unchanged v7 gates:

- delivery difference versus proposed at least `-0.003`;
- class-2 delivery difference versus proposed at least `-0.010`;
- exact avoidable switch micro-rate strictly below proposed;
- routing-switch mean no greater than proposed;
- routing-switch mean no greater than raw context.

Eligibility additionally requires, in every cell:

- raw-context routing-switch headroom at least `2.0` switches per episode;
- avoidable micro-rate improvement at least `0.005`;
- delivery-gate slack at least `1/1920`, one medium-load packet of design
  margin.

The avoidable rate is pooled as total avoidable switches divided by total
switch opportunities within one scenario by policy-seed cell. It is not an
episode macro-mean. No additional class-2 buffer is introduced.

## Frozen global selection

No scenario-specific or seed-specific parameters are permitted. Among
candidates eligible in all eight cells, select the unique lexicographic
maximum of:

1. worst delivery slack;
2. worst class-2 slack;
3. worst raw-context switch slack;
4. worst avoidable-rate slack;
5. negative calm bonus;
6. urgency relief;
7. negative frozen candidate index.

If no candidate is eligible, the result is `no_eligible_candidate`. If a
candidate is selected, it remains only an exposed-panel engineering choice.
In either case, `fresh_validation_allowed=false`,
`fresh_test_allowed=false`, and `test_allowed=false`.

## Commands

Read-only audit and plan, with no output writes and no evaluation:

```powershell
F:\leo-venv\Scripts\python.exe src\run_calm_selective_shield_design.py --dry-run
```

The non-dry-run command is intentionally not part of this implementation
handoff. It must be started only by a separate explicit decision after the
dry-run artifacts and tests have been reviewed.
