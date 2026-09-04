# Relative-anchor margin-capped shield design (v10)

## Status and scope

This protocol is an exposed-panel engineering design stage. It is exploratory,
is not validation or test evidence, and cannot support a paper claim or model
promotion. It does not train or alter actor weights. It may use only the
already exposed design workloads `60001..60010`.

Every fresh panel is permanently closed to v10, regardless of the design
outcome. In particular, v10 must not read, enumerate, evaluate, or write any
fresh validation or test workload beginning at `71001` or `72001`. Every v10
specification, shard, and decision must record:

```text
fresh_validation_allowed = false
fresh_test_allowed = false
test_allowed = false
promotion_decision_allowed = false
paper_claim_allowed = false
```

The v8 and v9 artifacts are immutable inputs. V10 must not modify, replace, or
recompute their specification, selection, runtime-fingerprint, shard, or
combined-artifact files.

## Required provenance

V10 is admissible only after replaying the complete v9 selection as
`no_eligible_candidate`. The v10 specification must bind the v9 selection
self-hash:

```text
b58aaf3ad17c07cf10dd6f2d03a6f12d5917cbb24ad604a18999d21db9520c8d
```

It must also bind the v9 design specification, runtime code fingerprint,
source training freeze, v7 validation freeze, and v8 design selection hashes
already recorded by v9. A missing or non-replaying dependency fails closed
before any v10 evaluation starts.

## Frozen mechanism

Let:

- `a = clip(deadline_normalized_waiting_age, 0, 1)`;
- `z` be the binary class-2 indicator;
- `g` be the frozen raw actor's switch-logit advantage over the single cached
  feasible incumbent.

The v8 anchor threshold is:

```text
beta_anchor = (1 - 0.5 * a) * (1 - 0.75 * z)
```

The frozen v9 candidate-5 control threshold is:

```text
beta_control = (1 - 0.625 * a) * (1 - 0.75 * z)
             + 0.75 * (1 - z) * max(0, 1 - 2 * a)
```

The only v10 change is a cap on the positive threshold increment relative to
the anchor:

```text
beta_eff(tau) = min(beta_control, beta_anchor + tau)
```

The raw switch is suppressed exactly when `g < beta_eff(tau)`. The cap can
only release a raw switch that v9 candidate 5 suppressed; it can never suppress
a choice that candidate 5 allowed. Class-2 choices, choices with `a >= 0.5`,
and any state where `beta_control <= beta_anchor` are unchanged by the cap.

Only cached-feasible states in which the raw actor requests a route change are
eligible for suppression. First-use states, forced-switch states, the actor
weights, candidate scores, masks, and tie behavior remain unchanged.

## Frozen candidate grid

Candidate indices and parameter order are fixed:

| Index | `tau` | Execution source |
|---:|---:|---|
| 0 | 0.750 | exact v9 candidate-5 row reuse control |
| 1 | 0.375 | new v10 evaluation |
| 2 | 0.250 | new v10 evaluation |

All other parameters are fixed at v9 candidate 5:

```text
stay_bonus = 1.0
urgency_relief = 0.625
class_2_relief = 0.75
calm_bonus = 0.75
```

No scenario-specific, policy-seed-specific, workload-specific, or
class-specific parameter is permitted.

For non-class-2 calm choices, the largest positive value of
`beta_control - beta_anchor` is exactly `0.75`, attained at `a = 0`.
Every other state has a smaller increment or a non-positive increment.
Therefore `tau=0.75` is exactly action-equivalent to v9 candidate 5 and is the
frozen reuse control.

The reuse maps v10 candidate 0 to v9 candidate index 5 with identifier:

```text
b1p000_u0p625_c0p750_k0p750
```

All eight reused cell rows and their 80 episode rows must be checked against
the original v9 shard metadata, shard CSV hashes, combined artifacts, and
policy schema before relabeling. The reuse provenance must be recorded as
`v9_row_reuse`. An exhaustive formula and action-equivalence test is required
before row reuse; the control is never evaluated again.

There are 24 logical scenario by policy-seed jobs. Candidate 0 accounts for
eight reused logical jobs and 80 reused episode rows. Candidates 1 and 2
account for exactly 16 new jobs and 160 new episodes:

```text
2 new candidates * 2 scenarios * 4 policy seeds = 16 new jobs
16 new jobs * 10 exposed workloads = 160 new episodes
```

## Frozen workloads and cells

The only workloads are:

```text
60001, 60002, 60003, 60004, 60005,
60006, 60007, 60008, 60009, 60010
```

The scenarios are `medium_load` and `hotspot_high_load`. The policy seeds are
`1710210210`, `2078783072`, `1047581915`, and `1245825580`. Each candidate is
judged in all eight scenario by policy-seed cells. Missing, extra, duplicate,
or reordered workload results are errors.

## Frozen five gates

Every cell must pass all five unchanged v7 gates:

- delivery difference versus proposed is at least `-0.003`;
- class-2 delivery difference versus proposed is at least `-0.010`;
- exact avoidable-switch micro-rate is strictly below proposed;
- routing-switch mean is no greater than proposed;
- routing-switch mean is no greater than raw context.

The avoidable-switch micro-rate is computed within one scenario by policy-seed
cell as:

```text
sum(avoidable_routing_switches) / sum(switch_opportunities)
```

It is not a mean of episode rates. A zero denominator is an error.

## Frozen three design buffers

Eligibility additionally requires all three buffers in every cell:

- raw-context routing-switch headroom is at least `2.0` switches per episode;
- avoidable micro-rate improvement is at least `0.005`;
- delivery-gate slack is at least `1/1920`, one medium-load packet of design
  margin.

No extra class-2 buffer is introduced. A candidate that passes the five gates
but fails any buffer is not eligible.

## Frozen global selection

Eligibility requires all five gates and all three buffers in all eight cells.
Among eligible candidates, select the unique lexicographic maximum of:

1. worst delivery buffer slack, defined as worst delivery-gate slack minus
   `1/1920`;
2. worst class-2 gate slack;
3. worst raw-context headroom buffer slack, defined as worst raw-context
   switch headroom minus `2.0`;
4. worst avoidable-rate buffer slack, defined as worst avoidable-rate
   improvement minus `0.005`;
5. `tau`, preferring the larger value and therefore the smaller behavioral
   change from the v9 control;
6. negative frozen candidate index.

If no candidate is eligible, the decision is `no_eligible_candidate`. A
selected candidate remains only an exposed-panel engineering choice. Either
decision leaves every fresh-panel permission false.

## Frozen first-order diagnostic

The v9 candidate-5 `hotspot_high_load/1245825580` cell observed 915 routing
switches over ten workloads. The raw-context headroom buffer permits at most
945 switches. Positive relative-anchor-margin choices suppressed by candidate
5 were distributed as:

| Relative anchor margin | Suppressed choices |
|---|---:|
| `[0, 0.125)` | 37 |
| `[0.125, 0.25)` | 25 |
| `[0.25, 0.375)` | 17 |
| `[0.375, 0.5)` | 8 |
| `[0.5, +inf)` | 0 |

On the fixed candidate-5 trajectory, treating every released suppressed choice
as one additional routing switch gives these first-order diagnostic bounds:

| `tau` | Choices released by recorded bins | First-order switch total |
|---:|---:|---:|
| 0.750 | 0 | 915 |
| 0.375 | 8 | 923 |
| 0.250 | 25 | 940 |

These calculations are diagnostics, not gate results and not closed-loop
guarantees. A changed action can alter later observations, raw choices, route
cache state, forced switches, delivery, and the number and margins of later
switch opportunities. Only complete deterministic evaluation of all 16 new
jobs may determine v10 eligibility. The first-order table must never be used
to fill, infer, or waive a missing gate cell.

## Interpretation

V10 tests one narrow hypothesis: preserving the v9 candidate-5 suppression of
low-margin calm switches while returning high-margin choices to the frozen raw
actor may retain hotspot switch headroom without the two medium-load delivery
failures. Passing the exposed design screen would support only this engineering
hypothesis. It would not establish generalization, authorize fresh evaluation,
or make the result suitable as final IEEE Transactions evidence.
