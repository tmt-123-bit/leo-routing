# Age-band shield design (v11)

## Status and scope

V11 is the final exposed-panel engineering design stage for the frozen
proposed checkpoint. It is exploratory, is not validation or test evidence,
and cannot support a paper claim or model promotion. It performs no training
and cannot alter actor weights, checkpoint contents, feature construction,
candidate scores, masks, or the environment.

V11 may use only the already exposed workloads `60001..60010`. It must not
read, enumerate, evaluate, infer, or write any fresh validation or test
workload. In particular, workloads beginning at `71001` or `72001` remain
closed. Every v11 specification, plan, shard metadata record, combined
artifact, and selection artifact must record:

```text
fresh_validation_allowed = false
fresh_test_allowed = false
test_allowed = false
promotion_decision_allowed = false
paper_claim_allowed = false
```

These values remain false whether v11 succeeds or fails.

## Required immutable provenance

V11 is admissible only after replaying the complete v10 selection as
`no_eligible_candidate`, with no selected candidate. The v10 selection is a
new immutable upstream input to v11. Both of these hashes must be bound and
verified before any v11 evaluation:

```text
v10 internal design-selection self-hash:
1f51e4a0bdfab0381e224403193a38783b1e54b6b550a3f6450d8754a05765f2

v10 design_selection.json file SHA-256:
0302875cc4cc34dd3c6001acfb3542fc8fe75df448ca18846f97a46538ad1c43
```

V11 must also bind and replay the v9 selection and candidate-5 source rows,
plus the runtime, source-training, source-runtime-equivalence, v7 validation,
and earlier design freezes transitively bound by v10. A missing hash, hash
mismatch, non-replaying dependency, changed permission flag, or changed v10
decision fails closed before evaluation.

V7 through v10 artifacts are immutable. V11 must not modify, replace,
recompute, or silently repair any prior specification, shard, combined
artifact, runtime fingerprint, or selection file.

## Frozen mechanism

Let:

- `a = clip(deadline_normalized_waiting_age, 0, 1)`, obtained from the
  canonical candidate-feature schema;
- `z` be the binary class-2 indicator from that schema;
- `g` be the frozen raw actor's switch-logit advantage over the single cached
  feasible incumbent;
- `lambda` be the frozen age-band mix, implemented as `age_band_mix`.

Define the unchanged v9 candidate-5 base and calm weight:

```text
base(a, z) = (1 - 0.625 * a) * (1 - 0.75 * z)
w_old(a)   = max(0, 1 - 2 * a)
```

Define the new continuous age-band weight:

```text
w_band(a) = clip(8 * (a - 0.125), 0, 1)
          * clip(2 - 4 * a, 0, 1)
```

Equivalently, for the already clipped `a`:

```text
w_band(a) = 0                 for 0 <= a <= 0.125
            8 * (a - 0.125)  for 0.125 < a < 0.25
            2 - 4 * a        for 0.25 <= a < 0.5
            0                 for 0.5 <= a <= 1
```

The effective threshold is frozen as:

```text
beta_eff(lambda) = base(a, z)
                 + 0.75 * (1 - z)
                   * ((1 - lambda) * w_old(a)
                      + lambda * w_band(a))
```

Only cached-feasible states in which the raw actor requests a route change
are eligible for suppression. The raw switch is suppressed exactly when
`g < beta_eff(lambda)`. Equality follows the existing tie behavior and is not
suppressed. First-use states, forced-switch states, infeasible cached routes,
raw action selection, and all non-routing behavior remain unchanged.

For class-2 packets, the age-dependent additive term is exactly zero. For
`a >= 0.5`, both calm weights are zero. At `lambda=1`, the new additive term
is also zero for `a <= 0.125`, rises continuously to its maximum at `a=0.25`,
and returns continuously to zero at `a=0.5`.

## Frozen candidates and order

The candidate set contains exactly two candidates in this order:

| Index | `lambda` | `age_band_mix` | Execution source |
|---:|---:|---:|---|
| 0 | 0 | 0.0 | exact v9 candidate-5 row reuse control |
| 1 | 1 | 1.0 | the only new v11 evaluation |

All other parameters remain fixed at v9 candidate 5:

```text
stay_bonus = 1.0
urgency_relief = 0.625
class_2_relief = 0.75
calm_bonus = 0.75
```

Candidate identifiers are frozen as:

```text
candidate 0: b1p000_u0p625_c0p750_k0p750_m0p000
candidate 1: b1p000_u0p625_c0p750_k0p750_m1p000
```

When `lambda=0`, `beta_eff` reduces pointwise to the v9 candidate-5 formula:

```text
base(a, z) + 0.75 * (1 - z) * max(0, 1 - 2 * a)
```

Candidate 0 is therefore action-equivalent to v9 candidate index 5,
`b1p000_u0p625_c0p750_k0p750`, for every valid state. Its eight cell rows and
80 episode rows must be verified against the original v9 shard metadata,
source shard CSV hashes, combined artifacts, policy schema, and frozen
formula before relabeling. Reuse provenance must be recorded as
`v9_candidate_5_row_reuse`. Candidate 0 must never be evaluated again.

Candidate 1 is the sole new candidate. It must be evaluated in every frozen
scenario by policy-seed cell. No intermediate `lambda`, alternate age-band
edge, different slope, scenario-specific parameter, seed-specific parameter,
workload-specific parameter, or class-specific parameter is permitted.

## Frozen workload and job counts

The only permitted workload order is:

```text
60001, 60002, 60003, 60004, 60005,
60006, 60007, 60008, 60009, 60010
```

The scenarios are `medium_load` and `hotspot_high_load`. The policy seeds are
`1710210210`, `2078783072`, `1047581915`, and `1245825580`. Each candidate is
judged in all eight scenario by policy-seed cells. Missing, extra, duplicate,
or reordered workloads are errors.

The frozen accounting is:

```text
2 logical candidates * 2 scenarios * 4 policy seeds = 16 logical jobs
candidate 0 reuse: 8 logical jobs, 80 reused episode rows
candidate 1 new:   8 evaluation jobs, 80 new episodes
training jobs:     0
```

Exactly eight new jobs and 80 new episodes are allowed. A retry may only
reproduce an identical missing or failed shard under the frozen specification;
it cannot create another logical job or another candidate.

## Frozen five v7 gates

Every candidate must pass all five unchanged v7 gates in every one of its
eight cells:

- delivery difference versus proposed is at least `-0.003`;
- class-2 delivery difference versus proposed is at least `-0.010`;
- exact avoidable-switch micro-rate is strictly below proposed;
- routing-switch mean is no greater than proposed;
- routing-switch mean is no greater than raw context.

The avoidable-switch micro-rate is pooled within one scenario by policy-seed
cell:

```text
sum(avoidable_routing_switches) / sum(switch_opportunities)
```

It is not an episode macro-mean. A zero denominator is an error.

## Frozen three v9 design buffers

Eligibility additionally requires all three unchanged v9 buffers in every
cell:

- raw-context routing-switch headroom is at least `2.0` switches per episode;
- avoidable micro-rate improvement is at least `0.005`;
- delivery-gate slack is at least `1/1920`, one medium-load packet of design
  margin.

In formulas:

```text
raw routing-switch mean - shield routing-switch mean >= 2.0
proposed avoidable micro-rate - shield avoidable micro-rate >= 0.005
delivery difference versus proposed + 0.003 >= 1/1920
```

No extra class-2 buffer or tolerance is introduced. The five gates and three
buffers cannot be relaxed, rounded, reinterpreted, or replaced after any v11
result is observed.

## Frozen global selection

Eligibility is global: a candidate is eligible only if all five gates and all
three buffers pass in all eight cells. Per-scenario, per-seed, or per-workload
selection is forbidden.

Among globally eligible candidates, select the unique lexicographic maximum
of:

1. worst delivery-buffer slack across all eight cells;
2. worst class-2 gate slack across all eight cells;
3. worst raw-context headroom-buffer slack across all eight cells;
4. worst avoidable-rate-buffer slack across all eight cells;
5. negative `lambda`, preferring the smaller change if all evidence terms tie;
6. negative frozen candidate index.

The buffer slacks subtract their frozen minimum margins: `1/1920`, `2.0`, and
`0.005`, respectively. If no candidate is eligible, the decision is
`no_eligible_candidate`. A selected candidate remains only an exposed-panel
engineering choice.

## Frozen hypothesis and diagnostics

V11 tests one narrow hypothesis: the v9 candidate-5 calm penalty may protect
switch behavior more selectively if it is removed from the youngest packets
and concentrated in the continuous `0.125 < a < 0.5` age band. The mechanism
is fixed before the single new candidate is evaluated.

Any age-stratified counts, threshold margins, released choices, delivery
changes, or switch changes recorded by v11 are descriptive diagnostics only.
They cannot be used to alter the band, add a candidate, waive a failed cell,
or infer a missing result. Only complete deterministic evaluation of all
eight new jobs and global application of the frozen gates and buffers may
determine eligibility.

## Final exposed-design termination rule

This is the last exposed-panel design attempt. The two-candidate set is closed
before evaluation. No candidate, parameter value, band edge, tie rule, gate,
buffer, or selection priority may be added or changed after a v11 result is
observed. This prohibition applies whether v11 fails or succeeds; the exposed
panel cannot be used for a v12 design iteration.

If v11 returns `no_eligible_candidate`, the shield design search stops and no
fresh panel may be opened. If v11 selects candidate 1, that result still does
not authorize validation, test, promotion, or a paper claim. The only allowed
forward step is to prepare and separately review a fail-closed validation
preflight that binds the immutable v11 selection. The validation panel must
remain unopened until that separate preflight and its authorization are
reviewed explicitly. V11 itself never grants fresh-panel permission.
