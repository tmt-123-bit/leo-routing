# Switch-Residual MAPPO Preflight v5

## Status

This protocol defines a diagnostic exploratory preflight. It is not a
confirmatory experiment, does not authorize a paper claim, and cannot be used
as final IEEE Transactions evidence.

v5 is a post-v4 mechanism diagnostic. The v4 greedy-margin policy failed its
switching gates: avoidable-switch rate and total routing switches worsened in
both scenarios, with the hotspot deterioration present on every paired test
workload. Those exposed v4 panels are not reused by v5.

## Frozen Mechanism

The deployed policy retains the schema-v2 decoupled adaptive hysteresis rule:

```text
adaptive_scale = (1 - 0.50 * urgency) * (1 - 0.00 * class_2)
stay_bonus = (0.20 + projected_residual) * adaptive_scale
projected_residual in [0.00, 0.20]
```

The residual is a single projected non-negative actor parameter initialized at
zero. PPO continues to train the normal actor. The switch regularizer receives
the actual deployed policy logits, including adaptive urgency relief, but its
gradient is isolated to the cached-route residual:

```text
penalty = relu(best_switch_actual - best_stay_actual + 0.01)
loss = actor_primary_loss + 0.01 * mean(penalty | eligible and active)
```

There is no counterfactual restoration of full hysteresis and no additional
urgency weighting. This targets the same decision boundary used at inference
while preventing the switch objective from rewriting learned route quality.
Adaptive hysteresis remains active at inference.

The existing avoidable-switch metric is not relaxed for urgent traffic. A
switch is avoidable whenever the cached route is feasible, regardless of
urgency or traffic class. Urgency affects the deployed policy bonus only; it
does not exempt or relabel an avoidable switch in validation or test metrics.

## Frozen Configuration

- Scenarios: `medium_load`, `hotspot_high_load`
- Policy seeds: `1710210210`, `2078783072`
- Training steps: 6,000 per scenario and policy seed
- Training workload seeds: `9001..9200`
- Validation workload seeds: `48001..48010`
- Test workload seeds: `49001..49025`
- Adaptive hysteresis beta: `0.20`
- Urgency relief: `0.50`
- Class-2 relief: `0.00`
- Residual parameterization: `projected_nonnegative_scalar`
- Residual initialization and cap: `0.00`, `0.20`
- Regularizer mode: `isolated_greedy_logit_margin`
- Regularizer coefficient and margin: `0.01`, `0.01`
- Validation delivery and class-2 tolerances: `0.003`, `0.010`
- Validation selection: minimum avoidable-switch micro-rate after delivery and
  class-2 filters, using validation data only

The validation and test panels are disjoint from training and all panels
exposed through v4.

## Integrity Requirements

Each candidate, selected, and final checkpoint must embed and exactly match:

- candidate actor schema 4, including residual parameterization, initialization,
  and cap;
- switch regularizer schema 2, including actual-policy-logit source and
  cached-route-residual-only gradient scope;
- validation-only selection schema 4;
- the complete run configuration and code fingerprint.

The runner must replay the frozen selector, recompute validation micro-rates
from exact totals, verify checkpoint and artifact hashes, and reject panel or
contract drift. The projected residual must remain within `[0.00, 0.20]`.

## Interpretation Gates

Results are inspected per scenario and policy seed. Aggregation cannot hide a
failing seed.

- Delivery difference versus proposed must be at least `-0.003`.
- Class-2 delivery difference versus proposed must be at least `-0.010`.
- Avoidable-switch rate must be lower than proposed.
- Total routing switches must not exceed raw context.
- Forced-switch reductions cannot compensate for worse avoidable switching.

Passing these gates permits only a broader, separately frozen confirmatory
design. It does not make this two-policy-seed preflight final paper data.
