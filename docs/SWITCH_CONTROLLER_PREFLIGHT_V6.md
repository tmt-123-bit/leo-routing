# Switch-Controller MAPPO Preflight v6

## Status

This protocol defines a diagnostic exploratory preflight. It is not a
confirmatory experiment, does not authorize an IEEE Transactions paper claim,
and cannot be reported as final paper data. The two policy seeds are a
mechanism check, not an adequate final evidence base.

v6 follows the failed v5 diagnostic. v5 aligned the isolated regularizer with
the deployed logits, but three of four scenario/seed cells failed at least one
frozen delivery, class-2, avoidable-switch, or raw-switch gate. Its regularizer
continued to receive nonzero pressure, but its scalar residual could not adapt
differently to calm and urgent traffic and did not show a beneficial hotspot
dose-response. v5 validation and test panels are not reused for v6 selection
or test.

## Frozen Mechanism

This is a modification of the existing MAPPO policy, not a replacement model.
The actor uses the schema-v5 urgency-linear residual controller:

```text
urgency in [0, 1]
projected_calm_residual in [0.00, 0.20]
projected_urgent_residual in [0.00, 0.20]
residual(urgency) = (1 - urgency) * projected_calm_residual
                    + urgency * projected_urgent_residual
adaptive_scale = (1 - 0.50 * urgency) * (1 - 0.00 * class_2)
stay_bonus = (0.20 + residual(urgency)) * adaptive_scale
```

Both endpoints start at zero. PPO continues to train the normal route-quality
actor parameters. The isolated switch regularizer receives the actual deployed
policy logits, and its gradient updates only the two residual endpoints:

```text
penalty = relu(best_switch_actual - best_stay_actual + 0.01)
loss = actor_primary_loss + 0.01 * rollout_micro_mean(penalty)
```

Eligibility is recorded before contention. Each minibatch contributes its
eligible sum scaled so that the arithmetic mean across active minibatches is
the rollout-wide micro mean. This prevents minibatch partitioning, including
empty eligible partitions, from changing the regularizer objective.

The environment variant is `with_avoidable_switch_reward`. It retains the
congestion-context observations but excludes forced reroutes from local and
team switch costs. Accounting metrics are unchanged: forced and avoidable
switches remain separately reported, and the avoidable-switch metric is not
relaxed for urgent or class-2 traffic.

The integrated treatment is trained and replayed with that reward variant.
Frozen `raw_context` and post-hoc reference checkpoints remain bound to their
native `with_congestion_context` variant, and the frozen proposed checkpoint
remains bound to `proposed`. The two context variants have identical
observations and transition dynamics; they differ only in the training reward.

## Frozen Configuration

- Screen: `SWITCH-CONTROLLER-PREFLIGHT-v6`, runner schema 6
- Environment: `with_avoidable_switch_reward`
- Scenarios: `medium_load`, `hotspot_high_load`
- Policy seeds: `1710210210`, `2078783072`
- Training steps: 6,000 per scenario and policy seed
- Training workload seeds: `9001..9200`
- Validation workload seeds: `50001..50010`
- Test workload seeds: `51001..51025`
- Adaptive hysteresis beta: `0.20`
- Urgency relief: `0.50`
- Class-2 relief: `0.00`
- Residual parameterization: `urgency_linear`
- Residual endpoint initialization and cap: `0.00`, `0.20`
- Regularizer mode: `isolated_greedy_logit_margin`
- Regularizer coefficient and margin: `0.01`, `0.01`
- Regularizer reduction: `rollout_micro_mean`
- Validation delivery and class-2 tolerances: `0.003`, `0.010`
- Validation selection: minimum avoidable-switch micro-rate after delivery and
  class-2 filters, using validation data only

The validation and test panels are disjoint from training and every workload
panel exposed through v5.

## Integrity Requirements

Each candidate, selected, and final checkpoint must embed and exactly match:

- candidate actor schema 5 with marker
  `projected_nonnegative_urgency_linear_endpoints`;
- switch regularizer schema 3 with actual-policy logits, endpoint-only
  gradients, pre-contention eligibility, and rollout-micro minibatch weighting;
- validation-only selection schema 5;
- the complete run configuration, environment variant, and code fingerprint.

The runner must replay the frozen selector, recompute validation micro-rates
from exact totals, verify checkpoint and artifact hashes, and reject panel or
contract drift. Both learned endpoint values must remain in `[0.00, 0.20]`.

## Test Isolation and Interpretation

The test panel must not be inspected while choosing coefficients, margins,
caps, training duration, checkpoint-selection rules, or any other mechanism.
Only the frozen validation panel may select a checkpoint. A failed validation
or test gate ends this design: it must not be promoted, retuned on the same
test workloads, or described as evidence for a paper claim.

Results are inspected per scenario and policy seed. Aggregation cannot hide a
failing seed.

- Delivery difference versus proposed must be at least `-0.003`.
- Class-2 delivery difference versus proposed must be at least `-0.010`.
- Avoidable-switch rate must be lower than proposed.
- Total routing switches must not exceed raw context.
- Forced-switch reductions cannot compensate for worse avoidable switching.

Passing every gate permits only a broader, separately frozen confirmatory
experiment with more policy seeds. It still does not make this preflight IEEE
Transactions-level final data.
