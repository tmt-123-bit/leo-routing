# Switch-Margin MAPPO Preflight v4

## Status

This protocol defines a diagnostic exploratory preflight. It is not a
confirmatory experiment, does not authorize a paper claim, and cannot by
itself establish IEEE Transactions-level evidence.

The v4 policy keeps the existing schema-v3 MAPPO actor, critic, environment,
training workloads, adaptive hysteresis, and validation selector. The only
algorithmic change from v3 is the avoidable-switch regularizer.

## Motivation

The v3 conditional-probability regularizer reduced total switches in some
conditions but continuously penalized every finite switch probability. Its
initial weighted loss was several times larger than the PPO actor objective,
and the hotspot reduction was driven by forced switches while the target
avoidable-switch rate worsened.

v4 uses a hinge on the best switching and cached-route logits:

```text
scale = (1 - 0.50 * urgency) * (1 - 0.00 * class_2)
full_beta_stay = best_stay + 0.20 * (1 - scale)
penalty = scale * relu(best_switch - full_beta_stay + 0.01)
loss = actor_primary_loss + 0.01 * mean(penalty | eligible and active)
```

Action zero is the wrapper's `NO_OP` action and is excluded. A decision is
eligible only when the cached route and at least one alternative route are
both feasible. Restoring the missing adaptive bonus before the hinge prevents
urgency relief from creating a penalty that did not exist in the corresponding
non-urgent state. The strictly positive margin also resolves argmax ties in
favor of the cached route.

## Frozen Configuration

- Scenarios: `medium_load`, `hotspot_high_load`
- Policy seeds: `1710210210`, `2078783072`
- Training steps: 6,000 per scenario and policy seed
- Training workload seeds: `9001..9200`
- Validation workload seeds: `46001..46010`
- Test workload seeds: `47001..47025`
- Adaptive hysteresis beta: `0.20`
- Urgency relief: `0.50`
- Class-2 relief: `0.00`
- Regularizer mode: `greedy_logit_margin`
- Regularizer coefficient: `0.01`
- Logit margin: `0.01`
- Validation delivery tolerance: `0.003`
- Validation class-2 tolerance: `0.010`
- Validation selection: minimum avoidable-switch micro-rate after the delivery
  and class-2 safety filters, using validation data only

The validation and test panels are disjoint from training and from panels
exposed by earlier screens or smoke runs through `45002`.

## Scale Calibration

Calibration used only training-loss and gradient scales, not delivery,
switch, validation, or test outcomes. A near-zero coefficient probe showed
that the hinge first became active around 840-1080 steps. The coefficient was
then checked on the fixed scale grid.

At coefficient `0.005`, the last-ten-update median weighted-regularizer to
actor-primary loss ratios were `0.183` and `0.185`, below the predeclared
`[0.25, 0.75]` band. At coefficient `0.01`, the ratios were `0.348` for
hotspot and `0.378` for medium; maxima were `0.575` and `0.418`. The maximum
weighted-regularizer to actor-primary gradient-norm ratios were `0.072` and
`0.055`. Therefore `0.01` was frozen without consulting performance metrics.

## Integrity Requirements

Each training artifact must embed and exactly match:

- the candidate actor specification;
- the full switch regularizer specification, including mode, coefficient,
  positive margin, reduction, `NO_OP`, feature indices, beta, relief values,
  and relief-weighting semantics;
- the validation-only selection specification;
- the run configuration and code fingerprint.

The runner replays the frozen checkpoint selector, recomputes validation
micro-rates from exact totals, verifies checkpoint hashes, and rejects source
or panel drift. Component loss identities and all logged values must remain
finite.

## Interpretation Gates

Results are inspected per policy seed; averages cannot hide a failing seed.

- Delivery difference versus proposed must be at least `-0.003` in every
  scenario and policy seed.
- Class-2 delivery difference versus proposed must be at least `-0.010` in
  every scenario and policy seed.
- Avoidable-switch rate must be lower than proposed in every scenario and
  policy seed.
- Total routing switches must not exceed raw context in every scenario and
  policy seed.
- A reduction in forced switches cannot compensate for a worse avoidable
  switch count or rate.

Passing these gates only permits a broader multi-seed confirmatory design. It
does not convert this two-policy-seed preflight into final paper data.
