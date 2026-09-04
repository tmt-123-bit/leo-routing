# Avoidable-Switch Constraint Smoke v1

## Status and scope

This document freezes a new, independent research series. It does not reopen,
continue, reinterpret, or overwrite the terminated v7-v11 shield-design
series. All historical checkpoints, result tables, and protocol decisions are
retained unchanged.

The smoke is a mechanism-integrity exercise only. Its outcomes cannot select a
budget, dual learning rate, model architecture, reward, scenario, seed panel,
gate, or checkpoint rule. It cannot support a paper claim or a promotion
decision.

## Research question

Can QoS-only MAPPO control avoidable cached-next-hop churn through an explicit
decision-level rate constraint, without embedding switch cost in the QoS
reward?

The constrained objective is not CPO and does not claim per-trajectory hard
guarantees. It is PPO with a projected Lagrange multiplier applied to a frozen
surrogate for a rollout-level empirical rate constraint.

## Frozen decision-level definition

For each active policy decision before contention resolution:

```text
o = 1 iff the cached next hop and at least one alternative are both feasible
c = o * 1{the policy selects a different next hop}
C_rollout = sum(c) / sum(o)
```

The first route, forced reroutes, NO_OP, inactive agents, and padding have
`o=0` and `c=0`. A valid proposal counts at the decision stage even if it is
later blocked by contention.

The actor and dual contracts are:

```text
actor loss = L_PPO + lambda_used * C_surrogate

lambda[k+1] = clip(
    lambda[k] + 0.05 * (C_rollout - 0.12),
    0,
    5,
)
```

One `lambda_used` is fixed across every epoch and minibatch belonging to the
same completed rollout. Lambda is updated exactly once after that rollout. If
`sum(o)=0`, the update is skipped and lambda is unchanged. There is no
smoothing, warmup, or hidden controller state.

The frozen values are:

```text
B_switch = 0.12
dual_lr = 0.05 per completed rollout
lambda_init = 0.0
lambda_projection = [0.0, 5.0]
```

`B_switch=0.12` was chosen only from already exposed v7 evidence: the old
proposed policy had `3922/26405 = 0.148532475`, while the shield had
`3084/26465 = 0.116531268`. No seed in this protocol contributed to that
choice.

## Reward and arms

Both arms use the canonical `qos_only` variant. Physical switch accounting is
retained, but both the local and team rewards exclude all switch penalties.

```text
qos_only_baseline:    unconstrained QoS-only MAPPO
qos_only_constrained: the same MAPPO plus the frozen constraint above
```

The smoke crosses the two arms with `medium_load` and
`hotspot_high_load`, using one shared policy seed. This produces four 6,000
environment-step jobs. The same architecture, QoS reward, train workloads,
validation workloads, optimizer settings, and checkpoint schedule are used in
both arms. The constrained arm alone uses budget-constrained validation.

## Seed registry

The following seed ranges are permanently assigned on publication of this
protocol, even if a job is interrupted:

```text
smoke train workloads:       75001..75008
smoke validation workloads:  75101..75104
formal train workloads:      76001..76200
formal validation workloads: 77001..77020
sealed test workloads:       78001..78050
```

The smoke policy seed is derived without outcome data from SHA-256 namespace
`ICC-AVOIDABLE-SWITCH-CONSTRAINT-v1-smoke-policy-seed-0`, yielding
`197359353`.

The following ranges are retired and cannot be used by this series:

```text
9001..9200
10001..10050, 11001..11050, 12001..12050, 13001..13050, 14001..14050
16001..16020, 17001..17020, 18001..18015, 19001..19015, 21001..21020
31001..31050, 32001..32020, 33001..33050, 34001..34020
35001..35050, 37001..37050
41001..41010, 42001..42025, 43001..43010, 44001..44025
45001..45002, 46001..46010, 47001..47025
48001..48010, 49001..49025, 50001..50010, 51001..51025
60001..60010, 61001..61025, 62001..62002
70001..70025, 71001..71010, 72001..72025
```

The sealed test range may appear as immutable metadata, but no smoke code may
instantiate, evaluate, summarize, or otherwise inspect a workload from that
range.

## Smoke configuration

```text
timesteps = 6000
batch_size = 4 episodes per rollout
epochs = 3
minibatches = 4
validation episodes = 4
validation every 5 rollouts
checkpoint interval = 1500 environment steps
maximum parallel jobs = 2
```

Every completed update and validation boundary must atomically refresh an
exact-resume `latest.pt`. A valid resume restores the actor, critic, both
optimizers, lambda state, update counters, reward normalization state,
validation state, metric-file boundary, and Python, NumPy, Torch, and CUDA RNG
states.

Before the four smoke jobs, the runner performs a CPU deterministic equivalence
check using retired workload seed `60001`: two uninterrupted rollout updates
must exactly match one update followed by a resume from `latest.pt` and a
second update. The comparison covers actor, critic, optimizer, dual, trainer,
RNG, and training-update JSON state. This check is implementation verification
only and cannot contribute experimental evidence.

## Mechanism gates

All gates are implementation checks, not performance tuning criteria:

1. Both constrained scenarios have a positive opportunity denominator.
2. Every rollout has `0 <= cost <= opportunity`.
3. Forced reroutes never contribute decision-level constraint cost.
4. Every lambda transition is exactly recomputable from its logged inputs.
5. All losses, gradients, rates, multipliers, and validation metrics are finite.
6. Interrupted-versus-uninterrupted resume equivalence passes its deterministic
   test.
7. The sealed test access count remains zero and `test_panel_consulted=false`.

Constraint attainment, QoS differences, and arm ranking are descriptive only
in this smoke. They cannot change the frozen method.

## After the smoke

Passing the smoke permits a separately preregistered formal training and
validation run on `76001..76200` and `77001..77020`. It does not open the
sealed test panel. Formal comparison must retrain the unconstrained and
constrained QoS-only arms on identical splits. Any comparison to the historical
paper method must also retrain that method on the same new splits; an old
checkpoint cannot be presented as a provenance-matched control.

The test panel can be opened exactly once only after training is frozen,
validation gates pass, all code and checkpoints are hashed, and a separate
formal protocol authorizes access. All test outcomes must then be reported.
