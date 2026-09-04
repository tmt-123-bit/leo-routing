# Avoidable-Switch Constraint Formal Study v1

## Status and scope

This document preregisters the formal training and validation stage of the
independent avoidable-switch-constraint series. It is separate from the
terminated v7-v11 shield series and follows the completed mechanism smoke in
`experiments/avoidable-switch-constraint-smoke-v1-r1`.

The formal stage may select and freeze checkpoints and may record whether the
MAPPO method-stage validation gate passed. It cannot, by itself, declare the
study eligible for sealed-test authorization: the separately preregistered
classical-baseline freeze and a later joint authorization protocol are also
required. Formal validation is not an unseen test and cannot, by itself,
support a paper performance claim. This protocol does not authorize any access
to workload seeds `78001..78050`.

No result from the mechanism smoke selected or changed the constraint budget,
dual learning rate, architecture, reward, optimizer, formal workload panels,
training horizon, checkpoint rule, statistical test, or validation gate.

The only accepted predecessor is
`experiments/avoidable-switch-constraint-smoke-v1-r1`, bound by:

```text
smoke report self-hash = e9d57ff0c8e5f6a2805c507c2fba4fe15b5d21635038af9f3a42f9814d468274
training freeze self-hash = 30f4f969442ec2ecf8e48b724a094a47c888a323b04e49168913c8474b131034
smoke spec self-hash = 3fad6686610f52993b584f6c70ea39150f2c9262cff973584fca1d512eebc129
exact-resume proof self-hash = 96ccb677d7299c290570aa058145a676f8319f94430176d8d0b5ab07984e059b
```

The earlier `experiments/avoidable-switch-constraint-smoke-v1` directory is a
preserved failed preflight and cannot satisfy this predecessor contract.

## Research question and claim boundary

The primary question is whether a QoS-only candidate-set MAPPO policy with an
explicit decision-level avoidable-switch constraint can reduce avoidable
next-hop decisions while retaining practically comparable delivery under
medium and concentrated-hotspot load.

The method is PPO with a projected Lagrange multiplier. It is not CPO and does
not provide a hard per-trajectory, per-seed, or deployment-time guarantee.
MAPPO itself is not claimed as novel. Any later paper claim must be limited to
the empirical decision-level constraint mechanism and the audited simulator.

The historical `proposed/no_lifetime` MAPPO method is retrained as a fair
reward-shaped control. Its reward counts accepted physical switch events,
whereas the new endpoint counts pre-contention avoidable decisions. It is not
a fixed-coefficient implementation of the same decision-level cost, so this
study cannot claim that an adaptive dual dominates an equivalent fixed
penalty.

## Frozen arms and scenarios

The scenarios, in analysis order, are:

```text
medium_load
hotspot_high_load
```

`medium_load` represents the nominal congested operating regime and
`hotspot_high_load` is the concentrated-traffic stress regime in which the
existing ICC study exposed its delivery/churn trade-off. Both were fixed in
the smoke protocol before smoke outcomes existed; they were not selected by
ranking smoke performance. Even a successful later test supports claims only
for these two synthetic load regimes, not broad failure robustness or
five-scenario generalization.

The arms, in pairing order, are:

```text
qos_only_baseline
qos_only_constrained
reward_shaped_control
```

`qos_only_baseline` and `qos_only_constrained` both use the canonical
`qos_only` environment, for which local and team rewards contain no switch
penalty. Only `qos_only_constrained` enables the explicit constraint.
`reward_shaped_control` uses canonical `proposed`, historically named
`no_lifetime`, and does not enable the explicit constraint.

All three arms use the same architecture, optimizer, train workloads,
validation workloads, policy-seed identities, training horizon, validation
schedule, and checkpoint schedule. Every arm is trained from initialization;
the number of reused historical or smoke checkpoints is zero.

## Frozen decision-level constraint

For every active policy decision before contention resolution:

```text
o = 1 iff the cached next hop and at least one alternative are both feasible
c = o * 1{the policy selects a different next hop}
C_rollout = sum(c) / sum(o)
```

The first route, forced reroutes, NO_OP, inactive agents, and padding have
`o=0` and `c=0`. A proposal is counted before contention even if contention
later blocks it.

```text
actor loss = L_PPO + lambda_used * C_surrogate

lambda[k+1] = clip(
    lambda[k] + 0.05 * (C_rollout - 0.12),
    0,
    5,
)
```

One multiplier is fixed across all epochs and minibatches from one completed
rollout. It is updated exactly once after that rollout. A zero-opportunity
rollout skips the update. There is no warmup, smoothing, or hidden controller
state.

```text
B_switch = 0.12
dual_lr = 0.05 per completed rollout
lambda_init = 0.0
lambda_projection = [0.0, 5.0]
```

## Seed registry

Formal policy seeds are derived before data access from the first four bytes
of SHA-256 over namespace
`ICC-AVOIDABLE-SWITCH-CONSTRAINT-v1-formal-policy-seed-` followed by counters
`0..7`, with the high bit cleared:

```text
179055553
626669596
746965870
183895110
310818925
2136406109
985998595
515636025
```

The workload assignments are:

```text
formal train workloads:       76001..76200
checkpoint-selection workloads: 77001..77010
independent gate workloads:      77011..77020
sealed test workloads:        78001..78050  (metadata only; access forbidden)
```

All smoke and retired ranges in
`AVOIDABLE_SWITCH_CONSTRAINT_SMOKE_V1.md` remain retired. Formal train and
validation seed identities are assigned permanently on publication of this
protocol, including after interruption. A failed seed cannot be replaced.

There are `2 scenarios x 3 arms x 8 policy seeds = 48` training jobs. Within
each scenario and policy seed, the three arms are paired on identical workload
orders. The independently trained policy seed, not an episode row, is the
algorithmic replication unit.

## Frozen training budget

```text
total_timesteps target = 50000
batch_size = 4 complete episodes per rollout
epochs = 3
minibatches = 4
validation every 40 rollouts
checkpoint-selection episodes = 10
checkpoint interval = 5000 environment steps
maximum parallel jobs = 2
maximum launches per training job = 2
performance early stopping = disabled
```

The trainer completes the first whole rollout that reaches or exceeds 50,000
environment steps and records the actual boundary. With the current 30-slot
episodes this is expected to be 50,040 steps; the audit derives the allowed
overshoot from the environment rather than treating 50,040 as an outcome
target. Roughly ten validation candidates per job are expected.

The actor and critic dimensions, Adam learning rates, decay, discount,
GAE/TD-lambda, normalization, PPO clipping, entropy coefficient, gradient
limits, target KL, candidate actor, QoS terms, and all other training semantics
must equal the completed smoke contract field by field. Extending a run by
changing its target is forbidden. An interrupted run may resume only from its
same-job `latest.pt`, with the source checkpoint copied and hashed before it is
overwritten.

## Checkpoint selection

Every job trains to the full budget before selection.

For `qos_only_baseline` and `reward_shaped_control`, select the lexicographic
maximum of:

```text
(delivery_ratio, mean_reward, -drop_rate, -average_delay_slots)
```

For `qos_only_constrained`, a candidate with `sum(o)=0` is not estimable and
fails that training job closed immediately; it is not skipped or replaced by a
later candidate. For estimable candidates, first retain candidates satisfying
`sum(c)/sum(o) <= 0.12 + 1e-12`, then select the lexicographic maximum of:

```text
(delivery_ratio,
 class_2_delivery_ratio,
 mean_reward,
 -drop_rate,
 -average_delay_slots,
 -decision_avoidable_switch_rate,
 -environment_steps)
```

The `1e-12` term is only a frozen numerical comparison tolerance in the
checkpoint-selection helper. It is not a scientific budget relaxation and is
not used by the independent gate. If no constrained candidate is feasible,
retain the candidate that is lexicographically minimal under:

```text
(decision_avoidable_switch_rate,
 -delivery_ratio,
 -class_2_delivery_ratio,
 -mean_reward,
 drop_rate,
 average_delay_slots,
 environment_steps)
```

This is the minimum-rate candidate, with QoS metrics breaking rate ties and
the earliest checkpoint breaking a complete tie. Mark that job infeasible,
fail the formal validation gate, and do not authorize test access.

Training-time checkpoint selection uses only `77001..77010`. After all
training is complete, each selected checkpoint is evaluated again on those 10
workloads solely to verify exact checkpoint reproduction, and is separately
evaluated on the untouched gate panel `77011..77020`. The 480 selection-recheck
rows and 480 independent-gate rows, integer numerators and denominators, source
checkpoint hashes, and environment schema are persisted in separate roles.
Only the independent-gate rows enter the statistical gate. Episode rates are
never averaged to estimate a policy-seed rate.

The selection recheck requires exact job, checkpoint, workload, integer-ledger,
step, episode-count, and seed-start identity. All recomputed floating selection
metrics use absolute tolerance `1e-8`; integer fields are exact. The independent
gate uses the strict scientific comparison `rate <= 0.12` with zero additional
numerical tolerance.

Training-time selection and both post-training evaluations use the same frozen
CUDA device. Mixed CPU/CUDA reproduction is not part of this protocol; the
runtime records the GPU model, CUDA stack, and dependency inventory.

## Formal validation estimands

For policy seed `i`, workload `j`, constrained arm `c`, and QoS baseline `b`:

```text
Delta_D = mean_i mean_j (delivery_c[i,j] - delivery_b[i,j])

R_a[i] = sum_j cost_a[i,j] / sum_j opportunity_a[i,j]
Delta_R = mean_i (R_c[i] - R_b[i])
R_c = mean_i R_c[i]
```

The reward-shaped control is a secondary fairness comparison and does not
expand the primary multiplicity family.

The engineering delivery non-inferiority margin is an absolute `0.02`. This
means a loss of two delivered packets per 100 generated packets is the largest
independent-gate loss accepted before spending the sealed panel. The margin is
a prospective operational tolerance and was not estimated from smoke outcomes.

For each scenario, the validation go/no-go conditions are:

1. Every constrained selected checkpoint is feasible on the selection panel;
   every constrained policy seed has nonzero gate opportunity and a gate-panel
   ratio-of-sums rate at or below `0.12`.
2. The one-sided 95% crossed-bootstrap upper bound for `R_c` is at or below
   `0.12`.
3. The one-sided 95% crossed-bootstrap upper bound for `Delta_R` is below zero.
4. The one-sided 95% crossed-bootstrap lower bound for `Delta_D` is at least
   `-0.02`.

The gate panel is disjoint from checkpoint selection. These are formal
development gates, not final paper claims, because the panel is used to decide
whether to spend the sealed test. Failure closes the sealed panel. Method
changes after failure require a new versioned research series with new train,
selection, and gate panels.

## Frozen statistical procedure

The two primary effect families are constrained-minus-QoS-baseline delivery
and decision-rate differences in each of the two scenarios, for four tests in
total.

- Build complete `8 x 10` paired gate matrices. Missing, duplicate, substituted,
  or non-finite cells fail the analysis.
- For delivery, average paired workload differences within each policy seed.
- For decision rate, aggregate cost and opportunity within each policy seed
  before differencing arms.
- Enumerate all `2^8 = 256` policy-seed sign assignments for a two-sided exact
  sign-flip sensitivity test, including tail ties under the frozen
  machine-precision comparison tolerance `8 * eps * scale`.
- Adjust the four primary p-values by Holm at familywise alpha `0.05`.
- Generate 5,000 crossed pigeonhole-bootstrap replicates by independently
  resampling the eight policy-seed rows and 10 gate-workload columns. Ratio
  endpoints are recomputed from resampled numerators and denominators. A draw
  with an undefined zero denominator is rejected and redrawn; both attempted
  and rejected replicate counts are reported. At most 5,000,000 total draws
  may be attempted to obtain the 5,000 defined replicates; exhausting that
  fixed resource bound fails the analysis closed rather than returning a
  partial interval.
- Report two-sided 95% intervals for effects and the prespecified one-sided
  bounds used by the validation gate. Percentiles use NumPy's default `linear`
  quantile interpolation.
- Use a frozen deterministic RNG namespace and record the integer seed for
  every scenario/endpoint analysis.

The two frozen secondary contrast families are
`reward_shaped_control - qos_only_baseline` and
`qos_only_constrained - reward_shaped_control`, each reported for delivery and
decision rate in both scenarios. They are descriptive, unadjusted, and cannot
alter a validation gate.

Report raw values, effect sizes, intervals, raw and adjusted p-values, policy-
seed directions, zero differences, and all failures. Class-specific delivery,
drop, throughput, mean/p95 delay, queue occupancy, accepted/avoidable/forced
switches, reward components, control overhead, and dual trajectories are
secondary or descriptive and cannot rescue failed primary conditions.

The two-sided delivery sign-flip result tests a zero difference and is a
sensitivity analysis; it is not evidence for the `-0.02` non-inferiority
hypothesis. The independent-gate lower bound alone implements the development
non-inferiority rule. A later sealed-test protocol must preregister a one-sided
non-inferiority procedure and its intersection/ordering with budget attainment
and rate reduction before test access.

## Integrity and stopping rules

Before a validation report can be finalized:

- all 48 jobs, 480 selection-recheck rows, and 480 independent-gate rows exist;
- train, validation, arm, scenario, and policy-seed identities match exactly;
- every logged constraint transition and pre-contention ledger recomputes;
- run configs, source, dependency inventory, every retained checkpoint,
  selected and final/latest checkpoints, manifests, logs, and row shards are
  hashed; constrained arms retain every deferred selection candidate, while
  online legacy selection is audited from its complete validation trace;
- preflight executes only the six explicitly named, hash-bound test modules;
  test discovery and globbing are forbidden, and the expected count is fixed;
- every training and panel-isolated evaluation status is self-hashed, binds its
  ordered attempt history, and is frozen together with its completed artifact;
- exact-resume verification remains valid for the frozen trainer fingerprint;
- no child process remains active and no output escaped its job root;
- `test_panel_consulted=false`, `test_access_count=0`, and
  `sealed_test_instantiated=false` remain true.

There is no efficacy, futility, variance, or significance early stopping.
Infrastructure failure permits one controlled retry with identical parameters,
seeds, code fingerprint, and budget, preferably by exact resume. A second
failure leaves the matrix incomplete: no seed replacement, imputation, or
complete-case headline analysis is allowed. A code or leakage defect pauses
the whole study and requires a documented amendment before symmetric reruns.
The two-launch cap applies independently to each training job and to each
post-training `(job, panel)` evaluation shard. Selection-recheck and
independent-gate attempt histories are isolated. Scheduling is fail-fast: a
failure stops new submissions, in-flight evaluation workers check cancellation
between workload seeds, and a failed selection-recheck panel prevents the
independent gate from starting.

Retry eligibility is fail-closed and recorded as `failure_category` plus
`retry_authorized` in the self-hashed attempt status. Automatic retry or retry
after process re-entry is allowed only for a recovered orphaned `running`
attempt, Python `TimeoutError`, Python `ConnectionError`, or `OSError` carrying
one of these frozen errno names:

```text
EAGAIN EBUSY ECONNABORTED ECONNRESET EHOSTUNREACH EINTR EIO
ENETDOWN ENETRESET ENETUNREACH EPIPE ESTALE ETIMEDOUT
```

Configuration, schema, source-fingerprint, hash, audit, scientific-computation,
and all other runtime failures record `retry_authorized=false` and cannot
consume a second attempt on re-entry. `KeyboardInterrupt`, `SystemExit`, and an
unbound manual or fail-fast cancellation are not retryable. A parallel worker
cancelled by fail-fast shutdown is retryable only when its cancellation is
explicitly bound to the first worker's recognized infrastructure failure;
cancellation caused by any non-infrastructure failure remains non-retryable.

## Sealed-test boundary and remaining ICC evidence

This protocol never opens `78001..78050`. Even when every MAPPO validation gate
passes, its output records `eligible_for_separate_test_authorization=false`,
`classical_baseline_freeze_bound=false`, and
`joint_authorization_prerequisites_complete=false`. It may only point to the
classical-baseline freeze and joint protocol as the next required stage. A
later one-time authorization may be written only after those prerequisites and
all training, evaluator, analysis code, manifests, and selected checkpoints
are frozen by hash. That authorization must predeclare every method and
scenario, then evaluate the entire panel regardless of interim results. Test
outcomes cannot tune the method, replace a seed, select a scenario, or select a
checkpoint.

Before an ICC submission presents broad comparative performance, Q-routing
must be retrained on the new train panel with the same eight policy seeds, and
Q-routing, OSPF-ECMP, and Global Dijkstra must be evaluated under the same
eventual one-time test authorization. Old ICC figures use accepted episode
switch totals, not the new decision-level estimand, and therefore cannot be
reused as the new method's main result.

Their definitions, Q-routing budget and hyperparameters, stochastic ECMP seed
handling, future comparison family, and claim limits are already frozen in
`AVOIDABLE_SWITCH_CLASSICAL_BASELINES_FORMAL_V1.md`. The formal preregistration
must bind that document by hash before the first formal workload is used.
