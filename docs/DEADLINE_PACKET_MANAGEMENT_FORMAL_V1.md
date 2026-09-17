# Deadline-Aware Packet Management Formal Study v1

Status: draft preregistration. Everything in this document is fixed before any
formal-panel workload is instantiated. Development evidence cited below used
only the 910001..910180 development panels and already-completed training; no
sealed or reserved workload has been read.

## Research claim

In the 24-satellite slot simulator, a deadline-aware in-network packet
management stack — provably-safe infeasibility purge, direct-delivery
correction, and shortest-remaining-path-first queue reordering — combined with
a constrained MAPPO policy retrained in that environment improves hotspot
delivery ratio relative to frozen classical baselines evaluated as-is, while
keeping the avoidable-switch budget and bounded delay. The stack is
method-agnostic: classical baselines given the identical stack are also
evaluated and reported as the fairness contrast. No claim is made that the
routing algorithm alone beats SOTA, and no 10% threshold claim is made.

## Development evidence (motivation only, not confirmatory)

- Loss structure at hotspot load 16: ~53% backlog, ~17% deadline drops, queue
  overflow ~0; service capacity at the destination ingress binds
  (development-load-diagnostics-20260909-v1).
- Provably-safe purge (margin = 1 is the exact certain-death boundary; the
  dist == remaining+1 case is still salvageable): +2.1..2.6 pp on every
  policy arm, reproduced on three panels (infeasible-drop / least-slack /
  srpf / stacked probes).
- SRPF reorder: +0.7..1.3 pp on retrained policies, success delay -8%,
  success hops +10..19% (known trade-off, reported descriptively).
- Two independent 8-seed retrainings (purge env, SRPF env) produce system
  means 0.3326 / 0.3325 vs Q-routing base 0.3090 / 0.3114 (+7.62% / +6.80%),
  16/16 seeds positive (final-system-eval-20260917-v1, -srpf-v1).
- Fairness contrast: Q-routing with the identical stack reaches 0.3351 /
  0.3354 (+8.45% / +7.73% over its own base).
- Negative results retained: least-slack scheduling harmful; margin-0 purge
  harmful (off-by-one, now understood); aggressive admission (12-point alpha/
  beta grid) monotonically harmful — the provably-safe boundary is
  empirically optimal (aggressive-purge-sweep-20260917-v1).
- Medium load: mechanism effect on MAPPO ~0 (+0.07 pp, panel 910171..910180);
  system trails Q-routing base by ~1 pp. Medium is a secondary,
  non-inferiority-only endpoint.

## Frozen system definition

- Environment: purge of certainly-infeasible packets at slot boundaries with
  margin 1 (alpha = beta = 0), plus SRPF queue reordering (ascending BFS hop
  distance, stable packet-ID tie-break); identical code for all arms.
- Policy: constrained MAPPO (`qos_only` variant, switch budget 0.12), trained
  in the SRPF environment (`LEO_PURGE_INFEASIBLE=2`), eight policy seeds
  179055553, 183895110, 310818925, 515636025, 626669596, 746965870, 985998595,
  2136406109; checkpoint = argmax validation delivery among candidates within
  the 12% switch budget, selected on training validation only.
- Deployment wrapper: direct-delivery action override (destination in the
  original feasible mask).
- No retraining, reselection, or threshold change after the training freeze.

## Arms and grid

Scenarios: `medium_load` and `hotspot_high_load`.

- `system` — the frozen system above, 8 policy seeds.
- `q_base` — frozen Q-routing models as-is (same eight identities).
- `q_mech` — frozen Q-routing models with the identical stack (fairness).
- `qos_only_base` — frozen QoS-only MAPPO as-is, 8 seeds (internal reference).
- `pd_mech` — persistent Dijkstra with the identical stack, one sentinel.

Formal panel: workloads `940101..940150`, evaluated once for every arm
(33 arm-instances x 50 workloads = 1,650 rows). Checkpoint selection uses
training validation `77001..77010`; the independent gate uses `77011..77020`;
neither may be reused for the formal panel. Workload ranges 78001..78050
(sealed, consumed) and 930101..930120 (reserved for other studies) are
excluded.

## Endpoints and decision rules

Primary (hotspot): relative delivery gain of `system` vs `q_base`. The paper
claim is authorized only if the one-sided 95% crossed-bootstrap lower bound of
the relative gain is at least +3% AND every policy seed's relative gain is
positive.

Secondary (hotspot), all reported regardless of outcome:

- fairness contrast: `system` vs `q_mech` relative difference (expected <= 0;
  reported as a boundary, not a failure);
- per policy seed: avoidable-switch rate <= 0.12;
- success-delay mean relative change <= +5% (expected negative);
- success-hop mean relative change reported descriptively (SRPF trades hops
  for delivery; predeclared, not gated);
- per-class and other-destination delivery non-inferiority vs `q_base`
  (one-sided 95% lower bound >= -2 pp).

Secondary (medium): delivery non-inferiority of `system` vs `q_base`
(one-sided 95% lower bound >= -2 pp). The mechanism effect at medium is
expected ~0; no improvement claim is made for medium under any outcome.

Statistics: 5,000 crossed bootstrap draws over policy seeds and workloads;
two-sided exact sign-flip sensitivity over the eight seed effects; Holm
correction across the primary and non-inferiority family.

## Execution stages and stopping rule

1. Training freeze: medium-load 8-seed SRPF-environment training (hotspot
   runs already complete under dev manifests and must be re-bound by hash
   into the formal freeze without retraining).
2. Independent gate on `77011..77020`: if `system` delivery mean is not above
   `q_base` at hotspot, the study stops before the formal panel and reports
   the gate failure.
3. One-shot formal evaluation on `940101..940150`, followed by the frozen
   statistics. No interim looks, no result-dependent changes.

## Amendment rule

Any change to arms, endpoints, thresholds, workloads, or selection rules
after this document is committed requires a numbered amendment that states
what changed and why, before the affected stage runs.
