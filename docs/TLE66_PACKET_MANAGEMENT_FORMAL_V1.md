# TLE-66 Packet-Management Formal Study v1

Status: preregistration. This document is committed to Git BEFORE any formal
panel workload is instantiated. The development evidence cited below used
disjoint workload ranges and completed before this protocol was written.

## Research claim

In the 66-satellite real-TLE environment (11 planes x 6 satellites, Starlink
2026-07 frozen TLEs, SGP4, per-slot connectivity validated), the
packet-management stack — provably-safe infeasibility purge (margin 1) plus
SRPF queue reordering — improves the delivery ratio of the strongest baseline
(plain Global Dijkstra, deterministic) by at least +10% relative in the deep
saturation regime, with uncertainty quantified by workload-level bootstrap.
The mechanism is method-agnostic; secondary arms report persistent routing
(ILPR-style) with and without the stack as the fairness contrast.

## Development evidence (motivation only, not confirmatory)

- Load curve (Dijkstra base vs +stack): +5.8/+6.8/+9.9/+13.3/+15.6% at loads
  4/6/8/10/12 (workloads 880031-880040).
- Confirmations: load 10 +12.97% (one-sided 95% lower +10.28%), load 12
  +16.20% (CI 12.39-19.83, lower +12.99%), each on 40 fresh workloads
  (880041-880080 and 880081-880120), 40/40 positive at both points.
- 24-satellite environment: preregistered formal result +7.80% (one-sided 95%
  lower +7.15%), unchanged and independent of this study.

## Frozen system definition

- Environment: `data/starlink_66_links.csv` (hash recorded in the run
  manifest), topology provider, n_planes=11, sats_per_plane=6, scenario
  semantics `hotspot_high_load`, 30 slots.
- Stack: purge of certainly-infeasible packets at slot boundaries with
  safety margin 1 (alpha=beta=0), SRPF reordering by ascending BFS hop
  distance, identical code for all arms.
- Policies: plain `GlobalDijkstraPolicy` (primary baseline, deterministic,
  one sentinel identity) and `PersistentDijkstraPolicy` (ILPR-style fairness
  arm, deterministic).
- Loads: primary endpoint at exogenous=12/initial=12; secondary at 10/10.
- No training, no selection, no tuning inside this study.

## Workload partition

- Development (already consumed): 880001-880120.
- Gate: 880131-880150 (20 workloads). If the gate shows the stack's mean
  delivery at load 12 not above plain Dijkstra's, the study stops and reports
  the failure; the formal panel is never touched.
- Formal one-shot: 880151-880200 (50 workloads), each evaluated once per
  arm per load. No interim looks, no result-dependent changes.

## Endpoints and decision rules

Primary (load 12): relative delivery gain of `dijkstra_mech` vs
`dijkstra_base`. Success requires BOTH:

1. the one-sided 95% workload-bootstrap lower bound of the relative gain is
   at least +10%;
2. at least 48 of 50 workloads show a positive paired gain.

Secondary (load 10): same two conditions, reported as a supporting endpoint
(no additional claim is authorized from it beyond its own interval).

Fairness contrast (reported regardless): `persistent_mech` vs
`persistent_base` relative gain at both loads.

Statistics: 5,000 bootstrap draws resampling the 50 workloads with
replacement; paired per-workload relative gains; exact two-sided sign test
over the 50 workload signs as sensitivity; Holm correction across the
primary family (two gated conditions treated as one joint success rule, so
no multiplicity adjustment beyond the joint rule).

Excluded arms, recorded as dev-tier context in the final report (not part of
this formal grid): table Q-routing (collapsed at this scale in development),
budget-matched DQN and 50k-step MAPPO (policies not persisted for frozen
reuse; both far below Dijkstra in development).

## Execution stages

1. This protocol is committed to Git (timestamped freeze).
2. Gate run on 880131-880150 (both Dijkstra arms, load 12 only).
3. If the gate passes, the formal runner executes the full grid on
   880151-880200: 2 loads x 4 arms x 50 workloads = 400 episodes, one shot,
   writing per-episode rows and a self-hashed completion record.
4. Frozen statistics computed once from the formal rows; results published
   with the manifest.

## Amendment rule

Any change after the protocol commit requires a numbered amendment stating
what changed and why, committed before the affected stage runs.
