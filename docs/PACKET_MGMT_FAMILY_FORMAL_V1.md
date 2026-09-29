# PACKET_MGMT_FAMILY_FORMAL_V1 — preregistered formal protocol

Frozen before touching the formal panel. Created 2026-09-26.

## Question

Does the preregistered family ordering from the development probe
(`experiments/packet-mgmt-family-20260925-v1`) hold on a fresh one-shot
panel: is the feasibility-purge packet-management stack (purge+SRPF) the
best packet-management arm, with the value of dropping policies monotonically
improving in criterion awareness?

## Arena

- Topology: TLE-66 (`data/starlink_66_links.csv`), scenario `hotspot_high_load`.
- Routing policy fixed: `GlobalDijkstraPolicy` (deterministic, untrained).
- Only the environment-side packet-management arm varies. No training, no
  sealed-panel access, no re-tuning on formal workloads.

## Panels (fresh workload seeds, one-shot)

- Gate: 880591-880610 (20 workloads), load 12 only.
- Formal: 880611-880650 (40 workloads), loads 12 (primary) / 8 (secondary) /
  4 (descriptive, harmlessness).

The formal panel must not be evaluated unless the gate passes.

## Arms (10, identical to the dev probe)

base (FIFO), edf, class_priority, lcfs, codel, red, dropfront, purge,
purge_srpf, purge_edf.

Dropping-family hyperparameters are frozen from the dev tune panel
(`packet-mgmt-family-tune-20260925-v1/tuning.json`): codel sojourn target 12,
RED (min_th 12, max_th 24, max_p 0.1), dropfront threshold 16. These are the
tuned-best values per family (tune-on-dev, disclosed); opponents run at
tuned-best.

## Statistics (frozen)

- Paired per-workload delivery ratio, workloads as the pairing unit.
- Bootstrap ratio-of-means vs the declared reference: 5,000 draws, seed
  20260926; report 95% CI and one-sided 95% lower bound.
- Exact sign test on paired gains (one-sided binomial tail, p0 = 0.5).

## Endpoints and success rules

- **P1 (primary, must pass)**: load 12, `purge_srpf` vs `base` — one-sided
  95% lower bound of relative gain >= +10%, AND one-sided sign-test p < 0.001.
- **S1 (secondary)**: load 12, `purge_srpf` vs `class_priority` (the
  strongest alternative arm in the dev probe) — one-sided lower bound > 0
  AND one-sided sign-test p < 0.01. Claim: family champion.
- **S2 (secondary, joint)**: load 12, `purge` vs each of `codel`, `red`,
  `dropfront` — one-sided lower bound > 0 each; joint rule: at least 2 of 3
  pass. Claim: dropping-family value grows with criterion awareness.
- **D (descriptive, no claim)**: loads 8 and 4; all other orderings;
  `purge_edf` vs `purge_srpf` partner comparison; EDF/EDC/LCFS orderings.

P1 must pass for the study to be reported as a formal success; S1/S2 are
reported with their own predeclared thresholds and are labeled secondary;
no other significance claims are allowed.

## Disclosures

- The dev probe (`packet-mgmt-family-20260925-v1`, workloads 880551-880590,
  load 12/8/4) and its tune panel (880511-880520) were run before this
  protocol was frozen; the formal panel (880611-880650) is untouched by any
  prior run and is evaluated exactly once.
- No sealed workloads are used. Sealed range 78001-78050 remains untouched.
- Runner: `src/run_packet_mgmt_family_formal.py` (sha recorded in the
  manifest); protocol sha recorded in the manifest and verified at runtime.
