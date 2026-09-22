# Amendment 1 to TLE66_PACKET_MANAGEMENT_FORMAL_V1

## What changed and why

Development screening on 2026-09-24 identified EDR-style backpressure
(max-weight with shortest-path bias, gamma tuned to 4.0 over a five-point
sweep {0,1,2,4,8}) as a stronger baseline than plain Dijkstra at load 12
(base 0.3754 vs 0.2755 on twenty workloads). A forty-workload confirmation
gave mechanism-on-backpressure +18.31% (one-sided 95% lower bound +16.39%,
40/40 positive, minimum workload gain +10.81%). The original study's frozen
primary (Dijkstra reference) stands unchanged; this amendment adds a new
formal stage that preregisters the strongest-baseline comparison the paper
should lead with.

## Added stage (new one-shot panel)

- Panel: workloads 880351..880400 (50 workloads, never touched).
- Arms: `bp_base` and `bp_mech` (BackpressurePolicy, gamma=4.0 frozen,
  deterministic; mech = identical purge+SRPF stack code).
- Load: 12 (primary). No other loads, no other arms, no interim looks.
- Primary success: one-sided 95% workload-bootstrap lower bound of
  bp_mech vs bp_base relative delivery gain >= +10% AND >= 48/50 workloads
  with positive paired gain.
- Statistics: identical to the base protocol (5,000 draws, ratio of means,
  paired signs, exact two-sided sign test as sensitivity).
- The amendment is committed before this stage runs; no result from the new
  panel existed at commit time.
