# IEEE Paper Outline — honest, evidence-mapped framing

Target: IEEE/ACM Trans. on Networking / IEEE INFOCOM / IEEE Trans. AES.
All claims below are mapped to the experiment that must support them after the
re-run (`run_ieee_reproduction.sh`). Do NOT write a claim whose evidence cell
is empty or "TBD".

---

## Title (honest)
**Distributed Next-Hop Routing for LEO Satellite Networks via Multi-Agent PPO:
Approaching Global-Optimal Delivery with Local 1-Hop Information**

One-line thesis: a decentralized MAPPO policy using only local queue + 1-hop
neighbor/link state approaches global-state-optimal Dijkstra within a few
percentage points, while dominating distributed baselines — the *information
asymmetry* is the contribution, not superiority over a centralized oracle.

## Abstract (structure)
1. Problem: LEO routing under fast topology/traffic change; centralized
   per-hop recompute is unrealistic; local-only policies ignore queue/load.
2. Method: Dec-POMDP next-hop routing; 24 sats = 24 agents; shared
   candidate-neighbor Actor; CTDE with graph Critic; team reward + zero-mean
   local credit.
3. Headline result: within 0.5–3.7 pp of centralized Dijkstra delivery across 5
   scenarios using only 1-hop info; **+12 to +35 pp over Q-routing and
   heuristics**; lower P95 tail and better load-balance under stress;
   Starlink→OneWeb zero-shot transfer −7.57 pp.
4. Honest caveat: does not comprehensively beat global Dijkstra; the comparison
   is decentralized-vs-centralized (favorable, not like-for-like).

## Contributions (ordered by strength of evidence)
| # | Claim | Evidence (after re-run) | Confidence |
|---|---|---|---|
| C1 | **Distributed dominance.** MAPPO beats Q-routing & all local heuristics in all 5 scenarios (wide margins). | `IEEE-EXP-004-FULL/paired_tests.csv`, p_BH<1e-8 | High |
| C2 | **Local-only near-optimality.** Within 0.5–3.7 pp of centralized Dijkstra **and OSPF/ECMP** on delivery, using only 1-hop info. | `aggregate_metrics.csv` per scenario + 95% CI (now incl. ospf_ecmp column) | High |
| C3 | **Cross-constellation zero-shot transfer** Starlink→OneWeb, −7.57 pp, no fine-tune. | TLE re-run (both directions) | High |
| C4 | **Credit assignment validated** — removing local credit: −9.4 pp (medium), −5.6 pp (frequent), p<1e-15. Lead the ablation section with this. | `IEEE-ABLATION-FULL/paired_ablation_effects.csv` | High |
| C5 | **Tail + balance under stress** — lower P95 and better Jain load-balance in hotspot/fault. | `aggregate_metrics.csv` P95/queue | Medium |
| D1 | Queue-awareness: SUPPORTED but modest (−3.4 pp medium only). Report, do not title. | ablation | Medium |
| D2 | Graph Critic: report as architectural exploration; adopt flat as default unless graph wins after the Phase-1 Critic fix at 50K steps. | ablation re-run | Pending fix |
| D3 | Link-lifetime mask: NOW testable (real breaks in 2.1). If protective → positive result; else drop. | ablation re-run | Pending fix |

## Section outline
1. **Intro** — 3 problems (topology/traffic churn; queue/load ignored by
   shortest-path; local optima collide). Contribute C1–C3.
2. **Related work** — Q-routing (1996), queue-aware LEO routing
   (arXiv:2306.01346), MAPPO in cooperative games (arXiv:2103.01955), Hypatia,
   distributed-LEO-routing survey (ScienceDirect). Position vs centralized OSPF.
3. **System model** — 24-sat Walker; ISL model; traffic classes + QoS floors;
   the (now-real) link-failure and (now-binding) bandwidth models.
4. **Method** — Dec-POMDP next-hop formulation; 26(+1)-dim candidate features
   (incl. the new contention feature); shared Actor; CTDE Critic; team reward
   (eq.) + zero-mean credit `r_i = R_team + β(r_local_i − mean)`.
5. **Training** — MAPPO/PPO details; **return normalization + separate critic
   clip + Huber** (the Phase-1 fix — frame as correct stabilization, cite the
   10× faster Critic convergence as a minor result); 200 train seeds + LR decay;
   train/val/test workload split.
6. **Evaluation**
   - 6.1 Setup: 5 scenarios, 8 policy seeds, hierarchical bootstrap, MDE on nulls.
   - 6.2 Main results (Table I = C1/C2): delivery, drop, P95, queue, switches.
     Baselines: Q-routing, full_heuristic (distributed) vs Dijkstra, **OSPF/ECMP**
     (centralized oracles). OSPF/ECMP isolates the "learning vs shortest-path"
     question — report MAPPO−ECMP gap separately from MAPPO−Dijkstra.
   - 6.3 Ablation (Table II = C4/D1/D2/D3) across all 5 scenarios.
   - 6.4 Cross-constellation (C3).
   - 6.5 **Training-budget sensitivity** (Fig.: delivery-vs-steps with crossover)
     — turns the former reproducibility bug into a sample-efficiency result.
   - 6.6 ns-3 packet-level validation (domain-adapted deployment; fixture-injected
     ISL outages; scale to 100+ drops per `run_ieee_reproduction.sh` ns-3 stage).
7. **Discussion / limitations** — explicit: does not beat Dijkstra; 8 seeds;
   ns-3 uses domain-adapted obs + 33-sat fixture (note topology mismatch, plan
   Starlink-24 fixture); TLE uses simplified ISL/capacity assumptions.
8. **Conclusion + future work** — real dynamic topology, full ns-3 closed loop,
   larger constellations.

## Figures (minimum set)
- F1 delivery-vs-training-steps with 95% bands + baseline crossovers (6.5).
- F2 per-scenario delivery (MAPPO vs Dijkstra vs Q-routing vs heuristics) (6.2).
- F3 P95 tail + Jain load-balance, stress scenarios (6.2/C5).
- F4 ablation effect plot across all 5 scenarios (6.3).
- F5 Critic explained-variance, with-vs-without normalization (minor result, 6.5)
  — reuse the smoke validation.
- F6 ns-3 closed-loop trace (delivery under ISL outages) (6.6).

## Required-honest-caveats checklist (copy into §7)
- [ ] "MAPPO does not comprehensively beat global Dijkstra on delivery."
- [ ] Headline claims require the FULL 50K-step budget (state the minimum).
- [ ] Delay always paired with delivery (survivor bias).
- [ ] Queue win is vs distributed baselines, not Dijkstra (Dijkstra lower queue).
- [ ] ns-3 = domain-adapted deployment validation, not identical-semantics transfer.
- [ ] TLE = real SGP4 positions but assumed ISL/capacity/reliability.

## Anti-rejection checklist (from the audit)
- [x] ≥8 policy seeds (done in code), report policy-seed std separately.
- [x] MDE on every "no significant difference" — `compute_mde.py` (Phase 4d);
      validated on old FULL data (frequent_break MAPPO≈Dijkstra: MDE = 1.02 pp,
      i.e. any difference ≥ ~1 pp would have been detected). Re-run on new data
      after the FULL re-run (`run_ieee_reproduction.sh mde`).
- [x] Stronger baselines: **OSPF/ECMP added** (`OspfEcmpPolicy`, Phase 4a) —
      the canonical distributed incumbent; wired into eval loop + paired tests.
      (Decentralized SP+Q greedy ≈ existing `full_heuristic`; OSPF/ECMP was the
      real gap.)
- [x] Ablation across all 5 scenarios (Phase 4b; was 2).
- [x] Reproducibility: `--mode full` documented; budget presets `x2k`/`x10k`
      added (Phase 4c); `write_repro_manifest.py` (Phase 4e) emits pip freeze +
      git refs + source/fixture SHA256. Run it with the experiment python before
      submission; commit both repos first for a clean manifest.
