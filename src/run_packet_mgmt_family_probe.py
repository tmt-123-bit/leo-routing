"""Packet-management family probe (development tier).

The project narrative moves to "LEO + packet management", so the
packet-management action itself must be compared as a method family, not
only against no-management. This runner fixes the routing policy
(GlobalDijkstra, deterministic, no training variance) on the TLE-66
topology and varies ONLY the environment-side packet-management arm:

  baseline:   base      FIFO, env default (what every prior arm inherits)
  scheduling: edf       earliest absolute deadline first (RC-EDF lineage)
              class_priority  strict traffic-class priority, FIFO within class
              lcfs      newest-first (AoI-style preemptive replacement)
  dropping:   codel     drop the HOL packet whose sojourn exceeds a target
              red       random early drop by instantaneous queue depth
              dropfront drop one HOL packet while the queue exceeds a threshold
  ours:       purge     feasibility purge (BFS optimistic lower bound)
              purge_srpf  purge + shortest-remaining-path-first (the stack)
              purge_edf   purge + EDF cross combo (is SRPF the best partner?)

Stage ``tune``: mini panel (load 12, 10 workloads) to pick the codel target,
RED thresholds, and dropfront threshold from small grids; the tuned values
are frozen into tuning.json and used for the main panel (tune-on-dev,
documented in the manifest).

Stage ``main``: 10 arms x 40 fresh workloads x loads {12 primary, 8 knee,
4 light} with arrival+physical digests asserted across arms per workload;
per-(load, arm) bootstrap ratio-of-means vs base (5,000 draws, one-sided
95% lower bound) and exact sign test on paired per-workload gains.

Development tier: no sealed workloads, no promotion claim; the point is the
family ordering, not a significance badge.
"""

import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from hypatia_topology_provider_stub import HypatiaTopologyProvider
from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import (
    MULTIAGENT_LOADS,
    MultiAgentConfig,
    SynchronousLeoMultiAgentEnv,
)
from mappo_evaluation import GlobalDijkstraPolicy, evaluate_policy_with_constraint_metrics
from run_development_load_diagnostics import (
    DiagnosticWrapper,
    canonical,
    packet_accounting,
    read_json,
    sha,
    write_json,
)
import run_development_load_diagnostics as diagnostics
from run_infeasible_drop_probe import PurgeInfeasibleEnv
from run_srpf_probe import LeastSlackEnv

TOPOLOGY = Path("../data/starlink_66_links.csv")
SCENARIO = "hotspot_high_load"
MAIN_LOADS = (12, 8, 4)
TUNE_WORKLOADS = list(range(880511, 880521))
MAIN_WORKLOADS = list(range(880551, 880591))

NEW_DROP_REASONS = ("codel_sojourn", "red_early", "front_pressure")
for _reason in NEW_DROP_REASONS:
    if _reason not in diagnostics.DROP_REASONS:
        diagnostics.DROP_REASONS = diagnostics.DROP_REASONS + (_reason,)


class FamilyEnvBase(SynchronousLeoMultiAgentEnv):
    """Slot-boundary packet-management hook; no routing/reward changes."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.managed_count = 0

    def observe(self):
        self._manage()
        return super().observe()

    def _manage(self):
        return

    def _deadline_bound(self, packet_id):
        packet = self.packets[packet_id]
        deadline = self.cfg.packet_class_deadlines[packet.traffic_class]
        return min(packet.created_slot + deadline - 1, self.cfg.episode_slots)


class EDFEnv(FamilyEnvBase):
    """Earliest-deadline-first queue order, FIFO within equal deadlines."""

    def _manage(self):
        if self.slot > self.cfg.episode_slots:
            return
        for sat in range(1, self.n_agents + 1):
            queue = self.queues[sat]
            if len(queue) > 1:
                self.queues[sat] = deque(sorted(
                    queue, key=lambda pid: (self._deadline_bound(pid), pid)))


class ClassPriorityEnv(FamilyEnvBase):
    """Strict traffic-class priority (urgent classes first), FIFO within."""

    def _manage(self):
        if self.slot > self.cfg.episode_slots:
            return
        for sat in range(1, self.n_agents + 1):
            queue = self.queues[sat]
            if len(queue) > 1:
                self.queues[sat] = deque(sorted(
                    queue,
                    key=lambda pid: (self.packets[pid].traffic_class,
                                     self._deadline_bound(pid), pid)))


class LCFSEnv(FamilyEnvBase):
    """Last-come-first-served: the newest packet takes the head of line."""

    def _manage(self):
        if self.slot > self.cfg.episode_slots:
            return
        for sat in range(1, self.n_agents + 1):
            queue = self.queues[sat]
            if len(queue) > 1:
                self.queues[sat] = deque(sorted(
                    queue,
                    key=lambda pid: (-self.packets[pid].created_slot, -pid)))


class CoDelEnv(FamilyEnvBase):
    """Drop the HOL packet once its queueing sojourn exceeds a target."""

    def __init__(self, cfg, sojourn_target=8):
        super().__init__(cfg)
        self.sojourn_target = int(sojourn_target)

    def _manage(self):
        if self.slot > self.cfg.episode_slots:
            return
        for sat in range(1, self.n_agents + 1):
            queue = self.queues[sat]
            if queue:
                hol = queue[0]
                if self.slot - self.packets[hol].created_slot > self.sojourn_target:
                    queue.popleft()
                    self._drop_packet(hol, "codel_sojourn")
                    self.managed_count += 1


class REDEnv(FamilyEnvBase):
    """Random early drop on instantaneous queue depth (no ECN marking)."""

    def __init__(self, cfg, min_th=6, max_th=18, max_p=0.1):
        super().__init__(cfg)
        self.min_th, self.max_th, self.max_p = int(min_th), int(max_th), float(max_p)
        self._rng = np.random.default_rng([int(cfg.seed), 20260925])

    def _manage(self):
        if self.slot > self.cfg.episode_slots:
            return
        for sat in range(1, self.n_agents + 1):
            queue = self.queues[sat]
            depth = len(queue)
            if depth > self.min_th:
                p = min(self.max_p, self.max_p * (depth - self.min_th)
                        / (self.max_th - self.min_th))
                kept = deque()
                for pid in queue:
                    if self._rng.random() < p:
                        self._drop_packet(pid, "red_early")
                        self.managed_count += 1
                    else:
                        kept.append(pid)
                self.queues[sat] = kept


class DropFrontEnv(FamilyEnvBase):
    """Pressure drop: while over a threshold, remove the HOL packet."""

    def __init__(self, cfg, queue_threshold=12):
        super().__init__(cfg)
        self.queue_threshold = int(queue_threshold)

    def _manage(self):
        if self.slot > self.cfg.episode_slots:
            return
        for sat in range(1, self.n_agents + 1):
            queue = self.queues[sat]
            if len(queue) > self.queue_threshold:
                hol = queue.popleft()
                self._drop_packet(hol, "front_pressure")
                self.managed_count += 1


class PurgeEDFEnv(PurgeInfeasibleEnv, FamilyEnvBase):
    """Feasibility purge, then earliest-deadline-first ordering (cross combo)."""

    def _purge_infeasible_packets(self):
        if self.slot > self.cfg.episode_slots:
            return
        if not any(self.queues.values()):
            return
        self._refresh_graph()
        reverse = defaultdict(list)
        for (u, v), edge in self.graph.items():
            if u >= 1 and v >= 1 and edge.available:
                reverse[v].append(u)
        distances = {}
        for sat in range(1, self.n_agents + 1):
            for packet_id in self.queues[sat]:
                dst = self.packets[packet_id].dst
                if dst not in distances:
                    distances[dst] = self._bfs_distances(dst, reverse)
        for sat in range(1, self.n_agents + 1):
            if not self.queues[sat]:
                continue
            kept = deque()
            for packet_id in self.queues[sat]:
                packet = self.packets[packet_id]
                deadline = self.cfg.packet_class_deadlines[packet.traffic_class]
                deadline_bound = packet.created_slot + deadline - 1
                horizon_bound = self.cfg.episode_slots
                effective_bound = min(deadline_bound, horizon_bound)
                remaining = effective_bound - self.slot
                dist = distances[packet.dst].get(sat)
                if dist is not None and dist > remaining + self.safety_margin_slots:
                    self._drop_packet(packet_id, self.drop_reason_name)
                    self.purged_count += 1
                    self.purged_by_class[packet.traffic_class] += 1
                else:
                    kept.append(packet_id)
            self.queues[sat] = deque(sorted(
                kept, key=lambda pid: (self._deadline_bound(pid), pid)))


def build_arms(tuned):
    """Arm registry; ordering is the report ordering."""
    return {
        "base": {"cls": None, "kwargs": {}, "family": "baseline"},
        "edf": {"cls": EDFEnv, "kwargs": {}, "family": "scheduling"},
        "class_priority": {"cls": ClassPriorityEnv, "kwargs": {}, "family": "scheduling"},
        "lcfs": {"cls": LCFSEnv, "kwargs": {}, "family": "scheduling"},
        "codel": {"cls": CoDelEnv,
                  "kwargs": {"sojourn_target": tuned["codel_sojourn_target"]},
                  "family": "dropping"},
        "red": {"cls": REDEnv,
                "kwargs": {"min_th": tuned["red_min_th"], "max_th": tuned["red_max_th"],
                           "max_p": tuned["red_max_p"]},
                "family": "dropping"},
        "dropfront": {"cls": DropFrontEnv,
                      "kwargs": {"queue_threshold": tuned["dropfront_threshold"]},
                      "family": "dropping"},
        "purge": {"cls": PurgeInfeasibleEnv, "kwargs": {}, "family": "ours"},
        "purge_srpf": {"cls": LeastSlackEnv, "kwargs": {}, "family": "ours"},
        "purge_edf": {"cls": PurgeEDFEnv, "kwargs": {}, "family": "ours"},
    }


class FamilyWrapper(DiagnosticWrapper):
    _provider = None

    def __init__(self, workload_seed, load, arm_spec):
        cfg = MultiAgentConfig(
            env=EnvConfig(seed=workload_seed, scenario=SCENARIOS[SCENARIO],
                          topology_provider=FamilyWrapper._provider,
                          n_planes=11, sats_per_plane=6),
            initial_packets=load, exogenous_packets_per_slot=load,
            seed=workload_seed, variant="qos_only",
        )
        super().__init__(scenario=SCENARIO, cfg=cfg, seed=workload_seed)
        if arm_spec["cls"] is not None:
            self.env = arm_spec["cls"](self.env.cfg, **arm_spec["kwargs"])


def run_row(policy, label, workload, load, arm_spec):
    wrapper = FamilyWrapper(workload, load, arm_spec)
    result = evaluate_policy_with_constraint_metrics(
        SCENARIO, label, policy, -1, [workload],
        wrapper_factory=lambda _: wrapper, variant="qos_only",
    )[0]
    row = asdict(result)
    extra, _packets = packet_accounting(wrapper.env)
    row.update(extra)
    row.update(load=load, arm=label.split("::")[-1], label=label,
               managed=int(getattr(wrapper.env, "managed_count", 0)),
               purged_infeasible=int(getattr(wrapper.env, "purged_count", 0)),
               physical_sha256=wrapper.physical_digest.hexdigest())
    if sum(row[f"drop_{r}"] for r in diagnostics.DROP_REASONS) != row["dropped"]:
        raise AssertionError("drop partition mismatch")
    return row


def bootstrap_ratio(base, mech, draws=5000, seed=20260925):
    rng = np.random.default_rng(seed)
    n = len(base)
    out = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        out.append(mech[idx].mean() / base[idx].mean() - 1.0)
    out = np.array(out)
    return {
        "ratio_of_means": float(mech.mean() / base.mean() - 1.0),
        "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
        "one_sided_95_lower": float(np.percentile(out, 5.0)),
    }


def sign_test(gains):
    positive = int((gains > 0).sum())
    n = len(gains)
    from math import comb
    tail = sum(comb(n, k) for k in range(0, min(positive, n - positive) + 1))
    return {"positive": positive, "n": n,
            "exact_two_sided_p": float(min(1.0, 2.0 * tail / 2 ** n))}


def pair_stats(rows, load, arm, base_rows):
    by_workload = {r["workload_seed"]: r for r in rows if r["load"] == load and r["arm"] == arm}
    base_by_workload = {r["workload_seed"]: r for r in base_rows if r["load"] == load}
    if set(by_workload) != set(base_by_workload):
        raise AssertionError(f"unpaired workloads at load={load} arm={arm}")
    workloads = sorted(by_workload)
    delivery = np.array([by_workload[w]["delivery_ratio"] for w in workloads])
    base_delivery = np.array([base_by_workload[w]["delivery_ratio"] for w in workloads])
    stats = {
        "arm": arm,
        "mean_delivery": float(delivery.mean()),
        "delta_pp_vs_base": float(100 * (delivery.mean() - base_delivery.mean())),
        "paired_gain_pp": [float(100 * (by_workload[w]["delivery_ratio"]
                                        - base_by_workload[w]["delivery_ratio"]))
                           for w in workloads],
        "mean_managed": float(np.mean([by_workload[w]["managed"] for w in workloads])),
        "mean_purged": float(np.mean([by_workload[w]["purged_infeasible"] for w in workloads])),
    }
    if arm != "base":
        stats["bootstrap_vs_base"] = bootstrap_ratio(base_delivery, delivery)
        stats["sign_test"] = sign_test(np.array(stats["paired_gain_pp"]))
    return stats


def write_report(output, loads, arms, per_load_stats, tuned, stage):
    lines = [f"# Packet-management family probe — {stage}", "",
             f"Tuned params: `{json.dumps(tuned, sort_keys=True)}`", ""]
    for load in loads:
        lines += [f"## Load {load}", "",
                  "| arm | family | mean delivery | Δ vs base (pp) | one-sided 95% lower | sign p | managed/pkt-slot |",
                  "|---|---|---|---|---|---|---|"]
        for stat in per_load_stats[load]:
            arm = stat["arm"]
            lower = (stat.get("bootstrap_vs_base", {}).get("one_sided_95_lower")
                     if arm != "base" else None)
            p = stat.get("sign_test", {}).get("exact_two_sided_p") if arm != "base" else None
            lines.append(
                f"| {arm} | {arms[arm]['family']} | {stat['mean_delivery']:.4f} "
                f"| {stat['delta_pp_vs_base']:+.2f} "
                f"| {('' if lower is None else f'{lower:+.3f}')} "
                f"| {('' if p is None else f'{p:.4f}')} "
                f"| {stat['mean_managed']:.1f} |")
        lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "tune", "main"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, default=None,
                        help="tuning.json from the tune stage (required for main)")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    grids = {
        "codel_sojourn_target": [4, 8, 12],
        "red_grid": [(6, 18, 0.1), (12, 24, 0.1), (6, 18, 0.3)],
        "dropfront_threshold": [8, 16, 24],
    }

    if args.stage == "main":
        if args.tuning is None:
            raise ValueError("main stage requires --tuning")
        tuned = read_json(args.tuning.resolve())["tuned"]

    FamilyWrapper._provider = HypatiaTopologyProvider.from_csv(TOPOLOGY)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()

    if args.stage in ("smoke", "tune"):
        tune_arms = {}
        for target in grids["codel_sojourn_target"]:
            tune_arms[f"codel_t{target}"] = {"cls": CoDelEnv,
                                             "kwargs": {"sojourn_target": target},
                                             "family": "dropping"}
        for min_th, max_th, max_p in grids["red_grid"]:
            tune_arms[f"red_{min_th}_{max_th}_{max_p}"] = {
                "cls": REDEnv, "kwargs": {"min_th": min_th, "max_th": max_th, "max_p": max_p},
                "family": "dropping"}
        for th in grids["dropfront_threshold"]:
            tune_arms[f"dropfront_{th}"] = {"cls": DropFrontEnv,
                                            "kwargs": {"queue_threshold": th},
                                            "family": "dropping"}
        tune_arms["base"] = {"cls": None, "kwargs": {}, "family": "baseline"}
        if args.stage == "smoke":
            tune_arms["edf"] = {"cls": EDFEnv, "kwargs": {}, "family": "scheduling"}
            tune_arms["purge_edf"] = {"cls": PurgeEDFEnv, "kwargs": {}, "family": "ours"}
        workloads = TUNE_WORKLOADS[:4] if args.stage == "smoke" else TUNE_WORKLOADS
        rows = []
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
            for arm_name, spec in tune_arms.items():
                for w in workloads:
                    row = run_row(GlobalDijkstraPolicy(), f"dijkstra::{arm_name}", w, 12, spec)
                    stream.write(canonical(row) + "\n")
                    rows.append(row)
                stream.flush()
                print(f"{len(rows)} {arm_name} elapsed={time.monotonic()-started:.0f}s", flush=True)
        means = defaultdict(list)
        for row in rows:
            means[row["arm"]].append(row["delivery_ratio"])
        tuning = {
            "stage": args.stage, "workloads": workloads, "load": 12,
            "mean_delivery_by_arm": {a: float(np.mean(v)) for a, v in means.items()},
            "grids": {
                "codel_sojourn_target": grids["codel_sojourn_target"],
                "red_grid": [list(g) for g in grids["red_grid"]],
                "dropfront_threshold": grids["dropfront_threshold"],
            },
        }
        if args.stage == "tune":
            best_codel = max((a for a in means if a.startswith("codel")),
                             key=lambda a: means[a])
            best_red = max((a for a in means if a.startswith("red_")),
                           key=lambda a: means[a])
            best_dropfront = max((a for a in means if a.startswith("dropfront")),
                                 key=lambda a: means[a])
            tuning["tuned"] = {
                "codel_sojourn_target": int(best_codel.split("_t")[1]),
                "red_min_th": int(best_red.split("_")[1]),
                "red_max_th": int(best_red.split("_")[2]),
                "red_max_p": float(best_red.split("_")[3]),
                "dropfront_threshold": int(best_dropfront.split("_")[1]),
            }
            write_json(output / "tuning.json", tuning)
        else:
            write_json(output / "smoke.json", tuning)
        print(canonical(tuning), flush=True)
        return

    arms = build_arms(tuned)
    write_json(output / "manifest.json", {
        "role": "packet_mgmt_family_probe_dev",
        "runner_sha256": sha(Path(__file__)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_note": ("development-tier family comparison; routing fixed to "
                          "GlobalDijkstra; only the environment-side packet-management "
                          "arm varies; tuned dropping params frozen from the tune stage"),
        "tuned_params": tuned,
        "topology": str(TOPOLOGY), "topology_sha256": sha(TOPOLOGY),
        "workloads": MAIN_WORKLOADS, "loads": list(MAIN_LOADS),
        "arms": {k: {"family": v["family"], "kwargs": v["kwargs"]} for k, v in arms.items()},
        "training": False, "sealed_test_access": False,
        "new_drop_reasons": list(NEW_DROP_REASONS),
    })
    rows = []
    pairing = {}
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for load in MAIN_LOADS:
            for w in MAIN_WORKLOADS:
                for arm_name, spec in arms.items():
                    row = run_row(GlobalDijkstraPolicy(), f"dijkstra::{arm_name}", w, load, spec)
                    events = (row["arrival_sha256"], row["physical_sha256"])
                    key = (load, w)
                    if key in pairing:
                        if pairing[key] != events:
                            raise AssertionError(f"pairing changed at {key}")
                    else:
                        pairing[key] = events
                    stream.write(canonical(row) + "\n")
                    rows.append(row)
                stream.flush()
            print(f"{len(rows)} load={load} done elapsed={time.monotonic()-started:.0f}s", flush=True)
    base_rows = [r for r in rows if r["arm"] == "base"]
    per_load_stats = {}
    for load in MAIN_LOADS:
        stats = []
        for arm_name in arms:
            stats.append(pair_stats(rows, load, arm_name, base_rows))
        per_load_stats[load] = stats
    write_json(output / "family_stats.json", per_load_stats)
    write_report(output, MAIN_LOADS, arms, per_load_stats, tuned, "main")
    with (output / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
        fields = ["load", "arm", "mean_delivery", "delta_pp_vs_base", "one_sided_95_lower",
                  "sign_p", "mean_managed"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for load in MAIN_LOADS:
            for stat in per_load_stats[load]:
                writer.writerow({
                    "load": load, "arm": stat["arm"],
                    "mean_delivery": f"{stat['mean_delivery']:.6f}",
                    "delta_pp_vs_base": f"{stat['delta_pp_vs_base']:.4f}",
                    "one_sided_95_lower": (stat.get("bootstrap_vs_base", {}).get("one_sided_95_lower")
                                           if "bootstrap_vs_base" in stat else ""),
                    "sign_p": (stat.get("sign_test", {}).get("exact_two_sided_p")
                               if "sign_test" in stat else ""),
                    "mean_managed": f"{stat['mean_managed']:.2f}",
                })
    write_json(output / "completion.json", {
        "status": "complete", "episodes": len(rows),
        "pairings_verified": len(pairing),
        "elapsed_seconds": time.monotonic() - started,
        "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    })
    print("family probe complete", flush=True)


if __name__ == "__main__":
    main()
