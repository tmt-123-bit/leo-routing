"""Packet-stratified analysis and replay-only queue residence diagnostics."""

from collections import Counter, defaultdict
from dataclasses import asdict
import argparse
import csv
import gzip
import json
from pathlib import Path
import time

import torch

from mappo_evaluation import evaluate_policy_with_constraint_metrics, load_checkpoint_policy
from run_development_load_diagnostics import (
    DiagnosticWrapper, PersistentDijkstraPolicy, ROOT, canonical, make_wrapper,
    packet_accounting, read_json, sha, write_json,
)

IDENTITY = ("scenario", "load", "policy", "policy_seed", "workload_seed")
FOCUS = {"medium_load": 6, "hotspot_high_load": 16}
REPLAY_METHODS = ("qos_only_constrained", "qos_only_baseline", "persistent_global_dijkstra")


def hotspot_id(config):
    env = config["env"]
    return env["scenario"]["force_hotspot_dst"] or (
        ((env["n_planes"] + 1) // 2 - 1) * env["sats_per_plane"] + (env["sats_per_plane"] + 1) // 2
    )


def decision_flags(row):
    neighbors, mask = row["neighbors"], row["mask"]
    old = row["old_next_hop"]
    feasible = [i for i, n in enumerate(neighbors) if n and mask[i + 1]]
    old_index = neighbors.index(old) if old in neighbors else None
    old_feasible = old_index in feasible
    old_queue = row["neighbor_queues"][old_index] if old_feasible else None
    previous_queue = row["neighbor_previous_queues"][old_index] if old_feasible else None
    rising = old_feasible and old_queue > previous_queue
    lower_alternative = old_feasible and any(row["neighbor_queues"][i] < old_queue for i in feasible if i != old_index)
    holding = old_feasible and row["action"] == old_index + 1
    direct_index = neighbors.index(row["destination"]) if row["destination"] in neighbors else None
    direct_feasible = direct_index in feasible
    return {
        "old_feasible": old_feasible, "old_queue_rising": rising,
        "old_rising_with_lower_alternative": rising and lower_alternative,
        "holding": holding, "holding_rising": holding and rising,
        "holding_rising_with_lower_alternative": holding and rising and lower_alternative,
        "direct_feasible": direct_feasible,
        "direct_bypassed": direct_feasible and row["action"] != direct_index + 1,
    }


def csv_rows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def packet_tables(source, manifest):
    packets, groups, locations = {}, defaultdict(Counter), defaultdict(Counter)
    with gzip.open(source / "packets.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            packet = json.loads(line)
            if packet["load"] != FOCUS[packet["scenario"]]:
                continue
            identity = tuple(packet[k] for k in IDENTITY)
            packet_key = (*identity, packet["packet_id"])
            if packet_key in packets:
                raise AssertionError("duplicate packet identity")
            packets[packet_key] = packet
            destination = "hotspot" if packet["dst"] == hotspot_id(manifest["configs"][packet["scenario"]]) else "other"
            key = (packet["scenario"], packet["policy"], destination, packet["traffic_class"])
            count = groups[key]
            count["generated"] += 1
            count[packet["status"]] += 1
            count["hop_sum_" + packet["status"]] += packet["hop_count"]
            age = packet["cutoff_age_slots"] if packet["status"] == "queued" else packet["terminal_slot"] - packet["created_slot"] + 1
            count["age_sum_" + packet["status"]] += age
            if packet["drop_reason"]:
                count[packet["drop_reason"]] += 1
            if packet["status"] == "queued":
                count["queued_at_source"] += packet["hop_count"] == 0
                count["queued_at_least_5_slots_old"] += age >= 5
                locations[(packet["scenario"], packet["policy"], packet["owner"])][destination] += 1
    result = []
    for (scenario, method, destination, traffic_class), count in sorted(groups.items()):
        row = {"scenario": scenario, "policy": method, "destination_group": destination,
               "traffic_class": traffic_class, **count}
        for state in ("delivered", "dropped", "queued"):
            row[state] = count[state]
            row[state + "_pct"] = 100 * count[state] / count["generated"]
            row[state + "_mean_hops"] = count["hop_sum_" + state] / count[state] if count[state] else None
            row[state + "_mean_age_slots"] = count["age_sum_" + state] / count[state] if count[state] else None
        result.append(row)
    loc_rows = [{"scenario": s, "policy": m, "owner": n, "hotspot_packets": c["hotspot"],
                 "other_packets": c["other"], "queued": sum(c.values())}
                for (s, m, n), c in sorted(locations.items())]
    return packets, result, loc_rows


class ResidenceWrapper(DiagnosticWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residence = defaultdict(Counter)
        self.node_slots = defaultdict(Counter)
        self.hotspot = hotspot_id(asdict(self.env.cfg))

    def step(self, actions):
        gateway_nodes = {u for (u, v), edge in self.env.graph.items() if v == self.hotspot and edge.available}
        for node, queue in self.env.queues.items():
            for position, packet_id in enumerate(queue):
                packet = self.env.packets[packet_id]
                count = self.residence[packet_id]
                count["observed_queue_slots"] += 1
                count["non_hol_queue_slots"] += position > 0
                count["gateway_queue_slots"] += node in gateway_nodes
                count["source_queue_slots"] += packet.hop_count == 0
                count["queue_position_sum"] += position
                dest_group = "hotspot" if packet.dst == self.hotspot else "other"
                self.node_slots[(node, dest_group)]["queue_slots"] += 1
                self.node_slots[(node, dest_group)]["non_hol_slots"] += position > 0
                self.node_slots[(node, dest_group)]["gateway_slots"] += node in gateway_nodes
        return super().step(actions)


def summarize_decisions(decisions, packets, identity, counts, outcome_ids):
    for row in decisions:
        packet_key = (*identity, row["packet_id"])
        packet = packets[packet_key]
        destination = "hotspot" if packet["dst"] == 9 else "other"
        key = (identity[0], identity[2], destination)
        group = counts[key]
        group["decisions"] += 1
        flags = decision_flags(row)
        group.update({name: int(value) for name, value in flags.items()})
        group["opportunities"] += row["opportunity"]
        group["avoidable_switches"] += row["avoidable_switch"]
        for condition in ("holding_rising_with_lower_alternative", "direct_bypassed"):
            if flags[condition]:
                outcome_ids[(key, condition, packet["status"])].add(packet_key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    completion = read_json(source / "completion.json")
    if completion["status"] != "complete":
        raise ValueError("completed source run required")
    for name, digest in completion["output_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError("source result hash mismatch: " + name)
    manifest = read_json(source / "manifest.json")
    original = {tuple(row[k] for k in IDENTITY): row
                for row in (json.loads(line) for line in (source / "episodes.jsonl").read_text().splitlines())}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {
        "role": "descriptive_packet_analysis_and_exact_replay", "source_completion_sha256": sha(source / "completion.json"),
        "analyzer_sha256": sha(Path(__file__)), "replay_methods": REPLAY_METHODS,
        "focus_loads": FOCUS, "threshold": "old next-hop queue increased since preceding snapshot; at least one feasible alternative has strictly smaller queue",
        "warnings": ["lower queue need not mean better route", "decision-outcome association is not causal", "queue slots are predecision occupancy samples, not physical milliseconds"],
    })
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    packets, packet_rows, locations = packet_tables(source, manifest)
    csv_rows(output / "packet_groups.csv", packet_rows)
    csv_rows(output / "final_queue_locations.csv", locations)
    counts, outcome_ids = defaultdict(Counter), defaultdict(set)
    residence_counts, node_counts = defaultdict(Counter), defaultdict(Counter)
    replay_count, started = 0, time.monotonic()
    try:
        with gzip.open(output / "decisions.jsonl.gz", "wt", encoding="utf-8") as trace_file:
            for scenario, load in FOCUS.items():
                for method in REPLAY_METHODS:
                    for seed in manifest["policy_seeds"] if method != "persistent_global_dijkstra" else [-1]:
                        if seed == -1:
                            policy = PersistentDijkstraPolicy()
                        else:
                            artifact = next(a for a in manifest["artifacts"] if (a["scenario"], a["method"], a["seed"]) == (scenario, method, seed))
                            if sha(artifact["path"]) != artifact["sha256"]:
                                raise ValueError("checkpoint changed")
                            policy, _ = load_checkpoint_policy(Path(artifact["path"]), device="cpu")
                        for workload in manifest["workload_seeds"]:
                            cfg = make_wrapper(scenario, load, workload, diagnostic=False).env.cfg
                            wrapper = ResidenceWrapper(scenario=scenario, cfg=cfg)
                            metric = asdict(evaluate_policy_with_constraint_metrics(
                                scenario, method, policy, seed, [workload], wrapper_factory=lambda _: wrapper,
                            )[0])
                            identity = (scenario, load, method, seed, workload)
                            old = original[identity]
                            for field, value in metric.items():
                                if old[field] is None and field in ("average_delay_slots", "p95_delay_slots", "decision_avoidable_switch_rate"):
                                    continue
                                if value != old[field]:
                                    raise AssertionError(f"replay differs: {identity} {field}")
                            extra, current_packets = packet_accounting(wrapper.env)
                            if extra["arrival_sha256"] != old["arrival_sha256"] or wrapper.physical_digest.hexdigest() != old["physical_sha256"]:
                                raise AssertionError("replay event sequence differs")
                            for packet in current_packets:
                                if packet != {k: v for k, v in packets[(*identity, packet["packet_id"])].items() if k not in IDENTITY}:
                                    raise AssertionError("replay final packet state differs")
                                destination = "hotspot" if packet["dst"] == wrapper.hotspot else "other"
                                key = (scenario, method, destination, packet["traffic_class"], packet["status"])
                                residence_counts[key].update(wrapper.residence[packet["packet_id"]])
                                residence_counts[key]["packets"] += 1
                            for (node, destination), values in wrapper.node_slots.items():
                                node_counts[(scenario, method, node, destination)].update(values)
                            summarize_decisions(wrapper.decisions, packets, identity, counts, outcome_ids)
                            for decision in wrapper.decisions:
                                trace_file.write(canonical({**dict(zip(IDENTITY, identity)), **decision}) + "\n")
                            replay_count += 1
                        print(f"replayed {replay_count}/140 {scenario} {method} seed={seed} elapsed={time.monotonic()-started:.1f}s", flush=True)
        decision_rows = []
        for key, values in sorted(counts.items()):
            row = {"scenario": key[0], "policy": key[1], "destination_group": key[2], **values}
            denom = values["old_rising_with_lower_alternative"]
            row["hold_pct_when_old_rising_and_lower_alternative"] = 100 * values["holding_rising_with_lower_alternative"] / denom if denom else None
            row["direct_bypass_pct"] = 100 * values["direct_bypassed"] / values["direct_feasible"] if values["direct_feasible"] else None
            for condition in ("holding_rising_with_lower_alternative", "direct_bypassed"):
                for state in ("delivered", "dropped", "queued"):
                    row[condition + "_unique_packets_" + state] = len(outcome_ids[(key, condition, state)])
            decision_rows.append(row)
        csv_rows(output / "decision_groups.csv", decision_rows)
        csv_rows(output / "queue_residence.csv", [
            {"scenario": s, "policy": m, "destination_group": d, "traffic_class": c, "status": status, **v}
            for (s, m, d, c, status), v in sorted(residence_counts.items())])
        csv_rows(output / "node_queue_residence.csv", [
            {"scenario": s, "policy": m, "node": n, "destination_group": d, **v}
            for (s, m, n, d), v in sorted(node_counts.items())])
        write_json(output / "completion.json", {"status": "complete", "replays": replay_count,
                   "all_episode_and_packet_states_identical": True, "focus_packets_analyzed": len(packets),
                   "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    except Exception as error:
        write_json(output / "failure.json", {"error": repr(error), "completed_replays": replay_count})
        raise


if __name__ == "__main__":
    main()
