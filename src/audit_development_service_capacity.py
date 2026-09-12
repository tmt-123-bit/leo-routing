"""Audit packet records and a topology-only optimistic delivery bound."""

from collections import Counter, defaultdict
from dataclasses import asdict
import argparse
import gzip
import json
from pathlib import Path

from mappo_evaluation import GlobalDijkstraPolicy, evaluate_policy_with_constraint_metrics
from run_development_load_diagnostics import make_wrapper, read_json, sha, write_json


def optimistic_delivery_bound(generated, final_slot_arrivals, hotspot_generated,
                              hotspot_final_slot, inbound_service_slots, total_service_slots):
    non_hotspot_eligible = generated - final_slot_arrivals - hotspot_generated + hotspot_final_slot
    hotspot_eligible = hotspot_generated - hotspot_final_slot
    return min(generated - final_slot_arrivals, total_service_slots,
               non_hotspot_eligible + min(hotspot_eligible, inbound_service_slots))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    completion = read_json(source / "completion.json")
    if completion["status"] != "complete":
        raise ValueError("a completed diagnostic run is required")
    for filename, expected in completion["output_sha256"].items():
        if sha(source / filename) != expected:
            raise ValueError(f"diagnostic artifact changed: {filename}")
    output = source / "service_capacity_audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = read_json(source / "manifest.json")
    episode_rows = [json.loads(line) for line in (source / "episodes.jsonl").read_text().splitlines()]
    identities = ("scenario", "load", "policy", "policy_seed", "workload_seed")
    episode_index = {tuple(row[k] for k in identities): row for row in episode_rows}
    counters = defaultdict(Counter)
    state_counts = defaultdict(Counter)
    hotspot_ids = {}
    for scenario, cfg in manifest["configs"].items():
        planes, per_plane = cfg["env"]["n_planes"], cfg["env"]["sats_per_plane"]
        hotspot_ids[scenario] = cfg["env"]["scenario"]["force_hotspot_dst"] or (((planes + 1) // 2 - 1) * per_plane + (per_plane + 1) // 2)
    seen_ids = set()
    with gzip.open(source / "packets.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            packet = json.loads(line)
            key = tuple(packet[k] for k in identities)
            identity = (*key, packet["packet_id"])
            if identity in seen_ids:
                raise AssertionError("duplicate packet record")
            seen_ids.add(identity)
            counts = counters[key]
            counts["generated"] += 1
            counts[packet["status"]] += 1
            final_slot = manifest["configs"][packet["scenario"]]["episode_slots"]
            counts["final_slot_arrivals"] += packet["created_slot"] == final_slot
            if packet["dst"] == hotspot_ids[packet["scenario"]]:
                counts["hotspot_generated"] += 1
                counts["hotspot_final_slot"] += packet["created_slot"] == final_slot
                counts["hotspot_delivered"] += packet["status"] == "delivered"
            if packet["status"] == "dropped":
                counts["drop_" + packet["drop_reason"]] += 1
            state_counts[(packet["scenario"], packet["load"], packet["policy"])][packet["status"] + "_class_" + str(packet["traffic_class"])] += 1
    bounds = {}
    for key, counts in counters.items():
        row = episode_index[key]
        for status, field in (("generated", "generated"), ("delivered", "delivered"), ("dropped", "dropped"), ("queued", "backlog")):
            if counts[status] != row[field]:
                raise AssertionError(f"packet ledger does not match episode: {key} {status}")
        for field in (k for k in row if k.startswith("drop_") and k != "drop_rate"):
            if counts[field] != row[field]:
                raise AssertionError(f"drop reason mismatch: {key} {field}")
        scenario, load, _, _, seed = key
        physical_key = (scenario, load, seed)
        if physical_key not in bounds:
            wrapper = make_wrapper(scenario, load, seed, diagnostic=False)
            wrapper.reset(seed=seed)
            env, inbound = wrapper.env, 0
            for slot in range(1, env.cfg.episode_slots + 1):
                env.base.time_slot = slot
                graph = env.base._build_topology(slot)
                inbound += len({u for (u, v), edge in graph.items() if v == hotspot_ids[scenario] and edge.available})
            # One HOL decision per sender per slot, including destination reception.
            # Ignore queues, masks, path length, and contention: this is optimistic.
            upper = optimistic_delivery_bound(
                counts["generated"], counts["final_slot_arrivals"], counts["hotspot_generated"],
                counts["hotspot_final_slot"], inbound, env.n_agents * env.cfg.episode_slots,
            )
            bounds[physical_key] = {"scenario": scenario, "load": load, "workload_seed": seed,
                "hotspot": hotspot_ids[scenario], "generated": counts["generated"],
                "hotspot_generated": counts["hotspot_generated"], "inbound_service_slots": inbound,
                "optimistic_delivered_upper_bound": upper, "delivery_upper_bound": upper / counts["generated"]}
        if counts["delivered"] > bounds[physical_key]["optimistic_delivered_upper_bound"]:
            raise AssertionError("observed deliveries exceed the derived bound")
    probes = []
    for scenario in ("medium_load", "hotspot_high_load"):
        def factory(seed, service=3, capacity=3):
            wrapper = make_wrapper(scenario, 16, seed, diagnostic=False)
            wrapper.env.cfg.env.service_packets_per_slot = service
            wrapper.env.cfg.link_capacity_packets = capacity
            return wrapper
        reference = None
        for service, capacity in ((3, 3), (1, 3), (9, 3), (3, 1), (3, 9)):
            metric = asdict(evaluate_policy_with_constraint_metrics(
                scenario, "global_dijkstra", GlobalDijkstraPolicy(), -1, [910095],
                wrapper_factory=lambda seed: factory(seed, service, capacity),
            )[0])
            if reference is None:
                reference = metric
            probes.append({"scenario": scenario, "workload_seed": 910095,
                           "service_packets_per_slot": service, "link_capacity_packets": capacity,
                           "all_episode_metrics_equal_to_default": metric == reference})
    write_json(output, {
        "role": "development_service_capacity_audit", "packet_records_verified": len(seen_ids),
        "episodes_verified": len(counters), "source_completion_sha256": sha(source / "completion.json"),
        "auditor_sha256": sha(Path(__file__)),
        "bound_assumptions": "one HOL forwarding decision per satellite per slot; all non-hotspot pre-cutoff packets granted delivery; hotspot final-hop service limited by available incoming sender slots; ignore all other constraints",
        "bounds": list(bounds.values()), "configuration_probes": probes,
        "packet_classes": [{"scenario": k[0], "load": k[1], "policy": k[2], **v} for k, v in state_counts.items()],
        "total_blocked_proposals": sum(row["blocked_proposals"] for row in episode_rows),
    })
    print(json.dumps({"episodes_verified": len(counters), "packets_verified": len(seen_ids),
                      "all_configuration_probes_equal": all(p["all_episode_metrics_equal_to_default"] for p in probes),
                      "total_blocked_proposals": sum(row["blocked_proposals"] for row in episode_rows)}))
    for scenario in ("medium_load", "hotspot_high_load"):
        load = 6 if scenario == "medium_load" else 16
        rates = [b["delivery_upper_bound"] for b in bounds.values() if b["scenario"] == scenario and b["load"] == load]
        print(scenario, "load", load, "optimistic delivery upper bound mean/min/max", sum(rates) / len(rates), min(rates), max(rates))


if __name__ == "__main__":
    main()
