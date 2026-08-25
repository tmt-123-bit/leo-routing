"""Orchestrate the CLOSED-LOOP ns-3 validation (server on Windows, ns-3 in WSL).

Steps per policy (mappo, dijkstra):
  1. (once) copy src/ns3_closed_loop.cc into the WSL ns-3.48 scratch dir + build
  2. start ClosedLoopServer in a thread on this Windows host (the venv has torch)
  3. ./ns3 run scratch/leo-closed-loop --bridge-host=<windows-ip-as-seen-from-wsl>
     (NAT mode: the WSL default gateway IS the Windows host)
  4. parse the RESULT line; collect per-policy CSVs into
     experiments/ns3-closedloop-5ep/

Traffic is the SAME packets CSV as the static replay (medium_load, wl 21001-05),
so closed-loop results are directly comparable with replay @1x and with the
slot-env ground truth.
"""
from __future__ import annotations

import argparse
import glob
import re
import subprocess
import threading
import time
from pathlib import Path

from ns3_closed_loop_server import ClosedLoopServer

REPO = Path(__file__).resolve().parent.parent
WSL = ["wsl", "-d", "Ubuntu-22.04", "-u", "nsuser"]


def wsl_run(cmd: str, timeout=1800) -> str:
    proc = subprocess.run(WSL + ["--", "bash", "-lc", cmd],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"wsl command failed: {cmd}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def find_checkpoint(pattern: str) -> str:
    hits = glob.glob(pattern)
    assert hits, f"no checkpoint matches {pattern}"
    return sorted(hits)[-1]


def to_wsl_path(p: Path) -> str:
    s = str(p.resolve())
    drive, rest = s[0].lower(), s[2:].replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def parse_result(line: str) -> dict:
    out = {}
    for field in line.strip().split(","):
        if "=" in field:
            k, v = field.split("=", 1)
            try:
                out[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
            except ValueError:
                out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=(
        "experiments/train-main/checkpoints/medium_load/no_lifetime/seed_1024/"
        "*/validation_best.pt"))
    ap.add_argument("--scenario", default="medium_load")
    ap.add_argument("--packets", type=Path,
                    default=Path("experiments/ns3-replay/packets_mappo.csv"))
    ap.add_argument("--outdir", type=Path,
                    default=Path("experiments/ns3-closedloop-5ep"))
    ap.add_argument("--workload-seeds", default="21001,21002,21003,21004,21005")
    ap.add_argument("--policies", default="mappo,dijkstra")
    ap.add_argument("--port", type=int, default=7341)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.workload_seeds.split(",")]
    checkpoint = find_checkpoint(args.checkpoint)
    print(f"checkpoint: {checkpoint}")

    # 1) build the scratch program once
    if not args.skip_build:
        print("copying + building ns-3 scratch program ...", flush=True)
        wsl_run("cp /mnt/f/leo-routing-preliminary-matlab/src/ns3_closed_loop.cc "
                "~/ns-3.48/scratch/leo-closed-loop.cc")
        t0 = time.time()
        out = wsl_run("cd ~/ns-3.48 && ./ns3 build 2>&1 | tail -2")
        print(f"build done in {time.time()-t0:.0f}s: {out.strip()}", flush=True)

    # 2) Windows host IP as seen from WSL (NAT: the default gateway)
    route = wsl_run("ip route show default").strip()
    m = re.search(r"via\s+(\d+\.\d+\.\d+\.\d+)", route)
    assert m, f"cannot parse WSL default route: {route}"
    host_ip = m.group(1)
    print(f"bridge host (windows) = {host_ip}:{args.port}", flush=True)

    # env-clock params read from the actual cfg (must match the env exactly)
    from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
    from leo_marl_env import EnvConfig, SCENARIOS
    from leo_multiagent_env import MULTIAGENT_LOADS, MultiAgentConfig
    ini, exo = MULTIAGENT_LOADS[args.scenario]
    cfg = MultiAgentConfig(env=EnvConfig(scenario=SCENARIOS[args.scenario],
                                         seed=seeds[0]),
                           initial_packets=ini, exogenous_packets_per_slot=exo,
                           seed=seeds[0], variant="no_lifetime")
    w = CleanMARLLeoMultiAgentWrapper(cfg=cfg)
    w.reset(seed=seeds[0])
    env_cfg = w.env.cfg
    episode_slots = env_cfg.episode_slots
    deadlines = ",".join(str(d) for d in env_cfg.packet_class_deadlines)
    link_cap = env_cfg.link_capacity_packets
    node_q = env_cfg.max_queue_packets
    max_hops = env_cfg.env.max_local_hops
    w.close()
    print(f"env clock: episode_slots={episode_slots} deadlines={deadlines} "
          f"link_cap={link_cap} node_q={node_q} max_hops={max_hops}")

    server = ClosedLoopServer(Path(checkpoint), args.scenario, seeds)

    results = {}
    for policy in args.policies.split(","):
        out_csv = args.outdir / f"clpkt_{policy}.csv"
        wsl_out = to_wsl_path(out_csv)
        wsl_in = to_wsl_path(args.packets)
        port = server.listen(port=args.port)
        thread = threading.Thread(target=server.serve_once, daemon=True)
        thread.start()
        cmd = (f"cd ~/ns-3.48 && ./ns3 run 'scratch/leo-closed-loop "
               f"--input={wsl_in} --output={wsl_out} "
               f"--bridge-host={host_ip} --bridge-port={port} "
               f"--policy-name={policy} --episode-slots={episode_slots} "
               f"--deadline-slots={deadlines} --link-capacity={link_cap} "
               f"--node-qsize={node_q} --max-hops={max_hops}' 2>&1")
        print(f"[{policy}] running ns-3 ...", flush=True)
        t0 = time.time()
        proc = subprocess.run(WSL + ["--", "bash", "-lc", cmd],
                              capture_output=True, text=True, timeout=3600)
        thread.join(timeout=60)
        m = re.search(r"^RESULT,.*$", proc.stdout, re.M)
        if not m:
            print(proc.stdout[-3000:], proc.stderr[-2000:])
            raise RuntimeError(f"no RESULT line for {policy}")
        results[policy] = parse_result(m.group(0))
        results[policy]["wall_sec"] = round(time.time() - t0, 1)
        r = results[policy]
        print(f"[{policy}] sent={r['sent']} delivered={r['delivered']} "
              f"ratio={r['delivery_ratio']:.3f} p95={r['p95_delay_ms']}ms "
              f"deadline_drops={r['deadline_drops']} queue_drops={r['queue_drops']} "
              f"ttl_drops={r['ttl_drops']} blocked={r['blocked_by_link_capacity']} "
              f"({r['wall_sec']}s)", flush=True)

    # summary CSV
    keys = ["policy", "sent", "delivered", "delivery_ratio", "mean_delay_ms",
            "p50_delay_ms", "p95_delay_ms", "deadline_drops", "ttl_drops",
            "queue_drops", "source_drops", "device_queue_drops",
            "blocked_by_link_capacity", "holds", "decisions",
            "load_imbalance", "wall_sec"]
    with open(args.outdir / "closedloop_summary.csv", "w", encoding="utf-8-sig") as f:
        f.write(",".join(keys) + "\n")
        for policy, r in results.items():
            row = dict(r)
            row["policy"] = policy
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    print(f"=> wrote {args.outdir/'closedloop_summary.csv'}")


if __name__ == "__main__":
    main()
