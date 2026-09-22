"""Amendment-1 stage: preregistered bp_base/bp_mech one-shot comparison on
workloads 880351..880400 (docs/..._AMENDMENT_1.md, committed before this
panel existed). Statistics identical to the base formal study."""

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from backpressure_baseline import BackpressurePolicy
from hypatia_topology_provider_stub import HypatiaTopologyProvider
from run_tle66_formal import W66, bootstrap_ratio, sign_test  # noqa: F401
from mappo_evaluation import evaluate_policy_with_constraint_metrics
from run_development_load_diagnostics import canonical, sha, write_json

WORKLOADS = list(range(880351, 880401))
SCENARIO = "hotspot_high_load"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    topology = Path("../data/starlink_66_links.csv")
    W66._provider = HypatiaTopologyProvider.from_csv(topology)
    write_json(output / "manifest.json", {
        "role": "tle66_formal_amendment1",
        "runner_sha256": sha(Path(__file__)),
        "amendment": "docs/TLE66_PACKET_MANAGEMENT_FORMAL_V1_AMENDMENT_1.md",
        "amendment_sha256": sha(Path("../docs/TLE66_PACKET_MANAGEMENT_FORMAL_V1_AMENDMENT_1.md")),
        "topology_sha256": sha(topology),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workloads": WORKLOADS, "load": 12, "gamma": 4.0,
        "training": False, "sealed_test_access": False,
    })
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = {"base": [], "mech": []}
    started = time.monotonic()
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for arm, key in (("base", "base"), ("purge_srpf", "mech")):
            policy = BackpressurePolicy(4.0)
            for w in WORKLOADS:
                wrapper = W66(w, 12, arm)
                result = evaluate_policy_with_constraint_metrics(
                    SCENARIO, f"bp_{key}", policy, -1, [w],
                    wrapper_factory=lambda _: wrapper, variant="qos_only",
                )[0]
                row = asdict(result)
                row["arm"] = arm
                stream.write(canonical(row) + "\n")
                rows[key].append(row["delivery_ratio"])
            stream.flush()
            print(f"{key}: mean={np.mean(rows[key]):.4f} elapsed={time.monotonic()-started:.0f}s", flush=True)
    base = np.array(rows["base"])
    mech = np.array(rows["mech"])
    result = {
        "episodes": len(rows["base"]) * 2,
        "bp_base_mean": float(base.mean()),
        "bp_mech_mean": float(mech.mean()),
        "bootstrap": bootstrap_ratio(base, mech),
        "paired_gain_positive": int((mech > base).sum()),
        "n": len(base),
        "sign_test": sign_test(mech - base),
    }
    result["primary_pass"] = bool(
        result["bootstrap"]["one_sided_95_lower"] >= 0.10
        and result["paired_gain_positive"] >= 48
    )
    write_json(output / "result.json", result)
    write_json(output / "completion.json", {"status": "complete",
               "elapsed_seconds": time.monotonic() - started,
               "primary_pass": result["primary_pass"],
               "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
