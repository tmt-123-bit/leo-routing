# CMADR-style switch-budget-constrained routing for official Hypatia
# (fstate post-processor, delta-aware).
#
# CMADR's core mechanism (as operationalized in the slot-based environment)
# is a Lagrangian constraint that bounds avoidable route changes to a budget.
# In the official fstate world this maps to: each node may adopt at most
# BUDGET next-hop changes per forwarding-state interval; all other changed
# entries keep their previously installed next hop (links are static +Grid,
# so the kept hop is always still feasible).
import os
import shutil
import sys

BUDGET = 2   # allowed route changes per node per 100 ms interval (~12%-style cap)


def parse_line(line):
    parts = line.strip().split(",")
    if len(parts) != 5:
        return None
    return tuple(int(x) for x in parts)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for f in ("tles.txt", "isls.txt", "ground_stations.txt",
              "gsl_interfaces_info.txt", "description.txt"):
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))

    state_dir = os.path.join(src, "dynamic_state_100ms_for_10s")
    out_dir = os.path.join(dst, "dynamic_state_100ms_for_10s")
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(state_dir):
        if not fname.startswith("fstate_"):
            shutil.copyfile(os.path.join(state_dir, fname),
                            os.path.join(out_dir, fname))

    installed = {}   # node -> {target: (nh, my_if, next_if)}
    total_changed = total_kept = 0
    for fname in sorted(os.listdir(state_dir)):
        if not fname.startswith("fstate_"):
            continue
        entries = [p for p in (parse_line(l)
                               for l in open(os.path.join(state_dir, fname)))
                   if p is not None]
        budget_left = {}
        out_lines = []
        for node, tgt, nh, mi, ni in entries:
            old = installed.get(node, {}).get(tgt)
            if old is not None and old[0] != nh:
                if budget_left.get(node, BUDGET) <= 0:
                    out_lines.append(f"{node},{tgt},{old[0]},{old[1]},{old[2]}")
                    installed.setdefault(node, {})[tgt] = old
                    total_kept += 1
                    continue
                budget_left[node] = budget_left.get(node, BUDGET) - 1
            out_lines.append(f"{node},{tgt},{nh},{mi},{ni}")
            installed.setdefault(node, {})[tgt] = (nh, mi, ni)
            total_changed += 1
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write("\n".join(out_lines) + "\n")
    print(f"CMADR_POST_DONE adopted={total_changed} budget_kept={total_kept}")


if __name__ == "__main__":
    main()
