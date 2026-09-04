"""Orchestrate the CLOSED-LOOP ns-3 validation (server on Windows, ns-3 in WSL).

Steps per policy (mappo, dijkstra):
  1. (once) freeze src/ns3_closed_loop.cc in the new output directory
  2. copy that frozen source into WSL, build, and bind its exact executable hash
  3. start ClosedLoopServer in a thread on this Windows host (the venv has torch)
  4. run the recorded executable --bridge-host=<windows-ip-as-seen-from-wsl>
     (NAT mode: the WSL default gateway IS the Windows host)
  5. validate per-packet terminal conservation against RESULT; collect outputs
     in a new audited directory that is never overwritten.

Traffic is the SAME packets CSV as the static replay (medium_load, wl 21001-05),
so closed-loop results are directly comparable with replay @1x and with the
slot-env ground truth.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from ns3_closed_loop_server import ClosedLoopServer

REPO = Path(__file__).resolve().parent.parent
WSL = ["wsl", "-d", "Ubuntu-22.04", "-u", "nsuser"]
WSL_NS3_ROOT = "/home/nsuser/ns-3.48"
WSL_NS3_SCRATCH_SOURCE = f"{WSL_NS3_ROOT}/scratch/leo-closed-loop.cc"
WSL_NS3_EXECUTABLE = (
    f"{WSL_NS3_ROOT}/build/scratch/ns3.48-leo-closed-loop-optimized"
)
GLOBAL_PACKET_ID_STRIDE = 1_000_000
N_NODES = 24
PACKET_TRACE_PREFIX = (
    "episode",
    "packet_id",
    "src",
    "dst",
    "traffic_class",
    "created_slot",
)


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


def extract_single_result(stdout: str, requested_policy: str) -> dict:
    lines = re.findall(r"^RESULT,.*$", stdout, re.M)
    if len(lines) != 1:
        raise ValueError(f"expected exactly one RESULT line; observed {len(lines)}")
    result = parse_result(lines[0])
    if result.get("policy") != requested_policy:
        raise ValueError(
            f"RESULT policy={result.get('policy')!r}; expected {requested_policy!r}"
        )
    if _result_count(result, "closed_loop") != 1:
        raise ValueError("RESULT closed_loop must equal 1")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packet_id_set_sha256(packet_ids) -> str:
    payload = json.dumps(
        sorted(set(int(packet_id) for packet_id in packet_ids)),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def require_file_sha256(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 changed: expected {expected}, observed {observed}"
        )
    return observed


def freeze_input_file(source: Path, destination: Path, expected_sha256: str) -> dict:
    """Copy one validated input exactly once and verify the copied bytes."""
    source = Path(source)
    destination = Path(destination)
    require_file_sha256(source, expected_sha256, "source input")
    digest = hashlib.sha256()
    with source.open("rb") as source_handle, destination.open("xb") as sink:
        for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
            digest.update(chunk)
            sink.write(chunk)
        sink.flush()
        os.fsync(sink.fileno())
    copied_sha256 = digest.hexdigest()
    if copied_sha256 != expected_sha256:
        raise ValueError(
            "source input changed while creating the frozen run copy"
        )
    require_file_sha256(destination, expected_sha256, "frozen input")
    return {
        "path": str(destination.resolve()),
        "sha256": copied_sha256,
    }


def wsl_sha256_file(path: str, label: str) -> str:
    """Hash one regular WSL file, rejecting missing or malformed output."""
    quoted_path = shlex.quote(path)
    command = f"test -f {quoted_path} && sha256sum -- {quoted_path}"
    try:
        output = wsl_run(command)
    except RuntimeError as error:
        raise FileNotFoundError(
            f"{label} is missing or unreadable in WSL: {path}"
        ) from error
    fields = output.strip().split(maxsplit=1)
    if not fields or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
        raise ValueError(f"cannot parse SHA-256 for {label}: {output!r}")
    return fields[0]


def require_wsl_file_sha256(path: str, expected: str, label: str) -> str:
    observed = wsl_sha256_file(path, label)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 changed: expected {expected}, observed {observed}"
        )
    return observed


def verify_ns3_program_provenance(program: dict, label: str) -> None:
    """Recheck every source/binary artifact bound by the run manifest."""
    frozen_source = program["frozen_source"]
    wsl_source = program["wsl_scratch_source"]
    executable = program["actual_executable"]
    if frozen_source["sha256"] != wsl_source["sha256"]:
        raise ValueError("ns-3 provenance source SHA-256 records disagree")
    require_file_sha256(
        Path(frozen_source["path"]),
        frozen_source["sha256"],
        f"{label} frozen ns-3 source",
    )
    require_wsl_file_sha256(
        wsl_source["path"],
        wsl_source["sha256"],
        f"{label} WSL scratch source",
    )
    require_wsl_file_sha256(
        executable["path"],
        executable["sha256"],
        f"{label} ns-3 executable",
    )


def prepare_ns3_program(
    frozen_source: Path,
    expected_source_sha256: str,
    *,
    skip_build: bool,
) -> dict:
    """Build or validate the exact WSL program used by this new run."""
    frozen_source = Path(frozen_source)
    require_file_sha256(
        frozen_source,
        expected_source_sha256,
        "frozen ns-3 source before build",
    )
    if skip_build:
        require_wsl_file_sha256(
            WSL_NS3_SCRATCH_SOURCE,
            expected_source_sha256,
            "existing WSL scratch source for --skip-build",
        )
    else:
        print("copying frozen source + building ns-3 scratch program ...", flush=True)
        source_in_wsl = to_wsl_path(frozen_source)
        wsl_run(
            f"cp -- {shlex.quote(source_in_wsl)} "
            f"{shlex.quote(WSL_NS3_SCRATCH_SOURCE)}"
        )
        require_wsl_file_sha256(
            WSL_NS3_SCRATCH_SOURCE,
            expected_source_sha256,
            "copied WSL scratch source before build",
        )
        t0 = time.time()
        build_output = wsl_run(
            f"cd {shlex.quote(WSL_NS3_ROOT)} && "
            "set -o pipefail && ./ns3 build 2>&1 | tail -2"
        )
        print(
            f"build done in {time.time()-t0:.0f}s: {build_output.strip()}",
            flush=True,
        )
        require_wsl_file_sha256(
            WSL_NS3_SCRATCH_SOURCE,
            expected_source_sha256,
            "WSL scratch source after build",
        )

    executable_sha256 = wsl_sha256_file(
        WSL_NS3_EXECUTABLE,
        "ns-3 optimized executable",
    )
    program = {
        "build_skipped": bool(skip_build),
        "frozen_source": {
            "path": str(frozen_source.resolve()),
            "sha256": expected_source_sha256,
        },
        "wsl_scratch_source": {
            "path": WSL_NS3_SCRATCH_SOURCE,
            "sha256": expected_source_sha256,
        },
        "actual_executable": {
            "path": WSL_NS3_EXECUTABLE,
            "sha256": executable_sha256,
        },
    }
    verify_ns3_program_provenance(program, "prepared")
    return program


def derive_traffic_manifest_path(packet_path: Path) -> Path:
    """Derive the exporter summary paired with a conventional packet trace."""
    packet_path = Path(packet_path)
    if packet_path.name.startswith("packets_"):
        suffix = packet_path.name[len("packets_") :]
        return packet_path.with_name(f"env_summary_{suffix}")
    if packet_path.name == "packets.csv":
        return packet_path.with_name("env_summary.csv")
    raise ValueError(
        "--traffic-manifest is required when --packets is not named "
        "packets.csv or packets_<suffix>.csv"
    )


def derive_packet_source_policy(packet_path: Path) -> str | None:
    name = Path(packet_path).name
    if name.startswith("packets_") and name.endswith(".csv"):
        policy = name[len("packets_") : -len(".csv")]
        return policy or None
    return None


def validate_traffic_manifest(
    path: Path,
    *,
    workload_seeds: list[int],
    initial_packets: int,
    exogenous_packets_per_slot: int,
    episode_slots: int,
    source_policy: str,
    source_scenario: str,
) -> dict:
    """Bind an exported packet trace to its ordered workload identities."""
    if (
        not workload_seeds
        or min(initial_packets, exogenous_packets_per_slot, episode_slots) < 0
        or episode_slots == 0
    ):
        raise ValueError("traffic manifest dimensions require seeds and a horizon")
    expected_generated = (
        initial_packets + exogenous_packets_per_slot * episode_slots
    )
    if not source_policy:
        raise ValueError("traffic source policy must be explicit")
    if not source_scenario:
        raise ValueError("traffic source scenario must be explicit")
    required = {"episode", "policy", "workload_seed", "generated"}
    rows = []
    observed_policies = []
    observed_scenarios = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(
                f"traffic manifest missing columns: {sorted(missing)}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append((
                    int(row["episode"]),
                    int(row["workload_seed"]),
                    int(row["generated"]),
                ))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid traffic manifest integer at line {line_number}"
                ) from error
            observed_policies.append(str(row["policy"]).strip())
            if "scenario" in fieldnames:
                observed_scenarios.append(str(row["scenario"]).strip())

    observed_episodes = [row[0] for row in rows]
    expected_episodes = list(range(len(workload_seeds)))
    if observed_episodes != expected_episodes:
        raise ValueError(
            "traffic manifest episodes must appear exactly as "
            f"{expected_episodes}; observed {observed_episodes}"
        )
    observed_seeds = [row[1] for row in rows]
    if observed_seeds != list(workload_seeds):
        raise ValueError(
            "traffic manifest workload_seed order does not match CLI: "
            f"expected {list(workload_seeds)}, observed {observed_seeds}"
        )
    for episode, _, generated in rows:
        if generated != expected_generated:
            raise ValueError(
                f"traffic manifest episode {episode} generated={generated}; "
                f"expected {expected_generated}"
            )
    unique_policies = set(observed_policies)
    if unique_policies != {source_policy}:
        raise ValueError(
            "traffic manifest policy does not match source policy: "
            f"expected {source_policy!r}, observed {sorted(unique_policies)}"
        )
    if observed_scenarios:
        unique_scenarios = set(observed_scenarios)
        if unique_scenarios != {source_scenario}:
            raise ValueError(
                "traffic manifest scenario does not match source scenario: "
                f"expected {source_scenario!r}, "
                f"observed {sorted(unique_scenarios)}"
            )
    return {
        "episodes": len(rows),
        "workload_seeds": observed_seeds,
        "generated_per_episode": expected_generated,
        "generated_total": expected_generated * len(rows),
        "source_policy": source_policy,
        "source_scenario": source_scenario,
        "manifest_policy_column": True,
        "manifest_scenario_column": bool(observed_scenarios),
        "sha256": sha256_file(path),
    }


TERMINAL_REASON_TO_RESULT = {
    "delivered": "delivered",
    "deadline_exceeded": "deadline_drops",
    "ttl_exceeded": "ttl_drops",
    "queue_overflow": "queue_drops",
    "source_queue_overflow": "source_drops",
    "device_queue_full": "device_queue_drops",
    "backlog": "truncated_backlog",
}


def _result_count(result: dict, field: str) -> int:
    if field not in result or isinstance(result[field], bool):
        raise ValueError(f"RESULT missing non-negative integer {field}")
    try:
        value = float(result[field])
    except (TypeError, ValueError) as error:
        raise ValueError(f"RESULT has invalid count {field}") from error
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        raise ValueError(f"RESULT has invalid count {field}={result[field]}")
    return int(value)


def _result_float(result: dict, field: str) -> float:
    if field not in result or isinstance(result[field], bool):
        raise ValueError(f"RESULT missing finite number {field}")
    try:
        value = float(result[field])
    except (TypeError, ValueError) as error:
        raise ValueError(f"RESULT has invalid number {field}") from error
    if not math.isfinite(value):
        raise ValueError(f"RESULT has invalid number {field}={result[field]}")
    return value


def validate_closed_loop_output(
    path: Path,
    result: dict,
    *,
    expected_sent: int | None = None,
    expected_packet_ids=None,
) -> dict:
    """Fail closed unless packet terminal states exactly conserve RESULT."""
    path = Path(path)
    audited_sha256 = sha256_file(path)
    required = {"packet_id", "delivered", "delay_ms", "drop_reason"}
    counts = {reason: 0 for reason in TERMINAL_REASON_TO_RESULT}
    packet_ids = set()
    delivered_delays = []
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"closed-loop output missing columns: {sorted(missing)}"
            )
        for line_number, row in enumerate(reader, start=2):
            rows += 1
            try:
                packet_id = int(row["packet_id"])
                delay_ms = float(row["delay_ms"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid closed-loop output value at line {line_number}"
                ) from error
            if packet_id <= 0 or packet_id in packet_ids:
                raise ValueError(
                    f"invalid or duplicate global packet_id {packet_id}"
                )
            packet_ids.add(packet_id)
            delivered_text = str(row["delivered"]).strip()
            if delivered_text not in {"0", "1"}:
                raise ValueError(
                    f"invalid delivered flag at line {line_number}: "
                    f"{delivered_text!r}"
                )
            delivered = delivered_text == "1"
            reason = str(row["drop_reason"]).strip()
            if reason not in TERMINAL_REASON_TO_RESULT:
                raise ValueError(
                    f"unknown terminal reason at line {line_number}: {reason!r}"
                )
            if delivered != (reason == "delivered"):
                raise ValueError(
                    f"delivered flag/reason mismatch at line {line_number}"
                )
            if not math.isfinite(delay_ms) or (delivered and delay_ms < 0):
                raise ValueError(f"invalid delivered delay at line {line_number}")
            if not delivered and delay_ms != -1.0:
                raise ValueError(
                    f"non-delivered packet delay must equal -1 at line {line_number}"
                )
            if delivered:
                delivered_delays.append(delay_ms)
            counts[reason] += 1

    sent = _result_count(result, "sent")
    if rows != sent:
        raise ValueError(f"packet output rows={rows}; RESULT sent={sent}")
    if expected_sent is not None and sent != expected_sent:
        raise ValueError(f"RESULT sent={sent}; trace declares {expected_sent}")
    if expected_packet_ids is not None:
        expected_ids = set(int(packet_id) for packet_id in expected_packet_ids)
        if packet_ids != expected_ids:
            missing = sorted(expected_ids - packet_ids)[:5]
            unexpected = sorted(packet_ids - expected_ids)[:5]
            raise ValueError(
                "global packet_id set mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
    result_terminal_total = 0
    for reason, field in TERMINAL_REASON_TO_RESULT.items():
        expected = _result_count(result, field)
        observed = counts[reason]
        if observed != expected:
            raise ValueError(
                f"terminal count mismatch for {reason}: "
                f"output={observed}, RESULT {field}={expected}"
            )
        result_terminal_total += expected
    if sent != result_terminal_total:
        raise ValueError(
            f"RESULT conservation failed: sent={sent}, "
            f"terminal_total={result_terminal_total}"
        )

    expected_ratio = counts["delivered"] / sent if sent else 0.0
    observed_ratio = _result_float(result, "delivery_ratio")
    if not math.isclose(
        observed_ratio,
        expected_ratio,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "RESULT delivery_ratio mismatch: "
            f"observed={observed_ratio}, expected={expected_ratio}"
        )

    delivered_delays.sort()
    if delivered_delays:
        recomputed_delays = {
            "mean_delay_ms": sum(delivered_delays) / len(delivered_delays),
            "p50_delay_ms": delivered_delays[
                int(0.50 * (len(delivered_delays) - 1))
            ],
            "p95_delay_ms": delivered_delays[
                int(0.95 * (len(delivered_delays) - 1))
            ],
        }
    else:
        recomputed_delays = {
            "mean_delay_ms": -1.0,
            "p50_delay_ms": -1.0,
            "p95_delay_ms": -1.0,
        }
    for field, expected in recomputed_delays.items():
        observed = _result_float(result, field)
        if not math.isclose(
            observed,
            expected,
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise ValueError(
                f"RESULT {field} mismatch: observed={observed}, expected={expected}"
            )
    require_file_sha256(path, audited_sha256, "audited packet output")
    return {
        "rows": rows,
        "unique_packet_ids": len(packet_ids),
        "sent": sent,
        "packet_id_set_sha256": packet_id_set_sha256(packet_ids),
        "sha256": audited_sha256,
        "terminal_counts": counts,
        "delay_metrics": recomputed_delays,
    }


def write_new_json_atomic(path: Path, payload: dict) -> None:
    """Publish complete JSON atomically while refusing an existing target."""
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_new_run_manifest(path: Path, payload: dict) -> None:
    """Create an immutable run manifest; never overwrite an existing run."""
    write_new_json_atomic(path, payload)


def write_completion_manifest(
    path: Path,
    *,
    run_manifest_path: Path,
    results: dict,
    packet_outputs: dict,
    packet_audits: dict,
    summary_path: Path,
    summary_sha256: str,
) -> dict:
    """Publish completion only after every requested artifact was audited."""
    run_manifest_path = Path(run_manifest_path)
    run_manifest_bytes = run_manifest_path.read_bytes()
    run_manifest_sha256 = hashlib.sha256(run_manifest_bytes).hexdigest()
    run_manifest = json.loads(run_manifest_bytes)
    if run_manifest.get("status") != "started":
        raise ValueError("run manifest is not in started state")
    requested_policies = run_manifest.get("policies")
    if (
        not isinstance(requested_policies, list)
        or not requested_policies
        or len(set(requested_policies)) != len(requested_policies)
    ):
        raise ValueError("run manifest has no valid requested policy set")
    policy_sets = (
        set(results),
        set(packet_outputs),
        set(packet_audits),
    )
    if (
        policy_sets[0] != set(requested_policies)
        or policy_sets[0] != policy_sets[1]
        or policy_sets[0] != policy_sets[2]
    ):
        raise ValueError("completion policies do not match audited outputs")
    traffic_manifest = run_manifest.get("traffic_manifest", {})
    manifest_traffic_source = {
        "policy": traffic_manifest.get("source_policy"),
        "scenario": traffic_manifest.get("source_scenario"),
    }
    traffic_source = run_manifest.get("traffic_source")
    if (
        not isinstance(traffic_source, dict)
        or not all(manifest_traffic_source.values())
        or traffic_source != manifest_traffic_source
    ):
        raise ValueError("run manifest has no complete traffic source identity")
    output_records = {}
    for policy in sorted(packet_outputs):
        output_path = Path(packet_outputs[policy])
        audited_sha256 = packet_audits[policy].get("sha256")
        if not audited_sha256:
            raise ValueError(f"packet audit for {policy} has no SHA-256")
        require_file_sha256(
            output_path,
            audited_sha256,
            f"audited packet output for {policy}",
        )
        output_records[policy] = {
            "path": str(output_path.resolve()),
            "sha256": audited_sha256,
            "audit": packet_audits[policy],
        }
    summary_path = Path(summary_path)
    require_file_sha256(summary_path, summary_sha256, "closed-loop summary")
    require_file_sha256(
        run_manifest_path,
        run_manifest_sha256,
        "validated run manifest",
    )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "run_manifest": {
            "path": str(run_manifest_path.resolve()),
            "sha256": run_manifest_sha256,
        },
        "traffic_source": traffic_source,
        "results": results,
        "packet_outputs": output_records,
        "summary": {
            "path": str(summary_path.resolve()),
            "sha256": summary_sha256,
        },
    }
    write_new_json_atomic(path, payload)
    return payload


def create_new_output_directory(path: Path) -> Path:
    """Create one fresh output directory and reject all reuse."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"refusing to reuse closed-loop output directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=False)
    return path


def validate_packet_trace_contract(
    path: Path,
    *,
    num_episodes: int,
    initial_packets: int,
    exogenous_packets_per_slot: int,
    episode_slots: int,
) -> dict:
    """Validate the exporter packet-ID/admission-phase contract."""
    if min(
        num_episodes,
        initial_packets,
        exogenous_packets_per_slot,
        episode_slots,
    ) < 0 or num_episodes == 0 or episode_slots == 0:
        raise ValueError("trace dimensions must be non-negative with a horizon")

    by_episode = {episode: {} for episode in range(num_episodes)}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        if fieldnames[: len(PACKET_TRACE_PREFIX)] != list(PACKET_TRACE_PREFIX):
            raise ValueError(
                "packet trace first six columns must be exactly "
                + ",".join(PACKET_TRACE_PREFIX)
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                episode = int(row["episode"])
                packet_id = int(row["packet_id"])
                src = int(row["src"])
                dst = int(row["dst"])
                traffic_class = int(row["traffic_class"])
                created_slot = int(row["created_slot"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid packet trace integer at line {line_number}"
                ) from error
            if episode not in by_episode:
                raise ValueError(f"episode outside configured range: {episode}")
            if packet_id <= 0 or packet_id in by_episode[episode]:
                raise ValueError(
                    f"invalid or duplicate packet_id {packet_id} in episode {episode}"
                )
            if not (1 <= src <= N_NODES) or not (1 <= dst <= N_NODES):
                raise ValueError(
                    f"packet trace node outside 1..{N_NODES} at line {line_number}"
                )
            if src == dst:
                raise ValueError(f"packet trace src equals dst at line {line_number}")
            if not (0 <= traffic_class <= 2):
                raise ValueError(
                    f"packet trace traffic_class outside 0..2 at line {line_number}"
                )
            by_episode[episode][packet_id] = created_slot

    expected_per_episode = (
        initial_packets + exogenous_packets_per_slot * episode_slots
    )
    if expected_per_episode >= GLOBAL_PACKET_ID_STRIDE:
        raise ValueError("per-episode packet IDs exceed global ID stride")
    for episode, observed in by_episode.items():
        expected_ids = set(range(1, expected_per_episode + 1))
        if set(observed) != expected_ids:
            raise ValueError(
                f"episode {episode} packet IDs do not match 1..{expected_per_episode}"
            )
        for packet_id, created_slot in observed.items():
            if packet_id <= initial_packets:
                expected_created_slot = 1
            else:
                if exogenous_packets_per_slot == 0:
                    raise ValueError("trace contains undeclared exogenous packets")
                expected_created_slot = (
                    (packet_id - initial_packets - 1)
                    // exogenous_packets_per_slot
                    + 1
                )
            if created_slot != expected_created_slot:
                raise ValueError(
                    f"episode {episode} packet {packet_id} has created_slot "
                    f"{created_slot}; expected {expected_created_slot}"
                )
    expected_global_packet_ids = [
        episode * GLOBAL_PACKET_ID_STRIDE + packet_id
        for episode in range(num_episodes)
        for packet_id in range(1, expected_per_episode + 1)
    ]
    return {
        "episodes": num_episodes,
        "packets": expected_per_episode * num_episodes,
        "initial_packets_per_episode": initial_packets,
        "exogenous_packets_per_slot": exogenous_packets_per_slot,
        "expected_global_packet_ids": expected_global_packet_ids,
        "expected_global_packet_ids_sha256": packet_id_set_sha256(
            expected_global_packet_ids
        ),
        "sha256": sha256_file(path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=(
        "experiments/train-main/checkpoints/medium_load/no_lifetime/seed_1024/"
        "*/validation_best.pt"))
    ap.add_argument("--scenario", default="medium_load")
    ap.add_argument("--packets", type=Path,
                    default=Path("experiments/ns3-replay/packets_mappo.csv"))
    ap.add_argument(
        "--traffic-manifest",
        type=Path,
        help=(
            "companion env_summary CSV binding episode workload seeds; "
            "defaults from a conventional --packets filename"
        ),
    )
    ap.add_argument(
        "--traffic-source-policy",
        help=(
            "policy identity that generated the traffic trace; required with "
            "an explicit --traffic-manifest"
        ),
    )
    ap.add_argument(
        "--traffic-source-scenario",
        default="medium_load",
        help=(
            "scenario identity that generated the traffic trace; must equal "
            "--scenario (default: medium_load)"
        ),
    )
    ap.add_argument("--outdir", type=Path,
                    default=Path("experiments/ns3-closedloop-audited-v2"))
    ap.add_argument("--workload-seeds", default="21001,21002,21003,21004,21005")
    ap.add_argument("--policies", default="mappo,dijkstra")
    ap.add_argument("--port", type=int, default=7341)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.workload_seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("--workload-seeds must contain at least one seed")
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("--policies must contain unique policy names")
    unsupported = set(policies) - {"mappo", "dijkstra"}
    if unsupported:
        raise ValueError(f"unsupported closed-loop policies: {sorted(unsupported)}")
    checkpoint = Path(find_checkpoint(args.checkpoint)).resolve()
    checkpoint_sha256 = sha256_file(checkpoint)
    print(f"checkpoint: {checkpoint} sha256={checkpoint_sha256}")
    inferred_source_policy = derive_packet_source_policy(args.packets)
    if args.traffic_manifest is None:
        traffic_manifest = derive_traffic_manifest_path(args.packets)
        traffic_manifest_source = "derived_from_packets"
        traffic_source_policy = (
            args.traffic_source_policy or inferred_source_policy
        )
    else:
        traffic_manifest = args.traffic_manifest
        traffic_manifest_source = "explicit_cli"
        if not args.traffic_source_policy:
            raise ValueError(
                "--traffic-source-policy is required with --traffic-manifest"
            )
        traffic_source_policy = args.traffic_source_policy
    if not traffic_source_policy:
        raise ValueError(
            "--traffic-source-policy is required for a non-conventional "
            "packet trace filename"
        )
    if (
        inferred_source_policy is not None
        and traffic_source_policy != inferred_source_policy
    ):
        raise ValueError(
            "traffic source policy does not match the --packets filename: "
            f"expected {inferred_source_policy!r}, "
            f"observed {traffic_source_policy!r}"
        )
    if args.traffic_source_scenario != args.scenario:
        raise ValueError(
            "traffic source scenario must match --scenario: "
            f"source={args.traffic_source_scenario!r}, "
            f"run={args.scenario!r}"
        )
    print(
        f"traffic manifest ({traffic_manifest_source}): "
        f"{traffic_manifest.resolve()} source_policy={traffic_source_policy} "
        f"source_scenario={args.traffic_source_scenario}"
    )

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
    trace_contract = validate_packet_trace_contract(
        args.packets,
        num_episodes=len(seeds),
        initial_packets=ini,
        exogenous_packets_per_slot=exo,
        episode_slots=episode_slots,
    )
    traffic_contract = validate_traffic_manifest(
        traffic_manifest,
        workload_seeds=seeds,
        initial_packets=ini,
        exogenous_packets_per_slot=exo,
        episode_slots=episode_slots,
        source_policy=traffic_source_policy,
        source_scenario=args.traffic_source_scenario,
    )
    if trace_contract["packets"] != traffic_contract["generated_total"]:
        raise ValueError(
            "packet trace count does not match traffic manifest generated total"
        )
    print(f"env clock: episode_slots={episode_slots} deadlines={deadlines} "
          f"link_cap={link_cap} node_q={node_q} max_hops={max_hops} "
          f"trace_packets={trace_contract['packets']}")
    print(
        f"packet trace sha256={trace_contract['sha256']} "
        f"traffic manifest sha256={traffic_contract['sha256']}"
    )

    create_new_output_directory(args.outdir)
    frozen_packets = args.outdir / "input_packets.csv"
    frozen_traffic_manifest = args.outdir / "traffic_manifest.csv"
    frozen_ns3_source = args.outdir / "ns3_closed_loop.cc"
    frozen_packet_record = freeze_input_file(
        args.packets,
        frozen_packets,
        trace_contract["sha256"],
    )
    frozen_traffic_record = freeze_input_file(
        traffic_manifest,
        frozen_traffic_manifest,
        traffic_contract["sha256"],
    )
    repository_ns3_source = REPO / "src" / "ns3_closed_loop.cc"
    repository_ns3_source_sha256 = sha256_file(repository_ns3_source)
    frozen_ns3_source_record = freeze_input_file(
        repository_ns3_source,
        frozen_ns3_source,
        repository_ns3_source_sha256,
    )
    ns3_program = prepare_ns3_program(
        frozen_ns3_source,
        frozen_ns3_source_record["sha256"],
        skip_build=args.skip_build,
    )
    run_manifest_path = args.outdir / "run_manifest.json"
    run_manifest = {
        "schema_version": 1,
        "status": "started",
        "run_type": "ns3_closed_loop_audited_v2",
        "scenario": args.scenario,
        "traffic_source": {
            "policy": traffic_source_policy,
            "scenario": args.traffic_source_scenario,
        },
        "policies": policies,
        "workload_seeds": seeds,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
        },
        "output_directory": str(args.outdir.resolve()),
        "packet_trace": {
            "path": str(args.packets.resolve()),
            "frozen_copy": frozen_packet_record,
            **trace_contract,
        },
        "traffic_manifest": {
            "path": str(traffic_manifest.resolve()),
            "selection": traffic_manifest_source,
            "frozen_copy": frozen_traffic_record,
            **traffic_contract,
        },
        "env_contract": {
            "episode_slots": episode_slots,
            "initial_packets": ini,
            "exogenous_packets_per_slot": exo,
            "packet_class_deadlines": list(env_cfg.packet_class_deadlines),
            "link_capacity_packets": link_cap,
            "max_queue_packets": node_q,
            "max_local_hops": max_hops,
        },
        "closed_loop_program": ns3_program,
    }
    write_new_run_manifest(run_manifest_path, run_manifest)
    print(f"run manifest: {run_manifest_path.resolve()}", flush=True)

    # Windows host IP as seen from WSL (NAT: the default gateway)
    route = wsl_run("ip route show default").strip()
    m = re.search(r"via\s+(\d+\.\d+\.\d+\.\d+)", route)
    assert m, f"cannot parse WSL default route: {route}"
    host_ip = m.group(1)
    print(f"bridge host (windows) = {host_ip}:{args.port}", flush=True)

    results = {}
    result_records = {}
    packet_outputs = {}
    packet_audits = {}
    for policy in policies:
        verify_ns3_program_provenance(ns3_program, f"{policy} pre-run")
        require_file_sha256(checkpoint, checkpoint_sha256, "checkpoint")
        require_file_sha256(
            frozen_packets,
            trace_contract["sha256"],
            "frozen packet trace",
        )
        require_file_sha256(
            frozen_traffic_manifest,
            traffic_contract["sha256"],
            "frozen traffic manifest",
        )
        # A fresh server gives every policy its own bridge and episode feature
        # contexts, so no route cache or decayed load can cross policy runs.
        server = ClosedLoopServer(checkpoint, args.scenario, seeds)
        out_csv = args.outdir / f"clpkt_{policy}.csv"
        wsl_out = to_wsl_path(out_csv)
        wsl_in = to_wsl_path(frozen_packets)
        port = server.listen(port=args.port)
        thread = threading.Thread(target=server.serve_once, daemon=True)
        thread.start()
        ns3_arguments = [
            WSL_NS3_EXECUTABLE,
            f"--input={wsl_in}",
            f"--output={wsl_out}",
            f"--bridge-host={host_ip}",
            f"--bridge-port={port}",
            f"--policy-name={policy}",
            f"--episode-slots={episode_slots}",
            f"--num-episodes={len(seeds)}",
            f"--initial-packets={ini}",
            f"--deadline-slots={deadlines}",
            f"--link-capacity={link_cap}",
            f"--node-qsize={node_q}",
            f"--max-hops={max_hops}",
        ]
        cmd = f"cd {shlex.quote(WSL_NS3_ROOT)} && "
        cmd += " ".join(shlex.quote(argument) for argument in ns3_arguments)
        cmd += " 2>&1"
        print(f"[{policy}] running ns-3 ...", flush=True)
        t0 = time.time()
        try:
            proc = subprocess.run(
                WSL + ["--", "bash", "-lc", cmd],
                capture_output=True,
                text=True,
                timeout=3600,
            )
        finally:
            thread.join(timeout=60)
            verify_ns3_program_provenance(ns3_program, f"{policy} post-run")
        if thread.is_alive():
            raise RuntimeError(f"closed-loop server did not finish for {policy}")
        if proc.returncode != 0:
            print(proc.stdout[-3000:], proc.stderr[-2000:])
            raise RuntimeError(f"ns-3 closed-loop run failed for {policy}")
        try:
            result = extract_single_result(proc.stdout, policy)
        except ValueError:
            print(proc.stdout[-3000:], proc.stderr[-2000:])
            raise
        require_file_sha256(checkpoint, checkpoint_sha256, "checkpoint")
        require_file_sha256(
            frozen_packets,
            trace_contract["sha256"],
            "frozen packet trace",
        )
        require_file_sha256(
            frozen_traffic_manifest,
            traffic_contract["sha256"],
            "frozen traffic manifest",
        )
        result_records[policy] = dict(result)
        results[policy] = dict(result)
        packet_audit = validate_closed_loop_output(
            out_csv,
            results[policy],
            expected_sent=trace_contract["packets"],
            expected_packet_ids=trace_contract["expected_global_packet_ids"],
        )
        packet_outputs[policy] = out_csv
        packet_audits[policy] = packet_audit
        results[policy]["wall_sec"] = round(time.time() - t0, 1)
        r = results[policy]
        print(f"[{policy}] sent={r['sent']} delivered={r['delivered']} "
              f"ratio={r['delivery_ratio']:.3f} p95={r['p95_delay_ms']}ms "
              f"deadline_drops={r['deadline_drops']} queue_drops={r['queue_drops']} "
              f"ttl_drops={r['ttl_drops']} backlog={r['truncated_backlog']} "
              f"blocked={r['blocked_by_link_capacity']} "
              f"audited_rows={packet_audit['rows']} "
              f"({r['wall_sec']}s)", flush=True)

    # summary CSV
    keys = ["policy", "sent", "delivered", "delivery_ratio", "mean_delay_ms",
            "p50_delay_ms", "p95_delay_ms", "deadline_drops", "ttl_drops",
            "queue_drops", "source_drops", "device_queue_drops",
            "truncated_backlog", "blocked_by_link_capacity", "holds", "decisions",
            "load_imbalance", "wall_sec"]
    summary_path = args.outdir / "closedloop_summary.csv"
    with summary_path.open("x", encoding="utf-8-sig", newline="\n") as f:
        f.write(",".join(keys) + "\n")
        for policy, r in results.items():
            row = dict(r)
            row["policy"] = policy
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    summary_sha256 = sha256_file(summary_path)
    completion_path = args.outdir / "completion_manifest.json"
    require_file_sha256(checkpoint, checkpoint_sha256, "checkpoint")
    require_file_sha256(
        frozen_packets,
        trace_contract["sha256"],
        "frozen packet trace",
    )
    require_file_sha256(
        frozen_traffic_manifest,
        traffic_contract["sha256"],
        "frozen traffic manifest",
    )
    verify_ns3_program_provenance(ns3_program, "completion")
    write_completion_manifest(
        completion_path,
        run_manifest_path=run_manifest_path,
        results=result_records,
        packet_outputs=packet_outputs,
        packet_audits=packet_audits,
        summary_path=summary_path,
        summary_sha256=summary_sha256,
    )
    print(f"=> wrote {summary_path}")
    print(f"=> completed {completion_path}")


if __name__ == "__main__":
    main()
