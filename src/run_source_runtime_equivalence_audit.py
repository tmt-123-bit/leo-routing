"""Replay the frozen source/runtime equivalence contract for adaptive shield v8."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zipfile


AUDIT_NAME = "SOURCE-RUNTIME-EQUIVALENCE-v8"
SCHEMA_VERSION = 1
WORKLOAD_SEED = 60001
WORKLOAD_SEED_PROVENANCE = "replay_of_exposed_v7_validation_seed_60001"
POLICY_SEED = 1710210210
STAY_BONUS = 0.40
SCENARIOS = ("medium_load", "hotspot_high_load")
POLICY_KINDS = ("first_feasible", "checkpoint_beta_0p40")
MAX_STEPS = 40
EXPECTED_EPISODE_STEPS = 30

EXPECTED_ARCHIVE_SHA256 = (
    "4c5a418258979960d77d2be6228a5ad69823ae39c196a101b5192afceedacfd7"
)
EXPECTED_ARCHIVE_ENTRIES = {
    "HYSTERESIS_SCREEN_V1.md": (
        "49d330fade7dfe40d971f8085749655b065f6f4663ebaa12fab0275c3f79b082",
        3825,
    ),
    "ablation_matrix_runner.py": (
        "6afed086f213af1f1fc7b84b4a3744281e33da37286e93b9a322e5fd3f2968ad",
        121114,
    ),
    "cleanmarl_leo_multiagent_wrapper.py": (
        "7852124bd81b941a18565970ec3f60896995b55dedc27317ba5886490c1c1c31",
        8163,
    ),
    "hierarchical_statistics.py": (
        "350f12904bd7b1f29983c77ba7d0b057bfceae104b610fdc2d08b0a693d6dd56",
        23092,
    ),
    "hysteresis_policy.py": (
        "9c0be24bcb35e3aef0ce0722bd455e89e3f56608f3742419221de5594ebc8be3",
        9077,
    ),
    "leo_marl_env.py": (
        "9c226d7d6320be328ea2272c9b31492c391ff8622eaf05cc18f43b2922b40002",
        28548,
    ),
    "leo_multiagent_env.py": (
        "f8000fc1d25be69130458de98f302e263cfccc6c7e9bcfaa3374c2aadea3a1a5",
        43911,
    ),
    "mappo_design.py": (
        "40f2b484225cbe05d5321e723c88a64294c68624bf5947432d0c2606d2cf445a",
        10891,
    ),
    "mappo_evaluation.py": (
        "79359f3086322b80e6482716a5a28bce681b7427274e474661af263eed2dabdf",
        20457,
    ),
    "run_hysteresis_screen.py": (
        "2c2db354eb177463805cb8b36c06856f1f60a5819c906763700bf6a9c184992a",
        60301,
    ),
    "test_hysteresis_screen.py": (
        "1f2e0a6e116947d001c3fe1b9fb1456c3eb0196fca7c4029042ac33ee83b3bb1",
        11122,
    ),
    "variant_definitions.py": (
        "96025d3200d35e02b79f54cf3abb5805f0e11920281918780fc3e82cde526a35",
        7556,
    ),
}

EXPECTED_SOURCE_SPEC_SHA256 = (
    "ac63c0149fb68651173c920ccff0373ecd8b5dc6af92a377de09cee0be75d1ac"
)
EXPECTED_SOURCE_SPEC_FILE_SHA256 = (
    "69b24960c691c943f60f3c90c6ce7fd128d5c726ed92bf8f881c4dabf4cc48d6"
)
EXPECTED_SOURCE_CODE_FINGERPRINT = {
    "base_environment": "9c226d7d6320be328ea2272c9b31492c391ff8622eaf05cc18f43b2922b40002",
    "design": "40f2b484225cbe05d5321e723c88a64294c68624bf5947432d0c2606d2cf445a",
    "environment": "f8000fc1d25be69130458de98f302e263cfccc6c7e9bcfaa3374c2aadea3a1a5",
    "evaluation": "79359f3086322b80e6482716a5a28bce681b7427274e474661af263eed2dabdf",
    "executed_design": "40f2b484225cbe05d5321e723c88a64294c68624bf5947432d0c2606d2cf445a",
    "screen_artifact_helpers": "6afed086f213af1f1fc7b84b4a3744281e33da37286e93b9a322e5fd3f2968ad",
    "screen_protocol": "b7b155496f15532a971171b0fbc6667abd284286bc742231d9f92a601690a8d3",
    "screen_runner": "e23c45d156ba11fd427f89a5b0ddb32c79acdf03a4c2576c4b613e382403af15",
    "screen_statistics": "350f12904bd7b1f29983c77ba7d0b057bfceae104b610fdc2d08b0a693d6dd56",
    "trainer": "9c45a9bbb118969cbd330c99402756940558ad18a5a046ab6e45c1e213c38f2f",
    "trainer_snapshot": "9c45a9bbb118969cbd330c99402756940558ad18a5a046ab6e45c1e213c38f2f",
    "training_entry": "573b50a74293dd5a2440f8135c86eee34c2d775c3ae0cf579ad613cd5a335ca7",
    "variant_definitions": "96025d3200d35e02b79f54cf3abb5805f0e11920281918780fc3e82cde526a35",
    "wrapper": "7852124bd81b941a18565970ec3f60896995b55dedc27317ba5886490c1c1c31",
}
SOURCE_FINGERPRINT_ENTRY_MAP = {
    "base_environment": "leo_marl_env.py",
    "design": "mappo_design.py",
    "environment": "leo_multiagent_env.py",
    "evaluation": "mappo_evaluation.py",
    "executed_design": "mappo_design.py",
    "screen_artifact_helpers": "ablation_matrix_runner.py",
    "screen_statistics": "hierarchical_statistics.py",
    "variant_definitions": "variant_definitions.py",
    "wrapper": "cleanmarl_leo_multiagent_wrapper.py",
}

EXPECTED_HYSTERESIS_SPEC_SHA256 = (
    "c3d05253666b0a18ff23d8d6efeeea6bb94a920d613b240f1b598afec6852dc0"
)
EXPECTED_HYSTERESIS_SPEC_FILE_SHA256 = (
    "83dc6f09d473ac3ceba32013586ac5a2ee162b317b417f5ab5fff79e7277b4dc"
)
HYSTERESIS_FINGERPRINT_ENTRY_MAP = {
    "artifact_helpers": "ablation_matrix_runner.py",
    "base_environment": "leo_marl_env.py",
    "design": "mappo_design.py",
    "evaluation": "mappo_evaluation.py",
    "hysteresis_policy": "hysteresis_policy.py",
    "multiagent_environment": "leo_multiagent_env.py",
    "protocol": "HYSTERESIS_SCREEN_V1.md",
    "runner": "run_hysteresis_screen.py",
    "statistics": "hierarchical_statistics.py",
    "variants": "variant_definitions.py",
    "wrapper": "cleanmarl_leo_multiagent_wrapper.py",
}

EXPECTED_CHECKPOINTS = {
    "medium_load": {
        "relative_path": (
            "experiments/archive/congestion-context-screen-20k-v1/checkpoints/"
            "medium_load/proposed/seed_1710210210/"
            "leo_multi__medium_load__seed-1710210210__2026-09-01_08-28-50__"
            "CONGESTION-CONTEXT-SCREEN-v1/validation_best.pt"
        ),
        "sha256": "c76981c6b0dedae6006572f1ce951c6e7ef3af0c283893aaa3f3c511d18ef522",
        "length": 677647,
    },
    "hotspot_high_load": {
        "relative_path": (
            "experiments/archive/congestion-context-screen-20k-v1/checkpoints/"
            "hotspot_high_load/proposed/seed_1710210210/"
            "leo_multi__hotspot_high_load__seed-1710210210__2026-09-01_08-59-48__"
            "CONGESTION-CONTEXT-SCREEN-v1/validation_best.pt"
        ),
        "sha256": "babd29cea437f3e21a20c03674ccea8915accfecce4f7d613021def82cd7d141",
        "length": 677647,
    },
}

EXPECTED_RUNTIME_SIGNATURES = {
    "first_feasible": {
        "medium_load": {
            "digest": "bf96f04605101bda1dec704689c48e4538686f6d4534ef7eaa4e3627d2d8a62d",
            "final_obs_sha256": "2a2850b9725a1bbdc7539abf787b457764f27714eb8ff19d17826778a1951ca1",
            "final_mask_sha256": "2cf081347810ab37efb320783125b180cd6d81e6aa1ce267d99a40b083346ee3",
            "steps": EXPECTED_EPISODE_STEPS,
        },
        "hotspot_high_load": {
            "digest": "dd1eff739594ccd33a510053eb81dc74ad869b64b430085fb83e7b148c84a067",
            "final_obs_sha256": "91917d44a9ec2e797577edf7c2abaefa11ee4cfb7db8c2096074c8d7b1d59e50",
            "final_mask_sha256": "d7d476beaf071e0a0e2b29edeeb1db780e414ddfbc045aa077cbfce22125a0db",
            "steps": EXPECTED_EPISODE_STEPS,
        },
    },
    "checkpoint_beta_0p40": {
        "medium_load": {
            "digest": "7837f5665de453ce3babbce1b018c8f3f0c7b5b9e1248c21dc5a01e88512686b",
            "final_obs_sha256": "84d3f6375b4edfe3ddeeac23f9b0f225152d8ebf8000871beff142e1d4851fb7",
            "final_mask_sha256": "8b80cf4d0366c99b7ffc814cbbb73d2461071620e7379eff62c2ac7eec3203a8",
            "steps": EXPECTED_EPISODE_STEPS,
        },
        "hotspot_high_load": {
            "digest": "7cd9a12eb9acdc4fd8093ee047b2aeb9a9ce1ee0fefd76fdab2c867dab0cf4d0",
            "final_obs_sha256": "fd4fc5235bcfc15a5af4ac5a756ef51951cd89b2001e0bdab81c0a9b1b5ead2d",
            "final_mask_sha256": "cc53058c15f29b2c5a5c2626fd27a7959d09c179fef9510495dd5e9b51d026c0",
            "steps": EXPECTED_EPISODE_STEPS,
        },
    },
}

EXPECTED_FEATURE_NAMES = (
    "current_queue_ratio",
    "candidate_queue_ratio",
    "link_delay_ratio",
    "remaining_bandwidth_ratio",
    "link_load_rho",
    "link_reliability",
    "remaining_link_lifetime_ratio",
    "previous_contention_ratio",
    "destination_progress",
    "candidate_orbit_u",
    "candidate_orbit_w",
    "current_orbit_u",
    "current_orbit_w",
    "destination_orbit_u",
    "destination_orbit_w",
    "remaining_hop_ratio",
    "used_hop_ratio",
    "route_switch_indicator",
    "visited_satellite_ratio",
    "topology_time_phase",
    "deadline_normalized_waiting_age",
    "traffic_class_0",
    "traffic_class_1",
    "traffic_class_2",
    "candidate_already_visited",
    "candidate_is_previous_node",
)
EXPECTED_CURRENT_FEATURE_CONTRACT = {
    "action_size": 7,
    "candidate_feature_dim": 26,
    "feature_indexes": {
        "route_class_2": 23,
        "route_switch": 17,
        "route_urgency": 20,
    },
    "feature_names": list(EXPECTED_FEATURE_NAMES),
    "schema_id": "leo_multi_candidate_features_v1_dim_26",
    "schema_sha256": "be660bb34d6d8579773b643f2824e6dac5069fe67cfd8a8d71a36b1b147f9f70",
}

EXPECTED_ENV_EQUAL_METHODS = (
    "_active_agents",
    "_agent_observation",
    "_backlog_ids",
    "_candidate_features",
    "_continuation_headroom",
    "_create_packet",
    "_decay_load",
    "_drop_packet",
    "_expire_deadline_packets",
    "_is_route_switch",
    "_jain_index",
    "_mask_reason",
    "_queue_lengths",
    "_queue_trend",
    "_refresh_graph",
    "_remaining_bandwidth",
    "_remove_hol",
    "_resolve_link_capacity",
    "_sample_destination",
    "_sample_initial_pairs",
    "_waiting_ratio",
    "average_delivery_delay_slots",
    "from_scenario",
    "global_state",
    "observe",
    "state_digest",
    "trace_hash",
)
EXPECTED_ENV_CHANGED_METHODS = (
    "__init__",
    "_forward_reward",
    "_global_reward_components",
    "reset",
    "step",
    "validate_invariants",
)
EXPECTED_ENV_ADDED_METHODS = (
    "_route_switch_context",
    "_switch_cost_applies",
    "class_delivery_ratios",
)
EXPECTED_WRAPPER_EQUAL_METHODS = (
    "__init__",
    "_obs_array",
    "close",
    "get_action_size",
    "get_avail_actions",
    "get_candidate_feature_dim",
    "get_critic_spec",
    "get_last_agent_rewards",
    "get_obs_size",
    "get_policy_active_mask",
    "get_state",
    "get_state_size",
    "get_variant_spec",
    "reset",
    "step",
)
EXPECTED_WRAPPER_ADDED_METHODS = (
    "get_candidate_feature_schema",
    "get_route_class_2_feature_index",
    "get_route_switch_feature_index",
    "get_route_urgency_feature_index",
)

CORE_INFO_FIELDS = (
    "delivered",
    "dropped",
    "generated",
    "delivery_ratio",
    "drop_rate",
    "average_delay_slots",
    "routing_switches",
    "global_reward",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_self_hash(
    value: Mapping[str, Any], field: str, expected: str
) -> None:
    body = dict(value)
    observed = body.pop(field, None)
    computed = sha256_json(body)
    if observed != computed or observed != expected:
        raise ValueError(
            f"{field} mismatch: observed={observed!r}, computed={computed}, "
            f"expected={expected}"
        )


def audit_archive(archive_path: Path) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    archive_sha = sha256_file(archive_path)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("source runtime archive SHA-256 mismatch")
    entries: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        files = [entry for entry in archive.infolist() if not entry.is_dir()]
        names = {entry.filename for entry in files}
        if names != set(EXPECTED_ARCHIVE_ENTRIES):
            raise ValueError("source runtime archive entry set mismatch")
        for entry in files:
            payload = archive.read(entry)
            observed_sha = hashlib.sha256(payload).hexdigest()
            expected_sha, expected_length = EXPECTED_ARCHIVE_ENTRIES[
                entry.filename
            ]
            if observed_sha != expected_sha or len(payload) != expected_length:
                raise ValueError(
                    f"source runtime archive entry mismatch: {entry.filename}"
                )
            entries[entry.filename] = {
                "sha256": observed_sha,
                "length": len(payload),
                "compressed_length": entry.compress_size,
            }
    return {
        "path": str(archive_path),
        "sha256": archive_sha,
        "length": archive_path.stat().st_size,
        "entries": {name: entries[name] for name in sorted(entries)},
    }


def _audit_spec(
    path: Path,
    *,
    expected_name: str,
    expected_file_sha256: str,
    expected_self_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    file_sha = sha256_file(path)
    if file_sha != expected_file_sha256:
        raise ValueError(f"frozen spec file SHA-256 mismatch: {path}")
    value = _load_json(path)
    if value.get("screen_name") != expected_name:
        raise ValueError(f"frozen spec screen mismatch: {path}")
    _validate_self_hash(value, "spec_sha256", expected_self_sha256)
    return value, {
        "path": str(path),
        "file_sha256": file_sha,
        "spec_sha256": expected_self_sha256,
    }


def audit_specs(
    source_spec_path: Path,
    hysteresis_spec_path: Path,
    archive_record: Mapping[str, Any],
) -> dict[str, Any]:
    source_spec, source_record = _audit_spec(
        source_spec_path,
        expected_name="CONGESTION-CONTEXT-SCREEN-v1",
        expected_file_sha256=EXPECTED_SOURCE_SPEC_FILE_SHA256,
        expected_self_sha256=EXPECTED_SOURCE_SPEC_SHA256,
    )
    source_fingerprint = source_spec.get("code_fingerprint")
    if source_fingerprint != EXPECTED_SOURCE_CODE_FINGERPRINT:
        raise ValueError("source screen code fingerprint mismatch")
    entries = archive_record["entries"]
    for field, entry in SOURCE_FINGERPRINT_ENTRY_MAP.items():
        if source_fingerprint[field] != entries[entry]["sha256"]:
            raise ValueError(
                f"source screen fingerprint is not bound to archive entry: {field}"
            )

    hysteresis_spec, hysteresis_record = _audit_spec(
        hysteresis_spec_path,
        expected_name="ACTOR-SCORE-HYSTERESIS-SCREEN-v1",
        expected_file_sha256=EXPECTED_HYSTERESIS_SPEC_FILE_SHA256,
        expected_self_sha256=EXPECTED_HYSTERESIS_SPEC_SHA256,
    )
    hysteresis_fingerprint = hysteresis_spec.get("code_fingerprint")
    expected_hysteresis_fingerprint = {
        field: entries[entry]["sha256"]
        for field, entry in HYSTERESIS_FINGERPRINT_ENTRY_MAP.items()
    }
    if hysteresis_fingerprint != expected_hysteresis_fingerprint:
        raise ValueError("hysteresis screen/archive fingerprint mismatch")
    if (
        hysteresis_spec.get("source", {}).get("spec_sha256")
        != EXPECTED_SOURCE_SPEC_SHA256
    ):
        raise ValueError("hysteresis screen is not bound to the source screen")
    source_record["code_fingerprint"] = dict(source_fingerprint)
    hysteresis_record["code_fingerprint"] = dict(hysteresis_fingerprint)
    return {
        "source_screen": source_record,
        "hysteresis_screen": hysteresis_record,
    }


def _class_methods(source: str, class_name: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise ValueError(f"expected exactly one class named {class_name}")
    return {
        node.name: node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _ast_sha256(node: ast.AST) -> str:
    encoded = ast.dump(node, include_attributes=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _method_contract(
    snapshot_source: str,
    current_source: str,
    *,
    class_name: str,
    expected_equal: Sequence[str],
    expected_changed: Sequence[str],
    expected_added: Sequence[str],
) -> dict[str, Any]:
    snapshot = _class_methods(snapshot_source, class_name)
    current = _class_methods(current_source, class_name)
    common = set(snapshot) & set(current)
    equal = {
        name
        for name in common
        if ast.dump(snapshot[name], include_attributes=False)
        == ast.dump(current[name], include_attributes=False)
    }
    changed = common - equal
    added = set(current) - set(snapshot)
    removed = set(snapshot) - set(current)
    if equal != set(expected_equal):
        raise ValueError(f"{class_name} equal-method contract drifted")
    if changed != set(expected_changed):
        raise ValueError(f"{class_name} changed-method contract drifted")
    if added != set(expected_added):
        raise ValueError(f"{class_name} added-method contract drifted")
    if removed:
        raise ValueError(f"{class_name} removed methods: {sorted(removed)}")
    method_hashes = {
        name: {
            "snapshot_ast_sha256": (
                _ast_sha256(snapshot[name]) if name in snapshot else None
            ),
            "current_ast_sha256": (
                _ast_sha256(current[name]) if name in current else None
            ),
            "relation": (
                "equal"
                if name in equal
                else "changed"
                if name in changed
                else "added"
            ),
        }
        for name in sorted(set(snapshot) | set(current))
    }
    return {
        "class_name": class_name,
        "equal_methods": sorted(equal),
        "changed_methods": sorted(changed),
        "added_methods": sorted(added),
        "removed_methods": [],
        "method_ast_sha256": method_hashes,
    }


def audit_ast_contract(archive_path: Path, current_src: Path) -> dict[str, Any]:
    current_src = current_src.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        snapshot_environment = archive.read("leo_multiagent_env.py").decode(
            "utf-8"
        )
        snapshot_wrapper = archive.read(
            "cleanmarl_leo_multiagent_wrapper.py"
        ).decode("utf-8")
    current_environment_path = current_src / "leo_multiagent_env.py"
    current_wrapper_path = current_src / "cleanmarl_leo_multiagent_wrapper.py"
    current_environment = current_environment_path.read_text(encoding="utf-8")
    current_wrapper = current_wrapper_path.read_text(encoding="utf-8")
    return {
        "environment": _method_contract(
            snapshot_environment,
            current_environment,
            class_name="SynchronousLeoMultiAgentEnv",
            expected_equal=EXPECTED_ENV_EQUAL_METHODS,
            expected_changed=EXPECTED_ENV_CHANGED_METHODS,
            expected_added=EXPECTED_ENV_ADDED_METHODS,
        ),
        "wrapper": _method_contract(
            snapshot_wrapper,
            current_wrapper,
            class_name="CleanMARLLeoMultiAgentWrapper",
            expected_equal=EXPECTED_WRAPPER_EQUAL_METHODS,
            expected_changed=(),
            expected_added=EXPECTED_WRAPPER_ADDED_METHODS,
        ),
        "current_files": {
            "leo_multiagent_env.py": {
                "path": str(current_environment_path.resolve()),
                "sha256": sha256_file(current_environment_path),
            },
            "cleanmarl_leo_multiagent_wrapper.py": {
                "path": str(current_wrapper_path.resolve()),
                "sha256": sha256_file(current_wrapper_path),
            },
        },
    }


def audit_checkpoints(project_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for scenario, expected in EXPECTED_CHECKPOINTS.items():
        path = (project_root / expected["relative_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_sha = sha256_file(path)
        observed_length = path.stat().st_size
        if (
            observed_sha != expected["sha256"]
            or observed_length != expected["length"]
        ):
            raise ValueError(f"frozen checkpoint mismatch: {scenario}")
        result[scenario] = {
            "path": str(path),
            "sha256": observed_sha,
            "length": observed_length,
            "policy_seed": POLICY_SEED,
            "stay_bonus": STAY_BONUS,
        }
    return result


def _worker_runtime(
    runtime: str,
    policy_kind: str,
    project_root: Path,
    archive_path: Path,
) -> dict[str, Any]:
    if runtime not in {"snapshot", "current"}:
        raise ValueError(f"unknown worker runtime: {runtime}")
    if policy_kind not in POLICY_KINDS:
        raise ValueError(f"unknown worker policy: {policy_kind}")
    sys.dont_write_bytecode = True
    import_root = archive_path if runtime == "snapshot" else project_root / "src"
    sys.path.insert(0, str(import_root.resolve()))

    import numpy as np

    wrapper_module = importlib.import_module("cleanmarl_leo_multiagent_wrapper")
    environment_module = importlib.import_module("leo_multiagent_env")
    wrapper_class = wrapper_module.CleanMARLLeoMultiAgentWrapper
    load_policy = None
    policy_module = None
    if policy_kind == "checkpoint_beta_0p40":
        policy_module = importlib.import_module("hysteresis_policy")
        load_policy = policy_module.load_hysteresis_policy

    results: dict[str, dict[str, Any]] = {}
    checkpoint_metadata: dict[str, dict[str, Any]] = {}
    feature_contract: dict[str, Any] | None = None
    for scenario in SCENARIOS:
        wrapper = wrapper_class(
            scenario,
            seed=WORKLOAD_SEED,
            variant="proposed",
        )
        observation, _ = wrapper.reset(seed=WORKLOAD_SEED)
        policy = None
        if load_policy is not None:
            checkpoint = EXPECTED_CHECKPOINTS[scenario]
            checkpoint_path = (
                project_root / checkpoint["relative_path"]
            ).resolve()
            policy, payload = load_policy(
                checkpoint_path,
                stay_bonus=STAY_BONUS,
                device="cpu",
            )
            checkpoint_metadata[scenario] = {
                "candidate_actor_spec": payload.get("candidate_actor_spec"),
                "candidate_feature_dim": int(payload["candidate_feature_dim"]),
                "action_size": int(payload["action_size"]),
                "obs_size": int(payload["obs_size"]),
                "n_agents": int(payload["n_agents"]),
                "variant": str(payload["args"]["leo_variant"]),
            }

        if runtime == "current" and feature_contract is None:
            schema = wrapper.get_candidate_feature_schema()
            feature_contract = {
                "action_size": int(wrapper.get_action_size()),
                "candidate_feature_dim": int(
                    wrapper.get_candidate_feature_dim()
                ),
                "feature_indexes": {
                    "route_class_2": int(
                        wrapper.get_route_class_2_feature_index()
                    ),
                    "route_switch": int(
                        wrapper.get_route_switch_feature_index()
                    ),
                    "route_urgency": int(
                        wrapper.get_route_urgency_feature_index()
                    ),
                },
                "feature_names": list(schema["feature_names"]),
                "schema_id": str(schema["schema_id"]),
                "schema_sha256": str(schema["sha256"]),
            }

        digest = hashlib.sha256()
        final_info: Mapping[str, Any] | None = None
        for step in range(MAX_STEPS):
            action_mask = wrapper.get_avail_actions()
            if policy is None:
                actions = np.asarray(
                    [
                        np.flatnonzero(agent_mask)[0]
                        for agent_mask in action_mask
                    ],
                    dtype=np.int64,
                )
            else:
                actions = policy(observation, action_mask)
            digest.update(observation.tobytes())
            digest.update(action_mask.tobytes())
            digest.update(actions.tobytes())
            observation, reward, done, truncated, final_info = wrapper.step(
                actions
            )
            digest.update(
                struct.pack(
                    "!d??",
                    float(reward),
                    bool(done),
                    bool(truncated),
                )
            )
            core = {
                field: final_info[field]
                for field in CORE_INFO_FIELDS
            }
            digest.update(canonical_json_bytes(core))
            if done or truncated:
                break
        if final_info is None:
            raise RuntimeError("runtime worker did not execute an environment step")
        results[scenario] = {
            "core": core,
            "digest": digest.hexdigest(),
            "final_mask_sha256": hashlib.sha256(
                wrapper.get_avail_actions().tobytes()
            ).hexdigest(),
            "final_obs_sha256": hashlib.sha256(
                observation.tobytes()
            ).hexdigest(),
            "policy_diagnostics": (
                policy.diagnostics() if policy is not None else None
            ),
            "steps": step + 1,
        }

    origins = {
        "cleanmarl_leo_multiagent_wrapper": str(
            Path(inspect.getfile(wrapper_module)).resolve()
        ),
        "leo_multiagent_env": str(
            Path(inspect.getfile(environment_module)).resolve()
        ),
    }
    if policy_module is not None:
        origins["hysteresis_policy"] = str(
            Path(inspect.getfile(policy_module)).resolve()
        )
    return {
        "runtime": runtime,
        "policy_kind": policy_kind,
        "module_origins": origins,
        "feature_contract": feature_contract,
        "checkpoint_metadata": checkpoint_metadata,
        "results": results,
    }


def _run_worker(
    runtime: str,
    policy_kind: str,
    project_root: Path,
    archive_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--worker-runtime",
        runtime,
        "--worker-policy",
        policy_kind,
        "--project",
        str(project_root.resolve()),
        "--archive",
        str(archive_path.resolve()),
    ]
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
        CUDA_VISIBLE_DEVICES="",
    )
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"equivalence worker failed ({runtime}/{policy_kind}): "
            f"{completed.stderr.strip()}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"equivalence worker returned invalid JSON: {runtime}/{policy_kind}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError("equivalence worker returned a non-object")
    return value


def audit_dynamic_contract(
    project_root: Path,
    archive_path: Path,
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    current_feature_contract: dict[str, Any] | None = None
    for policy_kind in POLICY_KINDS:
        snapshot = _run_worker(
            "snapshot", policy_kind, project_root, archive_path
        )
        current = _run_worker(
            "current", policy_kind, project_root, archive_path
        )
        if snapshot["results"] != current["results"]:
            raise RuntimeError(
                f"snapshot/current runtime drift: {policy_kind}"
            )
        if snapshot["feature_contract"] is not None:
            raise RuntimeError("snapshot unexpectedly claims a named feature schema")
        if current["feature_contract"] != EXPECTED_CURRENT_FEATURE_CONTRACT:
            raise RuntimeError("current candidate feature contract drifted")
        if current_feature_contract is None:
            current_feature_contract = current["feature_contract"]
        elif current_feature_contract != current["feature_contract"]:
            raise RuntimeError("worker feature contracts disagree")
        for scenario in SCENARIOS:
            observed = snapshot["results"][scenario]
            expected = EXPECTED_RUNTIME_SIGNATURES[policy_kind][scenario]
            for field, expected_value in expected.items():
                if observed.get(field) != expected_value:
                    raise RuntimeError(
                        f"runtime signature mismatch: "
                        f"{policy_kind}/{scenario}/{field}"
                    )
        if policy_kind == "checkpoint_beta_0p40":
            if snapshot["checkpoint_metadata"] != current["checkpoint_metadata"]:
                raise RuntimeError("snapshot/current checkpoint metadata drifted")
            for scenario in SCENARIOS:
                metadata = snapshot["checkpoint_metadata"][scenario]
                expected_metadata = {
                    "candidate_actor_spec": None,
                    "candidate_feature_dim": 26,
                    "action_size": 7,
                    "obs_size": 182,
                    "n_agents": 24,
                    "variant": "proposed",
                }
                if metadata != expected_metadata:
                    raise RuntimeError(
                        f"legacy checkpoint schema drifted: {scenario}"
                    )
        policies[policy_kind] = {
            "snapshot_module_origins": snapshot["module_origins"],
            "current_module_origins": current["module_origins"],
            "checkpoint_metadata": snapshot["checkpoint_metadata"],
            "results": snapshot["results"],
            "expected_signatures": EXPECTED_RUNTIME_SIGNATURES[policy_kind],
            "snapshot_current_exact_match": True,
        }
    if current_feature_contract is None:
        raise RuntimeError("dynamic audit did not observe a feature contract")
    return {
        "digest_algorithm": (
            "sha256_over_each_step_obs_mask_action_raw_bytes_then_"
            "network_order_float64_reward_done_truncated_then_canonical_core_json"
        ),
        "workload_seed": WORKLOAD_SEED,
        "workload_seed_provenance": WORKLOAD_SEED_PROVENANCE,
        "max_steps": MAX_STEPS,
        "expected_episode_steps": EXPECTED_EPISODE_STEPS,
        "feature_contract": current_feature_contract,
        "policies": policies,
    }


def _current_runtime_files(project_root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "HYSTERESIS_SCREEN_V1.md": project_root
        / "docs"
        / "HYSTERESIS_SCREEN_V1.md",
        **{
            name: project_root / "src" / name
            for name in EXPECTED_ARCHIVE_ENTRIES
            if name.endswith(".py")
        },
    }
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "length": path.stat().st_size,
        }
    return {name: result[name] for name in sorted(result)}


def build_equivalence_artifact(
    *,
    project_root: Path,
    archive_path: Path,
    source_spec_path: Path,
    hysteresis_spec_path: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    archive_path = archive_path.resolve()
    archive = audit_archive(archive_path)
    specs = audit_specs(source_spec_path, hysteresis_spec_path, archive)
    ast_contract = audit_ast_contract(archive_path, project_root / "src")
    checkpoints = audit_checkpoints(project_root)
    dynamic = audit_dynamic_contract(project_root, archive_path)
    script_path = Path(__file__).resolve()
    protocol_path = project_root / "docs" / "SOURCE_RUNTIME_EQUIVALENCE_V8.md"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit_name": AUDIT_NAME,
        "inferential_status": "runtime_semantic_equivalence_audit",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "new_evaluation_panel_accessed": False,
        "training_jobs": 0,
        "evaluation_jobs": 0,
        "archive": archive,
        "frozen_specs": specs,
        "current_runtime_files": _current_runtime_files(project_root),
        "ast_contract": ast_contract,
        "frozen_checkpoints": checkpoints,
        "dynamic_contract": dynamic,
        "audit_implementation": {
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
            "protocol_path": str(protocol_path.resolve()),
            "protocol_sha256": sha256_file(protocol_path),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
        },
        "decision": {
            "archive_contract_pass": True,
            "source_spec_contract_pass": True,
            "ast_contract_pass": True,
            "dynamic_contract_pass": True,
            "snapshot_current_exact_match": True,
            "all_pass": True,
        },
    }
    body["source_runtime_equivalence_sha256"] = sha256_json(body)
    return body


def validate_equivalence_artifact(value: Mapping[str, Any]) -> str:
    artifact = dict(value)
    observed = artifact.pop("source_runtime_equivalence_sha256", None)
    computed = sha256_json(artifact)
    if observed != computed:
        raise ValueError("source runtime equivalence self-hash mismatch")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("source runtime equivalence schema mismatch")
    if artifact.get("audit_name") != AUDIT_NAME:
        raise ValueError("source runtime equivalence audit name mismatch")
    if artifact.get("paper_claim_allowed") is not False:
        raise ValueError("equivalence artifact must prohibit paper claims")
    if artifact.get("promotion_decision_allowed") is not False:
        raise ValueError("equivalence artifact must prohibit promotion")
    if artifact.get("new_evaluation_panel_accessed") is not False:
        raise ValueError("equivalence artifact opened a new panel")
    decision = artifact.get("decision")
    if not isinstance(decision, Mapping) or decision.get("all_pass") is not True:
        raise ValueError("source runtime equivalence did not pass")
    return str(observed)


def load_and_replay_equivalence_artifact(
    artifact_path: Path,
    *,
    project_root: Path,
    archive_path: Path,
    source_spec_path: Path,
    hysteresis_spec_path: Path,
) -> dict[str, Any]:
    artifact_path = artifact_path.resolve()
    existing = _load_json(artifact_path)
    validate_equivalence_artifact(existing)
    replayed = build_equivalence_artifact(
        project_root=project_root,
        archive_path=archive_path,
        source_spec_path=source_spec_path,
        hysteresis_spec_path=hysteresis_spec_path,
    )
    if existing != replayed:
        raise ValueError("source runtime equivalence artifact does not replay")
    return existing


def ensure_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        if _load_json(path) != dict(value):
            raise RuntimeError(f"immutable artifact mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _defaults() -> dict[str, Path]:
    project_root = Path(__file__).resolve().parent.parent
    return {
        "project": project_root,
        "archive": project_root
        / "experiments"
        / "archive"
        / "source-snapshots"
        / "actor-score-hysteresis-c3d05253666b.zip",
        "source_spec": project_root
        / "experiments"
        / "archive"
        / "congestion-context-screen-20k-v1"
        / "screen_spec.json",
        "hysteresis_spec": project_root
        / "experiments"
        / "archive"
        / "congestion-hysteresis-screen-v1"
        / "hysteresis_spec.json",
        "output": project_root
        / "experiments"
        / "source-runtime-equivalence-v8"
        / "source_runtime_equivalence.json",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = _defaults()
    parser = argparse.ArgumentParser(
        description="Replay the frozen source/runtime equivalence audit."
    )
    parser.add_argument("--project", type=Path, default=defaults["project"])
    parser.add_argument("--archive", type=Path, default=defaults["archive"])
    parser.add_argument(
        "--source-spec", type=Path, default=defaults["source_spec"]
    )
    parser.add_argument(
        "--hysteresis-spec",
        type=Path,
        default=defaults["hysteresis_spec"],
    )
    parser.add_argument("--output", type=Path, default=defaults["output"])
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-runtime",
        choices=("snapshot", "current"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-policy", choices=POLICY_KINDS, help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args._worker:
        if args.worker_runtime is None or args.worker_policy is None:
            raise ValueError("worker runtime and policy are required")
        value = _worker_runtime(
            args.worker_runtime,
            args.worker_policy,
            args.project.resolve(),
            args.archive.resolve(),
        )
        print(canonical_json_bytes(value).decode("utf-8"))
        return 0

    parameters = {
        "project_root": args.project.resolve(),
        "archive_path": args.archive.resolve(),
        "source_spec_path": args.source_spec.resolve(),
        "hysteresis_spec_path": args.hysteresis_spec.resolve(),
    }
    if args.replay is not None:
        artifact = load_and_replay_equivalence_artifact(
            args.replay.resolve(), **parameters
        )
    else:
        artifact = build_equivalence_artifact(**parameters)
        if not args.dry_run:
            ensure_immutable_json(args.output.resolve(), artifact)
    summary = {
        "audit_name": AUDIT_NAME,
        "all_pass": artifact["decision"]["all_pass"],
        "new_evaluation_panel_accessed": False,
        "source_runtime_equivalence_sha256": artifact[
            "source_runtime_equivalence_sha256"
        ],
        "output": None if args.dry_run else str(args.output.resolve()),
        "replayed": args.replay is not None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
