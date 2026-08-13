"""Versioned JSONL bridge between an ns-3 data plane and the MAPPO actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import TextIO

import numpy as np

from mappo_evaluation import load_checkpoint_policy


SCHEMA_VERSION = 1


class Ns3PolicyBridge:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.policy, metadata = load_checkpoint_policy(checkpoint, device=device)
        self.action_size = int(metadata["action_size"])
        self.feature_dim = int(metadata["candidate_feature_dim"])
        self.n_agents = int(metadata["n_agents"])

    def decide(self, message: dict) -> dict:
        if message.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {message.get('schema_version')}")
        if message.get("type") != "slot_state":
            raise ValueError("expected type=slot_state")
        agents = sorted(message.get("agents", []), key=lambda row: row["agent_id"])
        expected_ids = list(range(1, self.n_agents + 1))
        if [row.get("agent_id") for row in agents] != expected_ids:
            raise ValueError(f"agents must contain IDs 1..{self.n_agents} exactly once")

        observations = np.zeros(
            (self.n_agents, self.action_size * self.feature_dim), dtype=np.float32
        )
        masks = np.zeros((self.n_agents, self.action_size), dtype=bool)
        next_hops = []
        packet_ids = []
        for index, agent in enumerate(agents):
            features = np.asarray(agent.get("candidate_features"), dtype=np.float32)
            mask = np.asarray(agent.get("action_mask"), dtype=bool)
            candidates = list(agent.get("candidate_next_hops", []))
            if features.shape != (self.action_size, self.feature_dim):
                raise ValueError(
                    f"agent {index + 1} candidate_features has shape {features.shape}; "
                    f"expected {(self.action_size, self.feature_dim)}"
                )
            if mask.shape != (self.action_size,):
                raise ValueError(
                    f"agent {index + 1} action_mask has shape {mask.shape}; "
                    f"expected {(self.action_size,)}"
                )
            if len(candidates) != self.action_size:
                raise ValueError(
                    f"agent {index + 1} must provide {self.action_size} next-hop slots"
                )
            if not mask.any():
                raise ValueError(f"agent {index + 1} has no feasible action")
            observations[index] = features.reshape(-1)
            masks[index] = mask
            next_hops.append(candidates)
            packet_ids.append(agent.get("packet_id"))

        selected = self.policy(observations, masks)
        decisions = []
        for index, action in enumerate(selected):
            action = int(action)
            if not masks[index, action]:
                raise AssertionError("actor selected an action rejected by the ns-3 mask")
            decisions.append(
                {
                    "agent_id": index + 1,
                    "packet_id": packet_ids[index],
                    "action_slot": action,
                    "next_hop_id": int(next_hops[index][action]),
                }
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "slot_actions",
            "episode_id": message.get("episode_id"),
            "time_slot": message.get("time_slot"),
            "decisions": decisions,
        }

    def describe(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "bridge_ready",
            "n_agents": self.n_agents,
            "action_size": self.action_size,
            "candidate_feature_dim": self.feature_dim,
        }


def serve(bridge: Ns3PolicyBridge, source: TextIO, sink: TextIO) -> None:
    sink.write(json.dumps(bridge.describe(), separators=(",", ":")) + "\n")
    sink.flush()
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            if message.get("type") == "shutdown":
                return
            response = bridge.decide(message)
        except Exception as error:
            response = {
                "schema_version": SCHEMA_VERSION,
                "type": "bridge_error",
                "line_number": line_number,
                "error": str(error),
            }
        sink.write(json.dumps(response, separators=(",", ":")) + "\n")
        sink.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bridge = Ns3PolicyBridge(args.checkpoint, device=args.device)

    source = args.input.open("r", encoding="utf-8") if args.input else sys.stdin
    sink = args.output.open("w", encoding="utf-8", newline="\n") if args.output else sys.stdout
    try:
        serve(bridge, source, sink)
    finally:
        if args.input:
            source.close()
        if args.output:
            sink.close()


if __name__ == "__main__":
    main()
