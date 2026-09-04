"""Deterministic actor-score hysteresis for cached LEO next hops."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mappo_design import SharedCandidateActor
from variant_definitions import canonical_variant_name


SWITCH_FEATURE_INDEX = 17


@dataclass
class HysteresisDiagnostics:
    action_batches: int = 0
    agent_decisions: int = 0
    no_cache_decisions: int = 0
    cached_route_feasible_decisions: int = 0
    cached_route_unavailable_decisions: int = 0
    raw_switch_choices: int = 0
    suppressed_switch_choices: int = 0
    allowed_switch_choices: int = 0
    raw_switch_gap_sum: float = 0.0
    raw_switch_gap_min: float | None = None
    raw_switch_gap_max: float | None = None

    def record_gap(self, gap: float) -> None:
        self.raw_switch_gap_sum += gap
        self.raw_switch_gap_min = (
            gap if self.raw_switch_gap_min is None else min(self.raw_switch_gap_min, gap)
        )
        self.raw_switch_gap_max = (
            gap if self.raw_switch_gap_max is None else max(self.raw_switch_gap_max, gap)
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["raw_switch_gap_mean"] = (
            self.raw_switch_gap_sum / self.raw_switch_choices
            if self.raw_switch_choices
            else None
        )
        return result


class ActorScoreHysteresisPolicy:
    """Greedy shared actor with a fixed bonus for a cached feasible route.

    Candidate feature 17 is zero for the cached next hop and one for a route
    switch. Before a route cache exists, every feasible next hop has value zero,
    so adding the same bonus leaves the actor ordering unchanged. If the cached
    route is unavailable, every feasible next hop has value one and no route is
    revived by the controller.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        *,
        feature_dim: int,
        action_size: int,
        stay_bonus: float,
        device: str,
        checkpoint_schema: Mapping[str, Any],
    ) -> None:
        if not np.isfinite(stay_bonus) or stay_bonus < 0.0:
            raise ValueError("stay_bonus must be finite and non-negative")
        if feature_dim <= SWITCH_FEATURE_INDEX:
            raise ValueError("checkpoint does not contain the route-switch feature")
        self.actor = actor
        self.feature_dim = int(feature_dim)
        self.action_size = int(action_size)
        self.stay_bonus = float(stay_bonus)
        self.device = str(device)
        self.checkpoint_schema = dict(checkpoint_schema)
        self._diagnostics = HysteresisDiagnostics()

    def __call__(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        expected_shape = (observation.shape[0], self.action_size * self.feature_dim)
        if observation.shape != expected_shape:
            raise ValueError(
                f"observation shape {observation.shape} != {expected_shape}"
            )
        if mask.shape != (observation.shape[0], self.action_size):
            raise ValueError("action-mask shape does not match the checkpoint")

        candidates = torch.from_numpy(observation).to(self.device).reshape(
            observation.shape[0], self.action_size, self.feature_dim
        )
        action_mask = torch.from_numpy(mask).to(self.device)
        with torch.no_grad():
            raw_logits = self.actor(candidates, action_mask)
            stay_mask = (
                candidates[..., SWITCH_FEATURE_INDEX] < 0.5
            ) & action_mask
            raw_actions = raw_logits.argmax(dim=-1)
        raw_logits_array = raw_logits.detach().cpu().numpy()
        raw_actions_array = raw_actions.detach().cpu().numpy()
        stay_mask_array = stay_mask.detach().cpu().numpy()
        actions = self._apply_hysteresis(
            raw_logits_array,
            raw_actions_array,
            mask,
            stay_mask_array,
        )

        self._record_diagnostics(
            raw_logits_array,
            raw_actions_array,
            actions,
            mask,
            stay_mask_array,
        )
        return actions.astype(np.int64, copy=False)

    def _apply_hysteresis(
        self,
        raw_logits: np.ndarray,
        raw_actions: np.ndarray,
        action_mask: np.ndarray,
        stay_mask: np.ndarray,
    ) -> np.ndarray:
        actions = raw_actions.astype(np.int64, copy=True)
        if self.stay_bonus == 0.0:
            return actions
        for index in range(action_mask.shape[0]):
            feasible_next = np.flatnonzero(action_mask[index, 1:]) + 1
            if feasible_next.size == 0:
                continue
            staying = feasible_next[stay_mask[index, feasible_next]]
            if staying.size in (0, feasible_next.size):
                continue
            if staying.size != 1:
                raise RuntimeError(
                    "route-cache invariant violated: multiple but not all feasible "
                    "candidates are marked as non-switching"
                )
            incumbent = int(staying[0])
            raw_action = int(raw_actions[index])
            if raw_action == incumbent:
                continue
            raw_advantage = float(
                raw_logits[index, raw_action] - raw_logits[index, incumbent]
            )
            if raw_advantage < self.stay_bonus:
                actions[index] = incumbent
        return actions

    def _record_diagnostics(
        self,
        raw_logits: np.ndarray,
        raw_actions: np.ndarray,
        actions: np.ndarray,
        action_mask: np.ndarray,
        stay_mask: np.ndarray,
    ) -> None:
        diagnostics = self._diagnostics
        diagnostics.action_batches += 1
        diagnostics.agent_decisions += int(action_mask.shape[0])
        for index in range(action_mask.shape[0]):
            feasible_next = np.flatnonzero(action_mask[index, 1:]) + 1
            if feasible_next.size == 0:
                continue
            staying = feasible_next[stay_mask[index, feasible_next]]
            if staying.size == feasible_next.size:
                diagnostics.no_cache_decisions += 1
                continue
            if staying.size == 0:
                diagnostics.cached_route_unavailable_decisions += 1
                continue
            if staying.size != 1:
                raise RuntimeError(
                    "route-cache invariant violated: multiple but not all feasible "
                    "candidates are marked as non-switching"
                )
            diagnostics.cached_route_feasible_decisions += 1
            incumbent = int(staying[0])
            raw_action = int(raw_actions[index])
            if raw_action == incumbent:
                continue
            diagnostics.raw_switch_choices += 1
            gap = float(raw_logits[index, raw_action] - raw_logits[index, incumbent])
            diagnostics.record_gap(gap)
            if int(actions[index]) == incumbent:
                diagnostics.suppressed_switch_choices += 1
            else:
                diagnostics.allowed_switch_choices += 1

    def diagnostics(self) -> dict[str, Any]:
        return self._diagnostics.as_dict()


def load_hysteresis_policy(
    checkpoint_path: str | Path,
    *,
    stay_bonus: float,
    device: str = "cpu",
) -> tuple[ActorScoreHysteresisPolicy, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor_spec = checkpoint.get("candidate_actor_spec")
    if actor_spec is not None:
        if not isinstance(actor_spec, dict):
            raise ValueError("checkpoint candidate actor spec must be a mapping")
        if actor_spec.get("schema_version") in {2, 3}:
            raise ValueError(
                "post-hoc hysteresis cannot load an adaptive checkpoint"
            )
        integrated_beta = float(actor_spec.get("route_hysteresis_beta", 0.0))
        if not np.isfinite(integrated_beta) or integrated_beta < 0.0:
            raise ValueError("checkpoint contains an invalid integrated beta")
        if integrated_beta > 0.0:
            raise ValueError(
                "post-hoc hysteresis cannot load a checkpoint with integrated beta"
            )
    feature_dim = int(checkpoint["candidate_feature_dim"])
    action_size = int(checkpoint["action_size"])
    actor_args = checkpoint["args"]
    actor = SharedCandidateActor(
        candidate_feature_dim=feature_dim,
        hidden_dim=int(actor_args["actor_hidden_dim"]),
        num_layers=int(actor_args["actor_num_layers"]),
    ).to(device)
    prefix = "shared_candidate_actor."
    actor_state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["actor"].items()
        if key.startswith(prefix)
    }
    if not actor_state:
        raise ValueError("checkpoint does not contain a shared candidate actor")
    actor.load_state_dict(actor_state)
    actor.eval()

    checkpoint_variant = actor_args.get("leo_variant")
    if checkpoint_variant:
        checkpoint_variant = canonical_variant_name(str(checkpoint_variant))
    schema = {
        "candidate_feature_dim": feature_dim,
        "action_size": action_size,
        "obs_size": int(checkpoint.get("obs_size", action_size * feature_dim)),
        "n_agents": int(checkpoint.get("n_agents", 0)),
        "variant": checkpoint_variant,
    }
    return (
        ActorScoreHysteresisPolicy(
            actor,
            feature_dim=feature_dim,
            action_size=action_size,
            stay_bonus=stay_bonus,
            device=device,
            checkpoint_schema=schema,
        ),
        checkpoint,
    )
