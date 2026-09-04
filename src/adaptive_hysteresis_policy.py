"""State-aware post-hoc hysteresis for frozen candidate-actor checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from hysteresis_policy import (
    SWITCH_FEATURE_INDEX,
    ActorScoreHysteresisPolicy,
    load_hysteresis_policy,
)
from leo_multiagent_env import (
    BASE_CANDIDATE_FEATURE_NAMES,
    ROUTE_CLASS_2_FEATURE_INDEX,
    ROUTE_URGENCY_FEATURE_INDEX,
    candidate_feature_schema,
)


_CANONICAL_FEATURE_SCHEMA = candidate_feature_schema()
_CANONICAL_FEATURE_NAMES = tuple(BASE_CANDIDATE_FEATURE_NAMES)
_CANONICAL_FEATURE_DIM = len(_CANONICAL_FEATURE_NAMES)
_CANONICAL_SCHEMA_ID = str(_CANONICAL_FEATURE_SCHEMA["schema_id"])
_CANONICAL_SCHEMA_SHA256 = str(_CANONICAL_FEATURE_SCHEMA["sha256"])
_PACKET_STATE_ATOL = 1e-6
_MAX_RAW_URGENCY = 2.0
_CHOICE_STRATA = (
    "class_2",
    "non_class_2",
    "urgent",
    "non_urgent",
    "class_2_urgent",
    "class_2_non_urgent",
    "non_class_2_urgent",
    "non_class_2_non_urgent",
)

_EXPECTED_SEMANTICS = {
    SWITCH_FEATURE_INDEX: "route_switch_indicator",
    ROUTE_URGENCY_FEATURE_INDEX: "deadline_normalized_waiting_age",
    ROUTE_CLASS_2_FEATURE_INDEX: "traffic_class_2",
}
for _index, _name in _EXPECTED_SEMANTICS.items():
    if _CANONICAL_FEATURE_NAMES[_index] != _name:
        raise RuntimeError(
            "canonical candidate feature semantics no longer match the adaptive "
            "hysteresis contract"
        )


def _unit_interval(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    return numeric


def _schema_integer(schema: Mapping[str, Any], field: str) -> int:
    value = schema.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"checkpoint schema field {field} must be an integer")
    return int(value)


def _validate_checkpoint_schema(
    checkpoint_schema: Mapping[str, Any],
    *,
    feature_dim: int,
    action_size: int,
) -> dict[str, Any]:
    if not isinstance(checkpoint_schema, Mapping):
        raise ValueError("checkpoint schema must be a mapping")
    schema = dict(checkpoint_schema)
    if feature_dim != _CANONICAL_FEATURE_DIM:
        raise ValueError(
            "adaptive hysteresis requires the canonical 26-dimensional "
            "candidate feature schema"
        )
    if _schema_integer(schema, "candidate_feature_dim") != feature_dim:
        raise ValueError("checkpoint candidate feature dimension mismatch")
    if _schema_integer(schema, "action_size") != action_size:
        raise ValueError("checkpoint action size mismatch")
    if _schema_integer(schema, "obs_size") != action_size * feature_dim:
        raise ValueError("checkpoint observation size mismatch")
    if _schema_integer(schema, "n_agents") <= 0:
        raise ValueError("checkpoint agent count must be positive")
    if schema.get("variant") != "proposed":
        raise ValueError("adaptive hysteresis requires a proposed checkpoint")
    if schema.get("candidate_feature_schema_id") != _CANONICAL_SCHEMA_ID:
        raise ValueError("checkpoint candidate feature schema id mismatch")
    if schema.get("candidate_feature_schema_sha256") != _CANONICAL_SCHEMA_SHA256:
        raise ValueError("checkpoint candidate feature schema hash mismatch")
    names = schema.get("candidate_feature_names")
    if not isinstance(names, (list, tuple)) or tuple(names) != _CANONICAL_FEATURE_NAMES:
        raise ValueError("checkpoint candidate feature names mismatch")
    return schema


class AdaptiveActorScoreHysteresisPolicy(ActorScoreHysteresisPolicy):
    """Apply a smaller stay bonus to urgent and class-2 packets.

    The frozen actor logits remain unchanged. Only cached-feasible route
    switches can be suppressed, using

        base_bonus * (1 - urgency_relief * urgency)
                   * (1 - class_2_relief * class_2).

    Urgency is clipped to [0, 1]. First-use and forced-switch states retain the
    raw actor action exactly, as in ``ActorScoreHysteresisPolicy``.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        *,
        feature_dim: int,
        action_size: int,
        stay_bonus: float,
        urgency_relief: float,
        class_2_relief: float,
        device: str,
        checkpoint_schema: Mapping[str, Any],
        urgency_feature_index: int = ROUTE_URGENCY_FEATURE_INDEX,
        class_2_feature_index: int = ROUTE_CLASS_2_FEATURE_INDEX,
    ) -> None:
        validated_schema = _validate_checkpoint_schema(
            checkpoint_schema,
            feature_dim=feature_dim,
            action_size=action_size,
        )
        if urgency_feature_index != ROUTE_URGENCY_FEATURE_INDEX:
            raise ValueError(
                "urgency feature index does not match the canonical candidate schema"
            )
        if class_2_feature_index != ROUTE_CLASS_2_FEATURE_INDEX:
            raise ValueError(
                "class-2 feature index does not match the canonical candidate schema"
            )
        super().__init__(
            actor,
            feature_dim=feature_dim,
            action_size=action_size,
            stay_bonus=stay_bonus,
            device=device,
            checkpoint_schema=validated_schema,
        )
        self.urgency_relief = _unit_interval(
            urgency_relief, "urgency relief"
        )
        self.class_2_relief = _unit_interval(
            class_2_relief, "class-2 relief"
        )
        self.urgency_feature_index = urgency_feature_index
        self.class_2_feature_index = class_2_feature_index
        self._effective_bonus_sum = 0.0
        self._effective_bonus_min: float | None = None
        self._effective_bonus_max: float | None = None
        self._zero_bonus_raw_switch_choices = 0
        self._class_2_raw_switch_choices = 0
        self._urgent_raw_switch_choices = 0
        self._choice_strata = {
            stratum: {"suppressed": 0, "allowed": 0}
            for stratum in _CHOICE_STRATA
        }

    def _validated_packet_state(
        self,
        candidates: np.ndarray,
        action_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        urgency = np.zeros(action_mask.shape[0], dtype=np.float32)
        class_2 = np.zeros(action_mask.shape[0], dtype=np.float32)
        for index in range(action_mask.shape[0]):
            feasible_next = np.flatnonzero(action_mask[index, 1:]) + 1
            if feasible_next.size == 0:
                continue

            switch_values = candidates[
                index, feasible_next, SWITCH_FEATURE_INDEX
            ]
            if not np.isfinite(switch_values).all() or not np.all(
                np.isclose(
                    switch_values,
                    0.0,
                    rtol=0.0,
                    atol=_PACKET_STATE_ATOL,
                )
                | np.isclose(
                    switch_values,
                    1.0,
                    rtol=0.0,
                    atol=_PACKET_STATE_ATOL,
                )
            ):
                raise ValueError(
                    "route-switch feature must be binary on feasible candidates"
                )

            urgency_values = candidates[
                index, feasible_next, self.urgency_feature_index
            ]
            if not np.isfinite(urgency_values).all():
                raise ValueError("urgency feature must be finite")
            if np.any(urgency_values < 0.0) or np.any(
                urgency_values > _MAX_RAW_URGENCY
            ):
                raise ValueError("urgency feature is outside its schema range [0, 2]")
            if not np.allclose(
                urgency_values,
                urgency_values[0],
                rtol=0.0,
                atol=_PACKET_STATE_ATOL,
            ):
                raise ValueError(
                    "urgency feature is inconsistent across feasible candidates"
                )

            class_2_values = candidates[
                index, feasible_next, self.class_2_feature_index
            ]
            if not np.isfinite(class_2_values).all() or not np.all(
                np.isclose(
                    class_2_values,
                    0.0,
                    rtol=0.0,
                    atol=_PACKET_STATE_ATOL,
                )
                | np.isclose(
                    class_2_values,
                    1.0,
                    rtol=0.0,
                    atol=_PACKET_STATE_ATOL,
                )
            ):
                raise ValueError(
                    "class-2 feature must be binary on feasible candidates"
                )
            if not np.allclose(
                class_2_values,
                class_2_values[0],
                rtol=0.0,
                atol=_PACKET_STATE_ATOL,
            ):
                raise ValueError(
                    "class-2 feature is inconsistent across feasible candidates"
                )

            urgency[index] = np.clip(urgency_values[0], 0.0, 1.0)
            class_2[index] = float(class_2_values[0] >= 0.5)
        return urgency, class_2

    def __call__(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        expected_shape = (
            observation.shape[0],
            self.action_size * self.feature_dim,
        )
        if observation.shape != expected_shape:
            raise ValueError(
                f"observation shape {observation.shape} != {expected_shape}"
            )
        if mask.shape != (observation.shape[0], self.action_size):
            raise ValueError("action-mask shape does not match the checkpoint")

        candidate_array = observation.reshape(
            observation.shape[0], self.action_size, self.feature_dim
        )
        urgency_array, class_2_array = self._validated_packet_state(
            candidate_array, mask
        )
        candidates = torch.from_numpy(candidate_array).to(self.device)
        action_mask = torch.from_numpy(mask).to(self.device)
        with torch.no_grad():
            raw_logits = self.actor(candidates, action_mask)
            stay_mask = (
                candidates[..., SWITCH_FEATURE_INDEX] < 0.5
            ) & action_mask
            raw_actions = raw_logits.argmax(dim=-1)
            urgency = torch.from_numpy(urgency_array).to(self.device)
            class_2 = torch.from_numpy(class_2_array).to(self.device)
            effective_bonus = self.stay_bonus * (
                1.0 - self.urgency_relief * urgency
            ) * (1.0 - self.class_2_relief * class_2)

        raw_logits_array = raw_logits.detach().cpu().numpy()
        raw_actions_array = raw_actions.detach().cpu().numpy()
        stay_mask_array = stay_mask.detach().cpu().numpy()
        effective_bonus_array = effective_bonus.detach().cpu().numpy()
        actions = self._apply_adaptive_hysteresis(
            raw_logits_array,
            raw_actions_array,
            mask,
            stay_mask_array,
            effective_bonus_array,
        )
        self._record_diagnostics(
            raw_logits_array,
            raw_actions_array,
            actions,
            mask,
            stay_mask_array,
        )
        self._record_adaptive_diagnostics(
            raw_actions_array,
            actions,
            mask,
            stay_mask_array,
            urgency_array,
            class_2_array,
            effective_bonus_array,
        )
        return actions.astype(np.int64, copy=False)

    @staticmethod
    def _apply_adaptive_hysteresis(
        raw_logits: np.ndarray,
        raw_actions: np.ndarray,
        action_mask: np.ndarray,
        stay_mask: np.ndarray,
        effective_bonus: np.ndarray,
    ) -> np.ndarray:
        actions = raw_actions.astype(np.int64, copy=True)
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
            if raw_advantage < float(effective_bonus[index]):
                actions[index] = incumbent
        return actions

    def _record_adaptive_diagnostics(
        self,
        raw_actions: np.ndarray,
        actions: np.ndarray,
        action_mask: np.ndarray,
        stay_mask: np.ndarray,
        urgency: np.ndarray,
        class_2: np.ndarray,
        effective_bonus: np.ndarray,
    ) -> None:
        for index in range(action_mask.shape[0]):
            feasible_next = np.flatnonzero(action_mask[index, 1:]) + 1
            if feasible_next.size == 0:
                continue
            staying = feasible_next[stay_mask[index, feasible_next]]
            if staying.size != 1 or int(raw_actions[index]) == int(staying[0]):
                continue
            bonus = float(effective_bonus[index])
            self._effective_bonus_sum += bonus
            self._effective_bonus_min = (
                bonus
                if self._effective_bonus_min is None
                else min(self._effective_bonus_min, bonus)
            )
            self._effective_bonus_max = (
                bonus
                if self._effective_bonus_max is None
                else max(self._effective_bonus_max, bonus)
            )
            self._zero_bonus_raw_switch_choices += int(bonus == 0.0)
            is_class_2 = bool(class_2[index] >= 0.5)
            is_urgent = bool(urgency[index] >= 0.5)
            self._class_2_raw_switch_choices += int(is_class_2)
            self._urgent_raw_switch_choices += int(is_urgent)

            incumbent = int(staying[0])
            action = int(actions[index])
            raw_action = int(raw_actions[index])
            if action == incumbent:
                outcome = "suppressed"
            elif action == raw_action:
                outcome = "allowed"
            else:
                raise RuntimeError(
                    "adaptive hysteresis produced neither the raw nor incumbent action"
                )
            class_stratum = "class_2" if is_class_2 else "non_class_2"
            urgency_stratum = "urgent" if is_urgent else "non_urgent"
            for stratum in (
                class_stratum,
                urgency_stratum,
                f"{class_stratum}_{urgency_stratum}",
            ):
                self._choice_strata[stratum][outcome] += 1

    def diagnostics(self) -> dict[str, Any]:
        result = super().diagnostics()
        raw_switch_choices = int(result["raw_switch_choices"])
        for partition in (
            ("class_2", "non_class_2"),
            ("urgent", "non_urgent"),
            (
                "class_2_urgent",
                "class_2_non_urgent",
                "non_class_2_urgent",
                "non_class_2_non_urgent",
            ),
        ):
            partition_total = sum(
                sum(self._choice_strata[stratum].values())
                for stratum in partition
            )
            if partition_total != raw_switch_choices:
                raise RuntimeError("adaptive diagnostic strata do not reconcile")
        flattened_strata = {
            f"{stratum}_{outcome}_switch_choices": count
            for stratum, counts in self._choice_strata.items()
            for outcome, count in counts.items()
        }
        result.update(
            {
                "adaptive_mode": "posthoc_multiplicative_relief",
                "base_stay_bonus": self.stay_bonus,
                "urgency_relief": self.urgency_relief,
                "class_2_relief": self.class_2_relief,
                "urgency_feature_index": self.urgency_feature_index,
                "class_2_feature_index": self.class_2_feature_index,
                "effective_bonus_sum": self._effective_bonus_sum,
                "effective_bonus_min": self._effective_bonus_min,
                "effective_bonus_max": self._effective_bonus_max,
                "effective_bonus_mean": (
                    self._effective_bonus_sum / raw_switch_choices
                    if raw_switch_choices
                    else None
                ),
                "zero_bonus_raw_switch_choices": (
                    self._zero_bonus_raw_switch_choices
                ),
                "class_2_raw_switch_choices": self._class_2_raw_switch_choices,
                "non_class_2_raw_switch_choices": (
                    raw_switch_choices - self._class_2_raw_switch_choices
                ),
                "urgent_raw_switch_choices": self._urgent_raw_switch_choices,
                "non_urgent_raw_switch_choices": (
                    raw_switch_choices - self._urgent_raw_switch_choices
                ),
                "adaptive_choice_strata": {
                    stratum: dict(counts)
                    for stratum, counts in self._choice_strata.items()
                },
                **flattened_strata,
            }
        )
        return result


def load_adaptive_hysteresis_policy(
    checkpoint_path: str | Path,
    *,
    stay_bonus: float,
    urgency_relief: float,
    class_2_relief: float,
    device: str = "cpu",
) -> tuple[AdaptiveActorScoreHysteresisPolicy, dict[str, Any]]:
    base, checkpoint = load_hysteresis_policy(
        checkpoint_path,
        stay_bonus=0.0,
        device=device,
    )
    if checkpoint.get("candidate_actor_spec") is not None:
        raise ValueError(
            "post-hoc adaptive hysteresis requires a legacy frozen actor"
        )
    checkpoint_schema = {
        **base.checkpoint_schema,
        "candidate_feature_schema_id": _CANONICAL_SCHEMA_ID,
        "candidate_feature_schema_sha256": _CANONICAL_SCHEMA_SHA256,
        "candidate_feature_names": list(_CANONICAL_FEATURE_NAMES),
    }
    return (
        AdaptiveActorScoreHysteresisPolicy(
            base.actor,
            feature_dim=base.feature_dim,
            action_size=base.action_size,
            stay_bonus=stay_bonus,
            urgency_relief=urgency_relief,
            class_2_relief=class_2_relief,
            device=device,
            checkpoint_schema=checkpoint_schema,
        ),
        checkpoint,
    )
