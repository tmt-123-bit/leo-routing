"""Age-band reshaping of the frozen v9 candidate-5 hysteresis policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from adaptive_hysteresis_policy import (
    AdaptiveActorScoreHysteresisPolicy,
    load_adaptive_hysteresis_policy,
)
from hysteresis_policy import SWITCH_FEATURE_INDEX


BASE_STAY_BONUS = 1.0
BASE_URGENCY_RELIEF = 0.625
BASE_CLASS_2_RELIEF = 0.75
ADDON_STAY_BONUS = 0.75
AGE_BIN_EDGES = (0.0, 0.125, 0.25, 0.375, 0.5, 1.0)
AGE_BIN_LABELS = (
    "0p000_to_0p125",
    "0p125_to_0p250",
    "0p250_to_0p375",
    "0p375_to_0p500",
    "0p500_to_1p000",
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


class AgeBandActorScoreHysteresisPolicy(
    AdaptiveActorScoreHysteresisPolicy
):
    """Move the v9 calm add-on from the youngest ages into a middle band.

    For clipped waiting age ``a``, class-2 indicator ``z``, and mixture ``m``:

        base = (1 - 0.625 * a) * (1 - 0.75 * z)
        w_old = max(0, 1 - 2 * a)
        w_band = clip(8 * (a - 0.125), 0, 1) * clip(2 - 4 * a, 0, 1)
        beta = base + 0.75 * (1 - z) * ((1 - m) * w_old + m * w_band)

    ``m=0`` is action-equivalent to v9 candidate 5. First-use and forced
    switches retain the raw frozen-actor action through the inherited policy
    contract.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        *,
        feature_dim: int,
        action_size: int,
        age_band_mix: float,
        device: str,
        checkpoint_schema: Mapping[str, Any],
    ) -> None:
        super().__init__(
            actor,
            feature_dim=feature_dim,
            action_size=action_size,
            stay_bonus=BASE_STAY_BONUS,
            urgency_relief=BASE_URGENCY_RELIEF,
            class_2_relief=BASE_CLASS_2_RELIEF,
            device=device,
            checkpoint_schema=checkpoint_schema,
        )
        self.age_band_mix = _unit_interval(age_band_mix, "age-band mix")
        self._weight_raw_switch_choices = 0
        self._old_weight_sum = 0.0
        self._old_weight_min: float | None = None
        self._old_weight_max: float | None = None
        self._band_weight_sum = 0.0
        self._band_weight_min: float | None = None
        self._band_weight_max: float | None = None
        self._mixed_weight_sum = 0.0
        self._mixed_weight_min: float | None = None
        self._mixed_weight_max: float | None = None
        self._threshold_shift_sum = 0.0
        self._threshold_shift_min: float | None = None
        self._threshold_shift_max: float | None = None
        self._threshold_shift_increase_sum = 0.0
        self._threshold_shift_decrease_sum = 0.0
        self._shift_increased_choices = 0
        self._shift_decreased_choices = 0
        self._shift_unchanged_choices = 0
        self._v9_would_suppress_choices = 0
        self._created_by_shift_choices = 0
        self._released_by_shift_choices = 0
        self._age_choice_bins = {
            label: {"suppressed": 0, "allowed": 0}
            for label in AGE_BIN_LABELS
        }
        self._age_shift_bins = {
            label: {"increased": 0, "decreased": 0, "unchanged": 0}
            for label in AGE_BIN_LABELS
        }

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
            base_bonus = (1.0 - BASE_URGENCY_RELIEF * urgency) * (
                1.0 - BASE_CLASS_2_RELIEF * class_2
            )
            old_weight = torch.clamp(
                1.0 - 2.0 * urgency, min=0.0, max=1.0
            )
            band_weight = torch.clamp(
                8.0 * (urgency - 0.125), min=0.0, max=1.0
            ) * torch.clamp(
                2.0 - 4.0 * urgency, min=0.0, max=1.0
            )
            mixed_weight = (
                (1.0 - self.age_band_mix) * old_weight
                + self.age_band_mix * band_weight
            )
            non_class_2 = 1.0 - class_2
            old_bonus = (
                base_bonus + ADDON_STAY_BONUS * non_class_2 * old_weight
            )
            effective_bonus = (
                base_bonus + ADDON_STAY_BONUS * non_class_2 * mixed_weight
            )

        raw_logits_array = raw_logits.detach().cpu().numpy()
        raw_actions_array = raw_actions.detach().cpu().numpy()
        stay_mask_array = stay_mask.detach().cpu().numpy()
        old_weight_array = old_weight.detach().cpu().numpy()
        band_weight_array = band_weight.detach().cpu().numpy()
        mixed_weight_array = mixed_weight.detach().cpu().numpy()
        old_bonus_array = old_bonus.detach().cpu().numpy()
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
        self._record_age_band_diagnostics(
            raw_logits_array,
            raw_actions_array,
            actions,
            mask,
            stay_mask_array,
            urgency_array,
            old_weight_array,
            band_weight_array,
            mixed_weight_array,
            old_bonus_array,
            effective_bonus_array,
        )
        return actions.astype(np.int64, copy=False)

    @staticmethod
    def _update_range(
        value: float,
        current_min: float | None,
        current_max: float | None,
    ) -> tuple[float, float]:
        return (
            value if current_min is None else min(current_min, value),
            value if current_max is None else max(current_max, value),
        )

    def _record_age_band_diagnostics(
        self,
        raw_logits: np.ndarray,
        raw_actions: np.ndarray,
        actions: np.ndarray,
        action_mask: np.ndarray,
        stay_mask: np.ndarray,
        urgency: np.ndarray,
        old_weight: np.ndarray,
        band_weight: np.ndarray,
        mixed_weight: np.ndarray,
        old_bonus: np.ndarray,
        effective_bonus: np.ndarray,
    ) -> None:
        for index in range(action_mask.shape[0]):
            feasible_next = np.flatnonzero(action_mask[index, 1:]) + 1
            if feasible_next.size == 0:
                continue
            staying = feasible_next[stay_mask[index, feasible_next]]
            if staying.size != 1 or int(raw_actions[index]) == int(staying[0]):
                continue

            self._weight_raw_switch_choices += 1
            old = float(old_weight[index])
            band = float(band_weight[index])
            mixed = float(mixed_weight[index])
            self._old_weight_sum += old
            self._old_weight_min, self._old_weight_max = self._update_range(
                old, self._old_weight_min, self._old_weight_max
            )
            self._band_weight_sum += band
            self._band_weight_min, self._band_weight_max = self._update_range(
                band, self._band_weight_min, self._band_weight_max
            )
            self._mixed_weight_sum += mixed
            self._mixed_weight_min, self._mixed_weight_max = self._update_range(
                mixed, self._mixed_weight_min, self._mixed_weight_max
            )

            old_threshold = float(old_bonus[index])
            threshold = float(effective_bonus[index])
            shift = threshold - old_threshold
            self._threshold_shift_sum += shift
            (
                self._threshold_shift_min,
                self._threshold_shift_max,
            ) = self._update_range(
                shift,
                self._threshold_shift_min,
                self._threshold_shift_max,
            )
            if shift > 0.0:
                shift_outcome = "increased"
                self._shift_increased_choices += 1
                self._threshold_shift_increase_sum += shift
            elif shift < 0.0:
                shift_outcome = "decreased"
                self._shift_decreased_choices += 1
                self._threshold_shift_decrease_sum += -shift
            else:
                shift_outcome = "unchanged"
                self._shift_unchanged_choices += 1

            incumbent = int(staying[0])
            raw_action = int(raw_actions[index])
            raw_advantage = float(
                raw_logits[index, raw_action] - raw_logits[index, incumbent]
            )
            v9_would_suppress = raw_advantage < old_threshold
            suppressed = int(actions[index]) == incumbent
            if int(actions[index]) not in (incumbent, raw_action):
                raise RuntimeError(
                    "age-band hysteresis produced an invalid action"
                )
            if shift > 0.0 and v9_would_suppress and not suppressed:
                raise RuntimeError("an increased threshold released a v9 switch")
            if shift < 0.0 and not v9_would_suppress and suppressed:
                raise RuntimeError("a decreased threshold created a suppression")
            if shift == 0.0 and v9_would_suppress != suppressed:
                raise RuntimeError(
                    "an unchanged threshold changed the v9 action"
                )
            created = suppressed and not v9_would_suppress
            released = v9_would_suppress and not suppressed
            self._v9_would_suppress_choices += int(v9_would_suppress)
            self._created_by_shift_choices += int(created)
            self._released_by_shift_choices += int(released)

            outcome = "suppressed" if suppressed else "allowed"
            age_bin = int(
                np.searchsorted(
                    AGE_BIN_EDGES[1:-1], float(urgency[index]), side="right"
                )
            )
            label = AGE_BIN_LABELS[age_bin]
            self._age_choice_bins[label][outcome] += 1
            self._age_shift_bins[label][shift_outcome] += 1

    @staticmethod
    def _mean(total: float, count: int) -> float | None:
        return total / count if count else None

    def diagnostics(self) -> dict[str, Any]:
        result = super().diagnostics()
        raw_switch_choices = int(result["raw_switch_choices"])
        suppressed_switch_choices = int(result["suppressed_switch_choices"])
        if self._weight_raw_switch_choices != raw_switch_choices:
            raise RuntimeError("age-band weight diagnostics do not reconcile")
        shift_total = (
            self._shift_increased_choices
            + self._shift_decreased_choices
            + self._shift_unchanged_choices
        )
        if shift_total != raw_switch_choices:
            raise RuntimeError("age-band shift diagnostics do not reconcile")
        for label in AGE_BIN_LABELS:
            choice_total = sum(self._age_choice_bins[label].values())
            shift_bin_total = sum(self._age_shift_bins[label].values())
            if choice_total != shift_bin_total:
                raise RuntimeError(
                    "age-band per-bin diagnostics do not reconcile"
                )
        if sum(
            sum(counts.values()) for counts in self._age_choice_bins.values()
        ) != raw_switch_choices:
            raise RuntimeError("age-band choice bins do not reconcile")
        if (
            self._v9_would_suppress_choices
            - self._released_by_shift_choices
            + self._created_by_shift_choices
            != suppressed_switch_choices
        ):
            raise RuntimeError(
                "age-band counterfactual outcomes do not reconcile"
            )

        result.update(
            {
                "adaptive_mode": "posthoc_age_band_mixture",
                "age_band_mix": self.age_band_mix,
                "v9_reference_candidate": (
                    "b1p000_u0p625_c0p750_k0p750"
                ),
                "age_band_definition": (
                    "base=(1-0.625*a)*(1-0.75*z); "
                    "w_old=max(0,1-2*a); "
                    "w_band=clip(8*(a-0.125),0,1)*clip(2-4*a,0,1); "
                    "beta=base+0.75*(1-z)*((1-m)*w_old+m*w_band)"
                ),
                "old_weight_sum": self._old_weight_sum,
                "old_weight_min": self._old_weight_min,
                "old_weight_max": self._old_weight_max,
                "old_weight_mean": self._mean(
                    self._old_weight_sum, raw_switch_choices
                ),
                "band_weight_sum": self._band_weight_sum,
                "band_weight_min": self._band_weight_min,
                "band_weight_max": self._band_weight_max,
                "band_weight_mean": self._mean(
                    self._band_weight_sum, raw_switch_choices
                ),
                "mixed_weight_sum": self._mixed_weight_sum,
                "mixed_weight_min": self._mixed_weight_min,
                "mixed_weight_max": self._mixed_weight_max,
                "mixed_weight_mean": self._mean(
                    self._mixed_weight_sum, raw_switch_choices
                ),
                "threshold_shift_sum": self._threshold_shift_sum,
                "threshold_shift_min": self._threshold_shift_min,
                "threshold_shift_max": self._threshold_shift_max,
                "threshold_shift_mean": self._mean(
                    self._threshold_shift_sum, raw_switch_choices
                ),
                "threshold_shift_increase_sum": (
                    self._threshold_shift_increase_sum
                ),
                "threshold_shift_decrease_sum": (
                    self._threshold_shift_decrease_sum
                ),
                "threshold_increased_raw_switch_choices": (
                    self._shift_increased_choices
                ),
                "threshold_decreased_raw_switch_choices": (
                    self._shift_decreased_choices
                ),
                "threshold_unchanged_raw_switch_choices": (
                    self._shift_unchanged_choices
                ),
                "v9_would_suppress_switch_choices": (
                    self._v9_would_suppress_choices
                ),
                "created_by_band_shift_switch_choices": (
                    self._created_by_shift_choices
                ),
                "released_by_band_shift_switch_choices": (
                    self._released_by_shift_choices
                ),
                "age_bin_edges": list(AGE_BIN_EDGES),
                "age_choice_bins": {
                    label: dict(counts)
                    for label, counts in self._age_choice_bins.items()
                },
                "age_shift_bins": {
                    label: dict(counts)
                    for label, counts in self._age_shift_bins.items()
                },
            }
        )
        return result


def load_age_band_hysteresis_policy(
    checkpoint_path: str | Path,
    *,
    age_band_mix: float,
    device: str = "cpu",
) -> tuple[AgeBandActorScoreHysteresisPolicy, dict[str, Any]]:
    base, checkpoint = load_adaptive_hysteresis_policy(
        checkpoint_path,
        stay_bonus=BASE_STAY_BONUS,
        urgency_relief=BASE_URGENCY_RELIEF,
        class_2_relief=BASE_CLASS_2_RELIEF,
        device=device,
    )
    return (
        AgeBandActorScoreHysteresisPolicy(
            base.actor,
            feature_dim=base.feature_dim,
            action_size=base.action_size,
            age_band_mix=age_band_mix,
            device=device,
            checkpoint_schema=base.checkpoint_schema,
        ),
        checkpoint,
    )
