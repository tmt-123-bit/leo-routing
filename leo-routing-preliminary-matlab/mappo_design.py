"""Testable MAPPO building blocks used by the revised LEO integration."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, values) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        batch_count = values.size
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (
            m_a
            + m_b
            + delta * delta * self.count * batch_count / total
        ) / total
        self.count = total


class SharedCandidateActor(nn.Module):
    """Permutation-equivariant scorer for a set of next-hop candidates."""

    def __init__(
        self,
        candidate_feature_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        encoder = [
            nn.Linear(candidate_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            encoder.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*encoder)
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidates: torch.Tensor, action_mask=None):
        encoded = self.encoder(candidates)
        if action_mask is None:
            weights = torch.ones_like(encoded[..., :1])
        else:
            weights = action_mask.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=-2) / weights.sum(
            dim=-2
        ).clamp_min(1.0)
        context = pooled.unsqueeze(-2).expand_as(encoded)
        logits = self.scorer(torch.cat([encoded, context], dim=-1)).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return logits


class PacketConditionedCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        layers = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor):
        return self.network(state)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    valid: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    """GAE with zero bootstrap only for true terminal states.

    Time-limit truncation uses next_values for the one-step delta, then stops
    carrying sampled future advantages across the rollout boundary.
    """
    advantages = torch.zeros_like(values)
    returns = torch.zeros_like(values)
    for batch in range(values.shape[0]):
        length = int(valid[batch].sum().item())
        next_gae = torch.zeros_like(values[batch, 0])
        for t in reversed(range(length)):
            nonterminal = 1.0 - terminated[batch, t]
            delta = (
                rewards[batch, t]
                + gamma * nonterminal * next_values[batch, t]
                - values[batch, t]
            )
            carry = nonterminal * (1.0 - truncated[batch, t])
            next_gae = delta + gamma * gae_lambda * carry * next_gae
            advantages[batch, t] = next_gae
            returns[batch, t] = next_gae + values[batch, t]
    return advantages, returns


def masked_standardize(values, mask, epsilon: float = 1e-8):
    selected = values[mask]
    mean = selected.mean()
    std = selected.std(unbiased=False)
    result = values.clone()
    result[mask] = (selected - mean) / (std + epsilon)
    return result


def feasible_normalized_entropy(distribution, action_mask):
    feasible = action_mask.sum(dim=-1).clamp_min(1)
    denominator = torch.log(feasible.clamp_min(2).float())
    return distribution.entropy() / denominator
