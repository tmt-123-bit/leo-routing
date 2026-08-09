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


class GraphAttentionCritic(nn.Module):
    """Permutation-invariant centralized value network for a graph state."""

    def __init__(
        self,
        n_nodes: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.global_feature_dim = global_feature_dim
        self.state_size = (
            global_feature_dim
            + n_nodes * node_feature_dim
            + n_nodes * n_nodes * edge_feature_dim
        )
        self.node_encoder = nn.Sequential(
            nn.LayerNorm(node_feature_dim),
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(edge_feature_dim),
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.message_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        for _ in range(max(1, num_layers)):
            self.message_layers.append(
                nn.Sequential(
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
            )
            self.attention_layers.append(
                nn.Sequential(
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                )
            )
            self.update_layers.append(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                )
            )
        self.global_encoder = nn.Sequential(
            nn.LayerNorm(global_feature_dim),
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def split_state(self, state: torch.Tensor):
        if state.shape[-1] != self.state_size:
            raise ValueError(
                f"graph state has {state.shape[-1]} values, expected {self.state_size}"
            )
        global_end = self.global_feature_dim
        node_end = global_end + self.n_nodes * self.node_feature_dim
        global_features = state[..., :global_end]
        node_features = state[..., global_end:node_end].reshape(
            *state.shape[:-1], self.n_nodes, self.node_feature_dim
        )
        edge_features = state[..., node_end:].reshape(
            *state.shape[:-1],
            self.n_nodes,
            self.n_nodes,
            self.edge_feature_dim,
        )
        return global_features, node_features, edge_features

    def forward(self, state: torch.Tensor):
        leading_shape = state.shape[:-1]
        flat_state = state.reshape(-1, state.shape[-1])
        global_features, node_features, edge_features = self.split_state(
            flat_state
        )
        nodes = self.node_encoder(node_features)
        edges = self.edge_encoder(edge_features)
        edge_mask = edge_features[..., 0] > 0.5
        identity = torch.eye(
            self.n_nodes, dtype=torch.bool, device=state.device
        ).unsqueeze(0)
        message_mask = edge_mask | identity

        for message_layer, attention_layer, update_layer in zip(
            self.message_layers,
            self.attention_layers,
            self.update_layers,
        ):
            src = nodes.unsqueeze(2).expand(-1, -1, self.n_nodes, -1)
            dst = nodes.unsqueeze(1).expand(-1, self.n_nodes, -1, -1)
            context = torch.cat([src, dst, edges], dim=-1)
            messages = message_layer(context)
            attention_logits = attention_layer(context).squeeze(-1)
            attention_logits = attention_logits.masked_fill(
                ~message_mask, -1e9
            )
            attention = torch.softmax(attention_logits, dim=1)
            aggregated = (attention.unsqueeze(-1) * messages).sum(dim=1)
            nodes = nodes + update_layer(torch.cat([nodes, aggregated], dim=-1))

        pooled_mean = nodes.mean(dim=1)
        pooled_max = nodes.max(dim=1).values
        global_context = self.global_encoder(global_features)
        value = self.value_head(
            torch.cat([pooled_mean, pooled_max, global_context], dim=-1)
        )
        return value.reshape(*leading_shape, 1)


def permute_flat_graph_state(
    state: torch.Tensor,
    permutation: torch.Tensor,
    *,
    n_nodes: int,
    node_feature_dim: int,
    edge_feature_dim: int,
    global_feature_dim: int,
) -> torch.Tensor:
    """Reorder node labels while preserving the represented physical graph."""
    global_end = global_feature_dim
    node_end = global_end + n_nodes * node_feature_dim
    global_features = state[..., :global_end]
    node_features = state[..., global_end:node_end].reshape(
        *state.shape[:-1], n_nodes, node_feature_dim
    )
    edge_features = state[..., node_end:].reshape(
        *state.shape[:-1], n_nodes, n_nodes, edge_feature_dim
    )
    node_features = node_features[..., permutation, :]
    edge_features = edge_features[..., permutation, :, :]
    edge_features = edge_features[..., :, permutation, :]
    return torch.cat(
        [
            global_features,
            node_features.reshape(*state.shape[:-1], -1),
            edge_features.reshape(*state.shape[:-1], -1),
        ],
        dim=-1,
    )


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


def shuffled_transition_minibatches(valid_mask, num_minibatches, generator=None):
    """Partition every valid (episode, time) index exactly once."""
    transitions = valid_mask.nonzero(as_tuple=False)
    if len(transitions) == 0:
        return []
    count = min(max(1, int(num_minibatches)), len(transitions))
    permutation = torch.randperm(
        len(transitions), device=transitions.device, generator=generator
    )
    return [transitions[index] for index in permutation.chunk(count)]
