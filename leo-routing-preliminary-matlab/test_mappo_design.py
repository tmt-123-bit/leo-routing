from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from leo_marl_env import LeoRoutingEnv
from leo_multiagent_env import (
    SynchronousLeoMultiAgentEnv,
    first_feasible_actions,
)
from mappo_design import (
    PacketConditionedCritic,
    SharedCandidateActor,
    compute_gae,
    masked_standardize,
)


class ObservationTests(unittest.TestCase):
    def test_actor_observation_distinguishes_switch_context(self):
        env = LeoRoutingEnv.from_scenario("medium_load")
        obs = env.reset(src=1, dst=12)
        neighbor = obs["neighbor_ids"][0]
        env.packet.last_next_hop = neighbor
        same_hop = env.as_mappo_inputs(env._make_obs())["actor_obs"]
        env.packet.last_next_hop = obs["neighbor_ids"][-1]
        switched_hop = env.as_mappo_inputs(env._make_obs())["actor_obs"]
        self.assertNotEqual(same_hop, switched_hop)

    def test_actor_and_critic_distinguish_ttl_context(self):
        env = LeoRoutingEnv.from_scenario("medium_load")
        env.reset(src=1, dst=12)
        env.packet.hop_count = 0
        low = env.as_mappo_inputs(env._make_obs())
        env.packet.hop_count = env.cfg.max_local_hops - 1
        high = env.as_mappo_inputs(env._make_obs())
        self.assertNotEqual(low["actor_obs"], high["actor_obs"])
        self.assertNotEqual(low["critic_state"], high["critic_state"])

    def test_single_packet_schema(self):
        env = LeoRoutingEnv.from_scenario("medium_load")
        inputs = env.as_mappo_inputs(env.reset(src=1, dst=12))
        self.assertEqual(len(inputs["candidate_obs"]), env.max_degree)
        self.assertTrue(
            all(
                len(row) == env.candidate_feature_dim
                for row in inputs["candidate_obs"]
            )
        )
        self.assertTrue(np.isfinite(inputs["critic_state"]).all())


class NetworkMathTests(unittest.TestCase):
    def test_candidate_logits_are_permutation_equivariant(self):
        torch.manual_seed(7)
        actor = SharedCandidateActor(20, 32, 1)
        candidates = torch.randn(4, 6, 20)
        mask = torch.tensor(
            [[True, True, False, True, True, False]] * 4,
            dtype=torch.bool,
        )
        permutation = torch.tensor([3, 0, 5, 1, 4, 2])
        inverse = torch.argsort(permutation)
        original = actor(candidates, mask)
        permuted = actor(candidates[:, permutation], mask[:, permutation])
        restored = permuted[:, inverse]
        self.assertTrue(torch.allclose(original, restored, atol=1e-6))

    def test_zero_variance_standardization_is_finite(self):
        values = torch.ones(2, 3, 4)
        mask = torch.ones_like(values, dtype=torch.bool)
        normalized = masked_standardize(values, mask)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.allclose(normalized, torch.zeros_like(normalized)))

    def test_terminal_and_truncation_bootstrap(self):
        rewards = torch.tensor([[[1.0]]])
        values = torch.tensor([[[0.5]]])
        next_values = torch.tensor([[[2.0]]])
        valid = torch.tensor([[True]])

        terminal_adv, terminal_ret = compute_gae(
            rewards,
            values,
            next_values,
            terminated=torch.tensor([[[1.0]]]),
            truncated=torch.tensor([[[0.0]]]),
            valid=valid,
        )
        trunc_adv, trunc_ret = compute_gae(
            rewards,
            values,
            next_values,
            terminated=torch.tensor([[[0.0]]]),
            truncated=torch.tensor([[[1.0]]]),
            valid=valid,
        )
        self.assertAlmostEqual(terminal_ret.item(), 1.0, places=6)
        self.assertAlmostEqual(terminal_adv.item(), 0.5, places=6)
        self.assertAlmostEqual(trunc_ret.item(), 2.98, places=6)
        self.assertAlmostEqual(trunc_adv.item(), 2.48, places=6)

    def test_supervised_candidate_capacity(self):
        torch.manual_seed(9)
        actor = SharedCandidateActor(20, 32, 1)
        optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
        candidates = torch.randn(128, 6, 20)
        mask = torch.ones(128, 6, dtype=torch.bool)
        target = candidates[:, :, 8].argmax(dim=-1)
        for _ in range(120):
            logits = actor(candidates, mask)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = (actor(candidates, mask).argmax(dim=-1) == target).float().mean()
        self.assertGreater(float(accuracy), 0.95)


class MultiAgentEnvironmentTests(unittest.TestCase):
    pairs = [(1, 12), (2, 13), (3, 14), (4, 15), (5, 16), (6, 17)]

    def test_concurrent_actions_and_packet_conservation(self):
        env = SynchronousLeoMultiAgentEnv.from_scenario("medium_load")
        obs, _ = env.reset(seed=33, initial_pairs=self.pairs)
        actions = first_feasible_actions(obs)
        _, _, _, _, info = env.step(actions)
        self.assertGreaterEqual(info["concurrent_non_noop"], 2)
        env.validate_invariants()
        self.assertEqual(
            len(env.generated),
            len(env.delivered) + len(env.dropped) + info["backlog"],
        )

    def test_agent_iteration_order_does_not_change_transition(self):
        env_a = SynchronousLeoMultiAgentEnv.from_scenario("medium_load")
        env_b = SynchronousLeoMultiAgentEnv.from_scenario("medium_load")
        obs_a, _ = env_a.reset(seed=44, initial_pairs=self.pairs)
        obs_b, _ = env_b.reset(seed=44, initial_pairs=self.pairs)
        actions = first_feasible_actions(obs_a)
        self.assertEqual(actions, first_feasible_actions(obs_b))
        env_a.step(actions, agent_order=list(range(1, env_a.n_agents + 1)))
        env_b.step(actions, agent_order=list(range(env_b.n_agents, 0, -1)))
        self.assertEqual(env_a.state_digest(), env_b.state_digest())
        self.assertEqual(env_a.trace_hash(), env_b.trace_hash())

    def test_deterministic_replay(self):
        digests = []
        for _ in range(2):
            env = SynchronousLeoMultiAgentEnv.from_scenario("fault_links", seed=55)
            obs, _ = env.reset(seed=55, initial_pairs=self.pairs)
            for _ in range(4):
                obs, _, terminated, truncated, _ = env.step(
                    first_feasible_actions(obs)
                )
                if terminated or truncated:
                    break
            digests.append((env.state_digest(), env.trace_hash()))
        self.assertEqual(digests[0], digests[1])

    def test_inactive_agents_are_forced_noop_and_excluded_from_policy(self):
        wrapper = CleanMARLLeoMultiAgentWrapper("medium_load")
        wrapper.reset()
        avail = wrapper.get_avail_actions()
        active = wrapper.get_policy_active_mask().astype(bool)
        self.assertTrue((avail[~active, 0] == 1).all())
        self.assertTrue((avail[~active, 1:] == 0).all())
        self.assertTrue((avail[active, 0] == 0).all())

    def test_actor_observation_has_no_global_state_field(self):
        env = SynchronousLeoMultiAgentEnv.from_scenario("medium_load")
        observations, _ = env.reset(seed=66, initial_pairs=self.pairs)
        forbidden = {"global_state", "all_queues", "future_topology"}
        for obs in observations:
            self.assertTrue(forbidden.isdisjoint(obs.keys()))

    def test_multiagent_single_ppo_update_is_finite(self):
        torch.manual_seed(10)
        wrapper = CleanMARLLeoMultiAgentWrapper("medium_load")
        obs, _ = wrapper.reset()
        mask = torch.from_numpy(wrapper.get_avail_actions()).bool()
        active = torch.from_numpy(wrapper.get_policy_active_mask()).bool()
        candidates = torch.from_numpy(obs).float().reshape(
            wrapper.n_agents,
            wrapper.get_action_size(),
            wrapper.get_candidate_feature_dim(),
        )
        actor = SharedCandidateActor(wrapper.get_candidate_feature_dim(), 32, 1)
        critic = PacketConditionedCritic(wrapper.get_state_size(), 64, 1)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=8e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=8e-4)

        old_logits = actor(candidates, mask)
        old_dist = torch.distributions.Categorical(logits=old_logits)
        actions = old_dist.sample()
        old_log_prob = old_dist.log_prob(actions).detach()
        state = torch.from_numpy(wrapper.get_state()).float()
        _, reward, terminated, truncated, _ = wrapper.step(actions.numpy())
        next_state = torch.from_numpy(wrapper.get_state()).float()

        values = critic(state).reshape(1, 1, 1).expand(1, 1, wrapper.n_agents)
        next_values = critic(next_state).reshape(1, 1, 1).expand_as(values)
        rewards = torch.full_like(values, reward)
        advantages, returns = compute_gae(
            rewards,
            values.detach(),
            next_values.detach(),
            terminated=torch.full_like(values, float(terminated)),
            truncated=torch.full_like(values, float(truncated)),
            valid=torch.tensor([[True]]),
        )
        active_mask = active.reshape(1, 1, -1)
        advantages = masked_standardize(advantages, active_mask)

        new_logits = actor(candidates, mask)
        new_dist = torch.distributions.Categorical(logits=new_logits)
        ratio = torch.exp(new_dist.log_prob(actions) - old_log_prob)
        ratio = ratio.reshape(1, 1, -1)
        actor_loss = -torch.min(
            ratio * advantages,
            ratio.clamp(0.8, 1.2) * advantages,
        )[active_mask].mean()
        critic_loss = F.mse_loss(
            critic(state).reshape(1), returns[active_mask].mean().reshape(1)
        )
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        actor_optimizer.step()
        critic_optimizer.step()
        self.assertTrue(torch.isfinite(actor_loss))
        self.assertTrue(torch.isfinite(critic_loss))


if __name__ == "__main__":
    unittest.main(verbosity=2)
