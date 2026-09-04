"""Contract tests for canonical and legacy LEO method variants."""

from __future__ import annotations

import hashlib
import json
import unittest

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from leo_marl_env import EnvConfig, LinkState, SCENARIOS
from leo_multiagent_env import (
    MultiAgentConfig,
    SynchronousLeoMultiAgentEnv,
    first_feasible_actions,
)
from variant_definitions import (
    FLAT_CRITIC,
    NO_CREDIT,
    NO_PACKET_CONTEXT,
    NO_QUEUE,
    PLANNED_CONTRASTS,
    PLANNED_CONTRAST_BY_NAME,
    PROPOSED,
    WITH_AVOIDABLE_SWITCH_REWARD,
    WITH_CONGESTION_CONTEXT,
    WITH_HARD_LIFETIME_MASK,
    WITH_LIFETIME_FEATURE,
    WITH_LIFETIME_REWARD,
    resolve_variant,
)


def changed_flags(left, right):
    left_flags = left.as_dict()
    right_flags = right.as_dict()
    left_flags.pop("name")
    right_flags.pop("name")
    return {
        key
        for key, value in left_flags.items()
        if right_flags[key] != value
    }


class VariantDefinitionTests(unittest.TestCase):
    def test_default_and_legacy_names_resolve_to_canonical_definitions(self):
        self.assertEqual(MultiAgentConfig().variant, "proposed")
        self.assertIs(resolve_variant("no_lifetime"), PROPOSED)
        self.assertIs(resolve_variant("full"), WITH_HARD_LIFETIME_MASK)
        self.assertIs(resolve_variant("no_ppo_protection"), PROPOSED)
        with self.assertRaisesRegex(ValueError, "unknown LEO variant"):
            resolve_variant("proposd")

    def test_lifetime_chain_adds_exactly_one_flag_per_step(self):
        self.assertEqual(
            changed_flags(PROPOSED, WITH_LIFETIME_FEATURE),
            {"lifetime_feature"},
        )
        self.assertEqual(
            changed_flags(WITH_LIFETIME_FEATURE, WITH_LIFETIME_REWARD),
            {"lifetime_reward"},
        )
        self.assertEqual(
            changed_flags(WITH_LIFETIME_REWARD, WITH_HARD_LIFETIME_MASK),
            {"hard_lifetime_mask"},
        )

    def test_component_removals_share_the_proposed_lifetime_base(self):
        expected_changes = {
            NO_QUEUE.name: {"queue_features", "queue_reward"},
            NO_CREDIT.name: {"centered_local_credit"},
            NO_PACKET_CONTEXT.name: {"packet_context"},
            FLAT_CRITIC.name: {"graph_critic"},
        }
        for definition in (NO_QUEUE, NO_CREDIT, NO_PACKET_CONTEXT, FLAT_CRITIC):
            with self.subTest(variant=definition.name):
                self.assertEqual(
                    changed_flags(PROPOSED, definition),
                    expected_changes[definition.name],
                )
                self.assertFalse(definition.lifetime_feature)
                self.assertFalse(definition.lifetime_reward)
                self.assertFalse(definition.hard_lifetime_mask)

    def test_congestion_context_is_an_unplanned_experimental_variant(self):
        self.assertIs(
            resolve_variant("with_congestion_context"),
            WITH_CONGESTION_CONTEXT,
        )
        self.assertEqual(
            changed_flags(PROPOSED, WITH_CONGESTION_CONTEXT),
            {"queue_trend_feature", "downstream_bottleneck_feature"},
        )
        frozen_methods = {
            method
            for contrast in PLANNED_CONTRASTS
            for method in (contrast.reference, contrast.treatment)
        }
        self.assertNotIn(WITH_CONGESTION_CONTEXT.name, frozen_methods)

    def test_avoidable_switch_reward_inherits_congestion_context(self):
        self.assertIs(
            resolve_variant("with_avoidable_switch_reward"),
            WITH_AVOIDABLE_SWITCH_REWARD,
        )
        self.assertEqual(
            changed_flags(
                WITH_CONGESTION_CONTEXT,
                WITH_AVOIDABLE_SWITCH_REWARD,
            ),
            {"avoidable_switch_cost_only"},
        )
        self.assertTrue(WITH_AVOIDABLE_SWITCH_REWARD.queue_trend_feature)
        self.assertTrue(
            WITH_AVOIDABLE_SWITCH_REWARD.downstream_bottleneck_feature
        )
        frozen_spec = WITH_AVOIDABLE_SWITCH_REWARD.as_dict()
        self.assertIs(frozen_spec["avoidable_switch_cost_only"], True)
        self.assertEqual(
            json.loads(json.dumps(frozen_spec, sort_keys=True)),
            frozen_spec,
        )
        frozen_methods = {
            method
            for contrast in PLANNED_CONTRASTS
            for method in (contrast.reference, contrast.treatment)
        }
        self.assertNotIn(WITH_AVOIDABLE_SWITCH_REWARD.name, frozen_methods)

    def test_planned_contrasts_match_the_declared_flag_differences(self):
        for contrast in PLANNED_CONTRASTS:
            with self.subTest(contrast=contrast.name):
                reference = resolve_variant(contrast.reference)
                treatment = resolve_variant(contrast.treatment)
                if contrast.component_kind == "trainer_safeguard_package":
                    self.assertEqual(changed_flags(reference, treatment), set())
                    self.assertEqual(
                        set(contrast.changed_flags),
                        {
                            "actor_gradient_clipping",
                            "target_kl_early_stopping",
                            "advantage_normalization",
                        },
                    )
                else:
                    self.assertEqual(
                        changed_flags(reference, treatment),
                        set(contrast.changed_flags),
                    )

        component_contrasts = [
            contrast
            for contrast in PLANNED_CONTRASTS
            if contrast.family == "component_removal"
        ]
        self.assertTrue(component_contrasts)
        self.assertTrue(
            all(contrast.reference == "proposed" for contrast in component_contrasts)
        )

        queue_contrast = PLANNED_CONTRAST_BY_NAME[
            "remove_queue_mechanism_package"
        ]
        self.assertEqual(queue_contrast.component_kind, "mechanism_package")
        self.assertEqual(
            queue_contrast.changed_flags,
            ("queue_features", "queue_reward"),
        )

    def _make_env(self, variant: str) -> SynchronousLeoMultiAgentEnv:
        cfg = MultiAgentConfig(
            env=EnvConfig(seed=36, scenario=SCENARIOS["medium_load"]),
            initial_packets=1,
            exogenous_packets_per_slot=0,
            variant=variant,
            seed=36,
        )
        env = SynchronousLeoMultiAgentEnv(cfg)
        env.reset(seed=36, initial_pairs=[(1, 12, 0)])
        return env

    def test_lifetime_chain_changes_feature_reward_and_mask_in_order(self):
        environments = {
            name: self._make_env(name)
            for name in (
                "proposed",
                "with_lifetime_feature",
                "with_lifetime_reward",
                "with_hard_lifetime_mask",
            )
        }
        for env in environments.values():
            neighbor = env.base._neighbors(1)[0]
            env.graph[(1, neighbor)].t_rem = 0.5

        neighbor = environments["proposed"].base._neighbors(1)[0]
        feature = {
            name: env._candidate_features(env.packets[1], 1, neighbor)[6]
            for name, env in environments.items()
        }
        self.assertEqual(feature["proposed"], 0.0)
        self.assertGreater(feature["with_lifetime_feature"], 0.0)
        self.assertEqual(
            feature["with_lifetime_feature"], feature["with_lifetime_reward"]
        )
        self.assertEqual(
            feature["with_lifetime_reward"], feature["with_hard_lifetime_mask"]
        )

        reasons = {
            name: env._mask_reason(env.packets[1], 1, neighbor)
            for name, env in environments.items()
        }
        self.assertNotEqual(reasons["proposed"], "lifetime")
        self.assertNotEqual(reasons["with_lifetime_feature"], "lifetime")
        self.assertNotEqual(reasons["with_lifetime_reward"], "lifetime")
        self.assertEqual(reasons["with_hard_lifetime_mask"], "lifetime")

        rewards = {
            name: env._forward_reward(
                env.packets[1], 1, neighbor, env.graph[(1, neighbor)]
            )
            for name, env in environments.items()
        }
        self.assertAlmostEqual(rewards["proposed"], rewards["with_lifetime_feature"])
        self.assertLess(rewards["with_lifetime_reward"], rewards["with_lifetime_feature"])
        self.assertAlmostEqual(
            rewards["with_lifetime_reward"], rewards["with_hard_lifetime_mask"]
        )

    def test_legacy_names_are_behaviorally_identical_to_canonical_names(self):
        for legacy, canonical in (
            ("no_lifetime", "proposed"),
            ("full", "with_hard_lifetime_mask"),
        ):
            with self.subTest(legacy=legacy):
                legacy_env = self._make_env(legacy)
                canonical_env = self._make_env(canonical)
                self.assertEqual(legacy_env.cfg.variant, canonical)
                self.assertEqual(legacy_env.observe(), canonical_env.observe())
                self.assertEqual(legacy_env.global_state(), canonical_env.global_state())

    def test_congestion_context_only_appends_actor_and_critic_features(self):
        proposed = self._make_env("proposed")
        congestion = self._make_env("with_congestion_context")
        proposed_obs = proposed.observe()
        congestion_obs = congestion.observe()

        self.assertEqual(proposed.candidate_feature_dim, 26)
        self.assertEqual(congestion.candidate_feature_dim, 28)
        for old_agent, new_agent in zip(proposed_obs, congestion_obs):
            self.assertEqual(old_agent["neighbor_ids"], new_agent["neighbor_ids"])
            self.assertEqual(old_agent["action_mask"], new_agent["action_mask"])
            for old_row, new_row in zip(
                old_agent["candidate_features"],
                new_agent["candidate_features"],
            ):
                self.assertEqual(old_row, new_row[:26])
                if new_agent["hol_packet_id"] is None:
                    self.assertEqual(new_row[26:], [0.0, 0.0])

        proposed_state = proposed.global_state()
        congestion_state = congestion.global_state()
        self.assertEqual(proposed_state["schema"]["node_feature_dim"], 25)
        self.assertEqual(congestion_state["schema"]["node_feature_dim"], 26)
        self.assertEqual(proposed_state["schema"]["edge_feature_dim"], 11)
        self.assertEqual(congestion_state["schema"]["edge_feature_dim"], 11)
        for old_node, new_node in zip(
            proposed_state["node_features"],
            congestion_state["node_features"],
        ):
            self.assertEqual(old_node, new_node[:25])

    def test_avoidable_switch_reward_excludes_forced_reroutes_only(self):
        def run(variant: str, switch_kind: str):
            env = self._make_env(variant)
            observation = env.observe()[0]
            feasible_actions = [
                action
                for action in range(1, len(observation["action_mask"]))
                if observation["action_mask"][action]
            ]
            self.assertGreaterEqual(len(feasible_actions), 2)
            selected_action = feasible_actions[0]
            if switch_kind == "avoidable":
                cached_action = feasible_actions[1]
                cached_hop = observation["neighbor_ids"][cached_action - 1]
            elif switch_kind == "forced":
                cached_hop = max(observation["neighbor_ids"]) + env.n_agents
            elif switch_kind == "none":
                cached_hop = None
            else:
                self.fail(f"unknown switch kind {switch_kind!r}")

            packet = env.packets[observation["hol_packet_id"]]
            if cached_hop is not None:
                env.route_cache[(1, packet.dst, packet.traffic_class)] = cached_hop
            actions = [0] * env.n_agents
            actions[0] = selected_action
            _, _, _, _, info = env.step(actions)
            self.assertIn(packet.packet_id, info["accepted"])
            return env, info

        aligned = {
            kind: run("with_avoidable_switch_reward", kind)
            for kind in ("none", "avoidable", "forced")
        }
        aligned_env, no_switch = aligned["none"]
        _, avoidable = aligned["avoidable"]
        _, forced = aligned["forced"]
        self.assertEqual(no_switch["reward_components"]["switch_cost"], 0.0)
        self.assertEqual(avoidable["reward_components"]["switch_cost"], 1.0)
        self.assertEqual(forced["reward_components"]["switch_cost"], 0.0)
        self.assertAlmostEqual(
            no_switch["local_rewards"][0] - avoidable["local_rewards"][0],
            aligned_env.cfg.env.w_switch,
        )
        self.assertAlmostEqual(
            forced["local_rewards"][0],
            no_switch["local_rewards"][0],
        )
        self.assertAlmostEqual(
            no_switch["global_reward"] - avoidable["global_reward"],
            aligned_env.cfg.global_switch_weight,
        )
        self.assertAlmostEqual(forced["global_reward"], no_switch["global_reward"])

        legacy = {
            kind: run("with_congestion_context", kind)
            for kind in ("none", "forced")
        }
        legacy_env, legacy_no_switch = legacy["none"]
        _, legacy_forced = legacy["forced"]
        self.assertEqual(legacy_forced["reward_components"]["switch_cost"], 1.0)
        self.assertAlmostEqual(
            legacy_no_switch["local_rewards"][0]
            - legacy_forced["local_rewards"][0],
            legacy_env.cfg.env.w_switch,
        )
        self.assertAlmostEqual(
            legacy_no_switch["global_reward"] - legacy_forced["global_reward"],
            legacy_env.cfg.global_switch_weight,
        )

    def test_proposed_observation_and_transition_fingerprints_are_frozen(self):
        def digest(payload) -> str:
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        env = SynchronousLeoMultiAgentEnv.from_scenario("medium_load")
        observations, _ = env.reset(
            seed=66,
            initial_pairs=[
                (1, 12),
                (2, 13),
                (3, 14),
                (4, 15),
                (5, 16),
                (6, 17),
            ],
        )
        self.assertEqual(env.candidate_feature_dim, 26)
        self.assertEqual(
            digest(observations),
            "5a57b0b047cb460901c04637f676df8ed3ce1a720f45544c9675d4daef032c8f",
        )
        self.assertEqual(
            digest(env.global_state()),
            "7203b01343c77e97f5da027932478a639a9ad5ad1b2d751941499149f285f3d0",
        )
        for _ in range(3):
            observations, _, _, _, _ = env.step(
                first_feasible_actions(observations)
            )
        self.assertEqual(
            env.state_digest(),
            "6096a2845c0fa7d5ddee303154fc1952a85edf1d022be36b9a1580377764452e",
        )
        self.assertEqual(
            env.trace_hash(),
            "ab7150b6effaf57803754ef375277a5c60285cf4e7035de03ffa67f09d7f0d55",
        )

    def test_queue_trend_uses_one_frozen_snapshot_without_observe_side_effects(self):
        env = self._make_env("with_congestion_context")
        observations = env.observe()
        active = observations[0]
        self.assertTrue(all(row[26] == 0.0 for row in active["candidate_features"]))

        candidate = active["neighbor_ids"][0]
        candidate_index = active["neighbor_ids"].index(candidate)
        env.previous_queue_lengths[candidate] = env.cfg.max_queue_packets
        decreasing = env.observe()[0]["candidate_features"][candidate_index][26]
        self.assertEqual(decreasing, -1.0)

        env.previous_queue_lengths[candidate] = len(env.queues[candidate])
        destination = 1 if candidate != 1 else 2
        env._create_packet(candidate, destination, traffic_class=0)
        env._create_packet(candidate, destination, traffic_class=0)
        expected = 2.0 / env.cfg.max_queue_packets
        first = env.observe()[0]["candidate_features"][candidate_index][26]
        history = list(env.previous_queue_lengths)
        second = env.observe()[0]["candidate_features"][candidate_index][26]
        env.global_state()
        self.assertAlmostEqual(first, expected)
        self.assertAlmostEqual(second, expected)
        self.assertEqual(env.previous_queue_lengths, history)

        frozen_lengths = env._queue_lengths()
        env.step(first_feasible_actions(env.observe()))
        self.assertEqual(env.previous_queue_lengths, frozen_lengths)
        advanced_history = list(env.previous_queue_lengths)
        env.observe()
        env.global_state()
        self.assertEqual(env.previous_queue_lengths, advanced_history)

        reset_obs, _ = env.reset(seed=36, initial_pairs=[(1, 12, 0)])
        self.assertTrue(
            all(row[26] == 0.0 for row in reset_obs[0]["candidate_features"])
        )

    def test_downstream_headroom_uses_one_real_feasible_continuation(self):
        def link(capacity: float, used: float, available: bool = True) -> LinkState:
            return LinkState(
                delay_ms=1.0,
                capacity_mbps=capacity,
                used_rate_mbps=used,
                rho=used / capacity,
                reliability=0.99,
                p_out=0.01,
                t_rem=30.0,
                available=available,
            )

        links = {
            (1, 2): link(20.0, 0.0),
            (2, 1): link(20.0, 0.0),
            (2, 3): link(20.0, 8.0),
            (2, 4): link(40.0, 24.0),
            (2, 5): link(10.0, 12.0),
        }
        env_cfg = EnvConfig(
            n_planes=1,
            sats_per_plane=5,
            seed=41,
            scenario=SCENARIOS["medium_load"],
            topology_provider=lambda _slot, _env: links,
        )
        cfg = MultiAgentConfig(
            env=env_cfg,
            max_queue_packets=10,
            initial_packets=1,
            exogenous_packets_per_slot=0,
            variant="with_congestion_context",
            seed=41,
        )
        env = SynchronousLeoMultiAgentEnv(cfg)
        env.reset(seed=41, initial_pairs=[(1, 5, 0)])
        env._create_packet(3, 5, traffic_class=0)
        env._create_packet(3, 5, traffic_class=0)
        env._create_packet(4, 5, traffic_class=0)

        observation = env.observe()[0]
        candidate_index = observation["neighbor_ids"].index(2)
        candidate_row = observation["candidate_features"][candidate_index]
        self.assertAlmostEqual(candidate_row[27], 0.6)

        env._create_packet(5, 1, traffic_class=0)
        remote_row = env.observe()[0]["candidate_features"][candidate_index]
        self.assertEqual(candidate_row, remote_row)

        links[(2, 3)].available = False
        self.assertAlmostEqual(
            env.observe()[0]["candidate_features"][candidate_index][27],
            0.4,
        )
        links[(2, 4)].available = False
        self.assertEqual(
            env.observe()[0]["candidate_features"][candidate_index][27],
            0.0,
        )
        self.assertEqual(env._continuation_headroom(env.packets[1], 5), 1.0)

    def test_wrapper_exposes_canonical_spec_and_critic_choice(self):
        wrapper = CleanMARLLeoMultiAgentWrapper(seed=9)
        self.assertEqual(wrapper.variant, "proposed")
        self.assertEqual(wrapper.get_variant_spec(), PROPOSED.as_dict())
        wrapper.reset(seed=9)
        self.assertIsNotNone(wrapper.get_critic_spec())

        flat = CleanMARLLeoMultiAgentWrapper(seed=9, variant="flat_critic")
        flat.reset(seed=9)
        self.assertIsNone(flat.get_critic_spec())

        congestion = CleanMARLLeoMultiAgentWrapper(
            seed=9,
            variant="with_congestion_context",
        )
        observation, _ = congestion.reset(seed=9)
        self.assertEqual(congestion.get_candidate_feature_dim(), 28)
        self.assertEqual(
            observation.shape,
            (congestion.n_agents, congestion.get_action_size() * 28),
        )
        self.assertEqual(congestion.get_critic_spec()["node_feature_dim"], 26)


if __name__ == "__main__":
    unittest.main()
