"""Unit and source-contract tests for the ns-3 closed-loop bridge."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import ns3_closed_loop_server as closed_loop
import run_ns3_closed_loop as runner


ROOT = Path(__file__).resolve().parent.parent


class EpisodeFeatureContextTests(unittest.TestCase):
    def test_advance_rebuilds_current_slot_graph_and_preserves_link_load(self):
        seed = 41001
        for scenario, marker_name in (
            ("frequent_break", "short_trem_links"),
            ("fault_links", "fault_links"),
        ):
            with self.subTest(scenario=scenario):
                ctx = closed_loop.EpisodeFeatureContext(scenario, seed)
                reference_wrapper = closed_loop.make_wrapper(scenario, seed)
                reference_wrapper.reset(seed=seed)
                reference = reference_wrapper.env

                markers = set(getattr(ctx.env.base, marker_name))
                self.assertTrue(markers)
                self.assertEqual(markers, getattr(reference.base, marker_name))
                edge_key = min(markers & set(ctx.env.graph))
                slot_edges = []

                for slot, link_tx in ((1, {}), (2, {edge_key: 3})):
                    ctx.advance_to_slot(slot, link_tx)
                    reference.slot = slot
                    for u in ctx.used_rate:
                        for v, rate in ctx.used_rate[u].items():
                            reference.used_rate[u][v] = rate
                    reference._refresh_graph()

                    self.assertEqual(ctx.env.slot, slot)
                    self.assertEqual(ctx.env.base.time_slot, slot)
                    self.assertEqual(
                        ctx.env.base.fault_links,
                        reference.base.fault_links,
                    )
                    self.assertEqual(
                        ctx.env.base.short_trem_links,
                        reference.base.short_trem_links,
                    )
                    self.assertEqual(set(getattr(ctx.env.base, marker_name)), markers)
                    self.assertIn(edge_key, ctx.env.graph)
                    actual = vars(ctx.env.graph[edge_key])
                    expected = vars(reference.graph[edge_key])
                    self.assertEqual(actual, expected)
                    slot_edges.append(dict(actual))

                expected_load = (
                    3
                    * ctx.env.cfg.env.packet_demand_mbps
                    * ctx.env.cfg.env.load_decay
                )
                u, v = edge_key
                self.assertAlmostEqual(ctx.used_rate[u][v], expected_load)
                self.assertAlmostEqual(ctx.env.used_rate[u][v], expected_load)
                self.assertAlmostEqual(
                    ctx.env.graph[edge_key].used_rate_mbps,
                    expected_load,
                )
                self.assertNotEqual(
                    slot_edges[0]["delay_ms"],
                    slot_edges[1]["delay_ms"],
                )


class ClosedLoopDynamicTopologyTests(unittest.TestCase):
    def test_slot_10_missing_seam_candidate_is_masked_without_slot_remap(self):
        class RecordingBridge:
            def decide(self, message):
                self.message = message
                decisions = []
                for agent in message["agents"]:
                    action = max(
                        index
                        for index, feasible in enumerate(agent["action_mask"])
                        if feasible
                    )
                    decisions.append({
                        "agent_id": agent["agent_id"],
                        "action_slot": action,
                        "next_hop_id": agent["candidate_next_hops"][action],
                    })
                return {"decisions": decisions}

        seed = 41001
        context = closed_loop.EpisodeFeatureContext("medium_load", seed)
        bridge = RecordingBridge()
        server = object.__new__(closed_loop.ClosedLoopServer)
        server.bridge = bridge
        server.scenario = "medium_load"
        server.seeds = [seed]
        server.contexts = {("mappo", 0): context}
        server.current_policy = "mappo"
        server.dijkstra = closed_loop.DijkstraProvider(server.contexts)
        server.feature_dim = context.env.candidate_feature_dim
        server.action_size = context.env.action_size

        static_slots = [0, 1, 5, 12, 24, 0, 0]
        lines = []
        for sat in range(1, context.n_agents + 1):
            if sat == 6:
                lines.append(
                    "AG 6 1 0 1 12 0 0 0 1 32 "
                    + " ".join(str(value) for value in static_slots)
                )
            else:
                lines.append(
                    f"AG {sat} 0 0 -1 0 0 0 0 0 0 "
                    + " ".join("0" for _ in range(server.action_size))
                )

        decisions = server.handle_slot(
            ["SLOT", "0", "10", str(context.n_agents), "0"],
            lines,
            [],
        )

        self.assertNotIn((6, 24), context.env.graph)
        agent = bridge.message["agents"][5]
        self.assertEqual(agent["candidate_next_hops"], static_slots)
        self.assertFalse(agent["action_mask"][4])
        self.assertEqual(
            agent["candidate_features"][4],
            [0.0] * server.feature_dim,
        )
        self.assertTrue(agent["action_mask"][3])
        self.assertNotEqual(
            agent["candidate_features"][3],
            [0.0] * server.feature_dim,
        )
        self.assertEqual(decisions[5], (6, 3, 12))
        self.assertEqual(context.env.route_cache[(6, 12, 0)], 12)


class ClosedLoopContextIsolationTests(unittest.TestCase):
    def make_server(self):
        server = object.__new__(closed_loop.ClosedLoopServer)
        server.scenario = "medium_load"
        server.seeds = [41001, 41001]
        server.contexts = {}
        server.current_policy = None
        return server

    def test_policy_and_episode_contexts_are_independent(self):
        instances = []

        class FakeContext:
            def __init__(self, scenario, seed):
                self.scenario = scenario
                self.seed = seed
                self.advance_calls = []
                self.used_rate = {"sentinel": 0}
                self.env = types.SimpleNamespace(route_cache={})
                instances.append(self)

            def advance_to_slot(self, slot, link_tx):
                self.advance_calls.append((slot, dict(link_tx)))

        server = self.make_server()
        with mock.patch.object(closed_loop, "EpisodeFeatureContext", FakeContext):
            server.begin_policy("mappo")
            first = server._advance_context(0, 1, {(1, 2): 99})
            same = server._advance_context(0, 2, {(1, 2): 2})
            second_episode = server._advance_context(1, 1, {(2, 3): 88})

            self.assertIs(first, same)
            self.assertIsNot(first, second_episode)
            self.assertEqual(first.seed, second_episode.seed)
            self.assertEqual(first.advance_calls, [(1, {}), (2, {(1, 2): 2})])
            self.assertEqual(second_episode.advance_calls, [(1, {})])

            first.used_rate["sentinel"] = 7
            first.env.route_cache[(1, 2, 0)] = 3
            server.begin_policy("dijkstra")
            fresh_policy = server._advance_context(0, 1, {(3, 4): 77})

        self.assertIsNot(first, fresh_policy)
        self.assertEqual(fresh_policy.used_rate, {"sentinel": 0})
        self.assertEqual(fresh_policy.env.route_cache, {})
        self.assertEqual(fresh_policy.advance_calls, [(1, {})])
        self.assertEqual(len(instances), 3)

    def test_each_connection_begins_a_clean_policy_run(self):
        class Duplex:
            def __init__(self):
                self.lines = iter((b"HELLO 24 7 mappo\n", b"BYE\n"))
                self.writes = []

            def readline(self):
                return next(self.lines, b"")

            def write(self, value):
                self.writes.append(value)

            def flush(self):
                pass

            def close(self):
                pass

        stream = Duplex()
        conn = types.SimpleNamespace(makefile=lambda *args, **kwargs: stream)
        server = self.make_server()
        server.contexts[("old", 0)] = object()

        server.serve_connection(conn)

        self.assertEqual(server.current_policy, "mappo")
        self.assertEqual(server.contexts, {})
        self.assertEqual(stream.writes, [b"READY 24 7\n"])


class PacketTraceContractTests(unittest.TestCase):
    def test_frozen_traces_bind_initial_and_step_end_packet_ids(self):
        seeds = [21001, 21002, 21003, 21004, 21005]
        for policy in ("mappo", "dijkstra"):
            with self.subTest(policy=policy):
                result = runner.validate_packet_trace_contract(
                    ROOT / "experiments" / "ns3-replay" / f"packets_{policy}.csv",
                    num_episodes=5,
                    initial_packets=12,
                    exogenous_packets_per_slot=6,
                    episode_slots=30,
                )
                self.assertEqual(result["packets"], 960)
                self.assertEqual(result["initial_packets_per_episode"], 12)
                self.assertEqual(result["expected_global_packet_ids"][0], 1)
                self.assertEqual(
                    result["expected_global_packet_ids"][192],
                    1_000_001,
                )
                self.assertEqual(len(result["sha256"]), 64)
                self.assertEqual(
                    result["expected_global_packet_ids_sha256"],
                    runner.packet_id_set_sha256(
                        result["expected_global_packet_ids"]
                    ),
                )
                manifest = runner.validate_traffic_manifest(
                    ROOT
                    / "experiments"
                    / "ns3-replay"
                    / f"env_summary_{policy}.csv",
                    workload_seeds=seeds,
                    initial_packets=12,
                    exogenous_packets_per_slot=6,
                    episode_slots=30,
                    source_policy=policy,
                    source_scenario="medium_load",
                )
                self.assertEqual(manifest["workload_seeds"], seeds)
                self.assertEqual(manifest["generated_total"], 960)
                self.assertEqual(manifest["source_policy"], policy)
                self.assertEqual(manifest["source_scenario"], "medium_load")
                self.assertFalse(manifest["manifest_scenario_column"])

    def test_trace_contract_rejects_wrong_exogenous_admission_slot(self):
        trace = (
            "episode,packet_id,src,dst,traffic_class,created_slot\n"
            "0,1,1,2,0,1\n"
            "0,2,2,3,1,1\n"
            "0,3,3,4,2,2\n"
            "0,4,4,5,0,2\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packets.csv"
            path.write_text(trace, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "packet 3.*expected 1"):
                runner.validate_packet_trace_contract(
                    path,
                    num_episodes=1,
                    initial_packets=2,
                    exogenous_packets_per_slot=1,
                    episode_slots=2,
                )

    def test_trace_contract_rejects_schema_node_and_class_drift(self):
        header = "episode,packet_id,src,dst,traffic_class,created_slot\n"
        cases = (
            (
                "misordered_prefix",
                "episode,packet_id,dst,src,traffic_class,created_slot\n"
                "0,1,2,1,0,1\n",
                "first six columns",
            ),
            ("src_zero", header + "0,1,0,2,0,1\n", "node outside"),
            ("src_25", header + "0,1,25,2,0,1\n", "node outside"),
            ("dst_zero", header + "0,1,1,0,0,1\n", "node outside"),
            ("dst_25", header + "0,1,1,25,0,1\n", "node outside"),
            ("same_endpoint", header + "0,1,2,2,0,1\n", "src equals dst"),
            ("class_3", header + "0,1,1,2,3,1\n", "traffic_class"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packets.csv"
            for name, contents, message in cases:
                with self.subTest(name=name):
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        runner.validate_packet_trace_contract(
                            path,
                            num_episodes=1,
                            initial_packets=1,
                            exogenous_packets_per_slot=0,
                            episode_slots=1,
                        )


class ClosedLoopArtifactAuditTests(unittest.TestCase):
    @staticmethod
    def result_counts(**overrides):
        result = {
            "sent": 7,
            "delivered": 1,
            "deadline_drops": 1,
            "ttl_drops": 1,
            "queue_drops": 1,
            "source_drops": 1,
            "device_queue_drops": 1,
            "truncated_backlog": 1,
            "mean_delay_ms": 2.5,
            "p50_delay_ms": 2.5,
            "p95_delay_ms": 2.5,
        }
        result.update(overrides)
        if "delivery_ratio" not in overrides:
            sent = result["sent"]
            result["delivery_ratio"] = result["delivered"] / sent if sent else 0.0
        return result

    def test_output_audit_accepts_exact_terminal_conservation(self):
        output = (
            "packet_id,delivered,delay_ms,drop_reason\n"
            "1,1,2.5,delivered\n"
            "2,0,-1,deadline_exceeded\n"
            "3,0,-1,ttl_exceeded\n"
            "4,0,-1,queue_overflow\n"
            "1000001,0,-1,source_queue_overflow\n"
            "1000002,0,-1,device_queue_full\n"
            "1000003,0,-1,backlog\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packets.csv"
            path.write_text(output, encoding="utf-8")
            audit = runner.validate_closed_loop_output(
                path,
                self.result_counts(),
                expected_sent=7,
                expected_packet_ids=[1, 2, 3, 4, 1_000_001, 1_000_002, 1_000_003],
            )
            output_sha256 = runner.sha256_file(path)

        self.assertEqual(audit["rows"], 7)
        self.assertEqual(audit["unique_packet_ids"], 7)
        self.assertEqual(audit["terminal_counts"]["delivered"], 1)
        self.assertEqual(audit["terminal_counts"]["backlog"], 1)
        self.assertEqual(audit["sha256"], output_sha256)
        self.assertEqual(
            audit["packet_id_set_sha256"],
            runner.packet_id_set_sha256(
                [1, 2, 3, 4, 1_000_001, 1_000_002, 1_000_003]
            ),
        )

    def test_output_audit_rejects_inconsistent_terminal_artifacts(self):
        header = "packet_id,delivered,delay_ms,drop_reason\n"
        cases = (
            (
                "delivered_as_backlog",
                header + "1,1,2.5,backlog\n",
                self.result_counts(
                    sent=1,
                    deadline_drops=0,
                    ttl_drops=0,
                    queue_drops=0,
                    source_drops=0,
                    device_queue_drops=0,
                    truncated_backlog=0,
                ),
                "flag/reason mismatch",
                1,
            ),
            (
                "duplicate_global_id",
                header + "1,1,2.5,delivered\n1,1,3.5,delivered\n",
                self.result_counts(
                    sent=2,
                    delivered=2,
                    deadline_drops=0,
                    ttl_drops=0,
                    queue_drops=0,
                    source_drops=0,
                    device_queue_drops=0,
                    truncated_backlog=0,
                ),
                "duplicate global packet_id",
                2,
            ),
            (
                "unknown_reason",
                header + "1,0,-1,mystery\n",
                self.result_counts(sent=1),
                "unknown terminal reason",
                1,
            ),
            (
                "result_drift",
                header + "1,1,2.5,delivered\n",
                self.result_counts(
                    sent=1,
                    delivered=0,
                    deadline_drops=0,
                    ttl_drops=0,
                    queue_drops=0,
                    source_drops=0,
                    device_queue_drops=0,
                    truncated_backlog=1,
                ),
                "terminal count mismatch",
                1,
            ),
            (
                "trace_count_drift",
                header + "1,1,2.5,delivered\n",
                self.result_counts(
                    sent=1,
                    deadline_drops=0,
                    ttl_drops=0,
                    queue_drops=0,
                    source_drops=0,
                    device_queue_drops=0,
                    truncated_backlog=0,
                ),
                "trace declares 2",
                2,
                None,
            ),
            (
                "wrong_global_id",
                header + "2,1,2.5,delivered\n",
                self.result_counts(
                    sent=1,
                    deadline_drops=0,
                    ttl_drops=0,
                    queue_drops=0,
                    source_drops=0,
                    device_queue_drops=0,
                    truncated_backlog=0,
                ),
                "global packet_id set mismatch",
                1,
                [1],
            ),
            (
                "non_delivery_delay",
                header + "1,0,-2,deadline_exceeded\n",
                self.result_counts(
                    sent=1,
                    delivered=0,
                    deadline_drops=1,
                    ttl_drops=0,
                    queue_drops=0,
                    source_drops=0,
                    device_queue_drops=0,
                    truncated_backlog=0,
                    mean_delay_ms=-1,
                    p50_delay_ms=-1,
                    p95_delay_ms=-1,
                ),
                "delay must equal -1",
                1,
                [1],
            ),
            (
                "delivery_ratio",
                header + "1,1,2.5,delivered\n",
                self.result_counts(
                    sent=1,
                    deadline_drops=0,
                    ttl_drops=0,
                    queue_drops=0,
                    source_drops=0,
                    device_queue_drops=0,
                    truncated_backlog=0,
                    delivery_ratio=0.9,
                ),
                "delivery_ratio mismatch",
                1,
                [1],
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packets.csv"
            normalized_cases = []
            for case in cases:
                normalized_cases.append(case if len(case) == 6 else (*case, None))
            for (
                name,
                output,
                result,
                message,
                expected_sent,
                expected_packet_ids,
            ) in normalized_cases:
                with self.subTest(name=name):
                    path.write_text(output, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        runner.validate_closed_loop_output(
                            path,
                            result,
                            expected_sent=expected_sent,
                            expected_packet_ids=expected_packet_ids,
                        )

    def test_output_audit_recomputes_cpp_delay_statistics(self):
        header = "packet_id,delivered,delay_ms,drop_reason\n"
        output = header + "".join(
            f"{packet_id},1,{delay},delivered\n"
            for packet_id, delay in enumerate((10, 1, 4, 7, 2), start=1)
        )
        result = self.result_counts(
            sent=5,
            delivered=5,
            deadline_drops=0,
            ttl_drops=0,
            queue_drops=0,
            source_drops=0,
            device_queue_drops=0,
            truncated_backlog=0,
            mean_delay_ms=4.8,
            p50_delay_ms=4,
            p95_delay_ms=7,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packets.csv"
            path.write_text(output, encoding="utf-8")
            audit = runner.validate_closed_loop_output(
                path,
                result,
                expected_sent=5,
                expected_packet_ids=range(1, 6),
            )
            self.assertEqual(
                audit["delay_metrics"],
                {
                    "mean_delay_ms": 4.8,
                    "p50_delay_ms": 4.0,
                    "p95_delay_ms": 7.0,
                },
            )

            for field in ("mean_delay_ms", "p50_delay_ms", "p95_delay_ms"):
                with self.subTest(field=field):
                    wrong_result = dict(result)
                    wrong_result[field] += 1
                    with self.assertRaisesRegex(ValueError, field):
                        runner.validate_closed_loop_output(
                            path,
                            wrong_result,
                            expected_packet_ids=range(1, 6),
                        )

    def test_output_audit_accepts_empty_delivered_delay_set(self):
        result = self.result_counts(
            sent=1,
            delivered=0,
            deadline_drops=1,
            ttl_drops=0,
            queue_drops=0,
            source_drops=0,
            device_queue_drops=0,
            truncated_backlog=0,
            mean_delay_ms=-1,
            p50_delay_ms=-1,
            p95_delay_ms=-1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packets.csv"
            path.write_text(
                "packet_id,delivered,delay_ms,drop_reason\n"
                "1,0,-1,deadline_exceeded\n",
                encoding="utf-8",
            )
            audit = runner.validate_closed_loop_output(
                path,
                result,
                expected_packet_ids=[1],
            )
        self.assertEqual(
            audit["delay_metrics"],
            {"mean_delay_ms": -1.0, "p50_delay_ms": -1.0, "p95_delay_ms": -1.0},
        )

    def test_result_extraction_requires_one_matching_closed_loop_record(self):
        valid = "RESULT,policy=mappo,closed_loop=1,sent=1\n"
        self.assertEqual(
            runner.extract_single_result(valid, "mappo")["sent"],
            1,
        )
        cases = (
            ("missing", "diagnostic only\n", "exactly one RESULT"),
            ("duplicate", valid + valid, "observed 2"),
            (
                "wrong_policy",
                "RESULT,policy=dijkstra,closed_loop=1\n",
                "expected 'mappo'",
            ),
            (
                "not_closed_loop",
                "RESULT,policy=mappo,closed_loop=0\n",
                "closed_loop must equal 1",
            ),
        )
        for name, stdout, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    runner.extract_single_result(stdout, "mappo")

    def test_traffic_manifest_and_run_directory_are_bound_and_immutable(self):
        self.assertEqual(
            runner.derive_traffic_manifest_path(Path("packets_mappo.csv")),
            Path("env_summary_mappo.csv"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packets = root / "packets_mappo.csv"
            packets.write_text(
                "episode,packet_id,src,dst,traffic_class,created_slot\n"
                "0,1,1,2,0,1\n"
                "1,1,3,4,1,1\n",
                encoding="utf-8",
            )
            traffic = root / "env_summary_mappo.csv"
            traffic.write_text(
                "episode,policy,scenario,workload_seed,generated\n"
                "0,mappo,medium_load,101,1\n"
                "1,mappo,medium_load,102,1\n",
                encoding="utf-8",
            )
            trace_contract = runner.validate_packet_trace_contract(
                packets,
                num_episodes=2,
                initial_packets=1,
                exogenous_packets_per_slot=0,
                episode_slots=1,
            )
            traffic_contract = runner.validate_traffic_manifest(
                traffic,
                workload_seeds=[101, 102],
                initial_packets=1,
                exogenous_packets_per_slot=0,
                episode_slots=1,
                source_policy="mappo",
                source_scenario="medium_load",
            )
            self.assertEqual(
                trace_contract["expected_global_packet_ids"],
                [1, 1_000_001],
            )
            self.assertEqual(
                trace_contract["expected_global_packet_ids_sha256"],
                runner.packet_id_set_sha256([1, 1_000_001]),
            )
            self.assertTrue(traffic_contract["manifest_scenario_column"])
            outdir = root / "audited-v2"
            runner.create_new_output_directory(outdir)
            frozen_packets = runner.freeze_input_file(
                packets,
                outdir / "input_packets.csv",
                trace_contract["sha256"],
            )
            frozen_traffic = runner.freeze_input_file(
                traffic,
                outdir / "traffic_manifest.csv",
                traffic_contract["sha256"],
            )
            self.assertEqual(frozen_packets["sha256"], runner.sha256_file(packets))
            self.assertEqual(frozen_traffic["sha256"], runner.sha256_file(traffic))
            checkpoint = root / "validation_best.pt"
            checkpoint.write_bytes(b"checkpoint")
            manifest_path = outdir / "run_manifest.json"
            runner.write_new_run_manifest(
                manifest_path,
                {
                    "status": "started",
                    "policies": ["mappo"],
                    "traffic_source": {
                        "policy": "mappo",
                        "scenario": "medium_load",
                    },
                    "checkpoint": {
                        "path": str(checkpoint.resolve()),
                        "sha256": runner.sha256_file(checkpoint),
                    },
                    "packet_trace": {
                        **trace_contract,
                        "frozen_copy": frozen_packets,
                    },
                    "traffic_manifest": traffic_contract,
                },
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["packet_trace"]["sha256"],
                runner.sha256_file(packets),
            )
            self.assertEqual(
                manifest["traffic_manifest"]["sha256"],
                runner.sha256_file(traffic),
            )
            self.assertEqual(manifest["status"], "started")
            self.assertEqual(
                manifest["checkpoint"]["sha256"],
                runner.sha256_file(checkpoint),
            )
            with self.assertRaises(FileExistsError):
                runner.create_new_output_directory(outdir)
            with self.assertRaises(FileExistsError):
                runner.write_new_run_manifest(manifest_path, manifest)

            packet_output = outdir / "clpkt_mappo.csv"
            invalid_output = (
                "packet_id,delivered,delay_ms,drop_reason\n"
                "1,1,2.5,delivered\n"
                "1000002,0,-1,deadline_exceeded\n"
            )
            result = self.result_counts(
                sent=2,
                delivered=1,
                deadline_drops=1,
                ttl_drops=0,
                queue_drops=0,
                source_drops=0,
                device_queue_drops=0,
                truncated_backlog=0,
            )
            packet_output.write_text(invalid_output, encoding="utf-8")
            completion_path = outdir / "completion_manifest.json"
            with self.assertRaisesRegex(ValueError, "global packet_id set mismatch"):
                runner.validate_closed_loop_output(
                    packet_output,
                    result,
                    expected_packet_ids=trace_contract[
                        "expected_global_packet_ids"
                    ],
                )
            self.assertFalse(completion_path.exists())

            valid_output = invalid_output.replace("1000002", "1000001")
            packet_output.write_text(valid_output, encoding="utf-8")
            packet_audit = runner.validate_closed_loop_output(
                packet_output,
                result,
                expected_sent=2,
                expected_packet_ids=trace_contract["expected_global_packet_ids"],
            )
            summary_path = outdir / "closedloop_summary.csv"
            summary_path.write_text(
                "policy,sent,delivered\nmappo,2,1\n",
                encoding="utf-8",
            )
            summary_sha256 = runner.sha256_file(summary_path)

            packet_output.write_text(valid_output + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                runner.write_completion_manifest(
                    completion_path,
                    run_manifest_path=manifest_path,
                    results={"mappo": result},
                    packet_outputs={"mappo": packet_output},
                    packet_audits={"mappo": packet_audit},
                    summary_path=summary_path,
                    summary_sha256=summary_sha256,
                )
            self.assertFalse(completion_path.exists())

            packet_output.write_text(valid_output, encoding="utf-8")
            packet_audit = runner.validate_closed_loop_output(
                packet_output,
                result,
                expected_packet_ids=trace_contract["expected_global_packet_ids"],
            )
            summary_path.write_text(
                "policy,sent,delivered\nmappo,2,1\n\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "closed-loop summary SHA-256"):
                runner.write_completion_manifest(
                    completion_path,
                    run_manifest_path=manifest_path,
                    results={"mappo": result},
                    packet_outputs={"mappo": packet_output},
                    packet_audits={"mappo": packet_audit},
                    summary_path=summary_path,
                    summary_sha256=summary_sha256,
                )
            self.assertFalse(completion_path.exists())
            summary_path.write_text(
                "policy,sent,delivered\nmappo,2,1\n",
                encoding="utf-8",
            )
            completion = runner.write_completion_manifest(
                completion_path,
                run_manifest_path=manifest_path,
                results={"mappo": result},
                packet_outputs={"mappo": packet_output},
                packet_audits={"mappo": packet_audit},
                summary_path=summary_path,
                summary_sha256=summary_sha256,
            )
            self.assertEqual(completion["status"], "complete")
            self.assertEqual(
                completion["run_manifest"]["sha256"],
                runner.sha256_file(manifest_path),
            )
            self.assertEqual(completion["results"]["mappo"], result)
            self.assertEqual(
                completion["packet_outputs"]["mappo"]["sha256"],
                runner.sha256_file(packet_output),
            )
            self.assertEqual(
                completion["summary"]["sha256"],
                runner.sha256_file(summary_path),
            )
            self.assertEqual(
                completion["traffic_source"],
                {"policy": "mappo", "scenario": "medium_load"},
            )
            with self.assertRaises(FileExistsError):
                runner.write_completion_manifest(
                    completion_path,
                    run_manifest_path=manifest_path,
                    results={"mappo": result},
                    packet_outputs={"mappo": packet_output},
                    packet_audits={"mappo": packet_audit},
                    summary_path=summary_path,
                    summary_sha256=summary_sha256,
                )

    def test_completion_rejects_a_subset_of_requested_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_manifest_path = root / "run_manifest.json"
            runner.write_new_run_manifest(
                run_manifest_path,
                {
                    "status": "started",
                    "policies": ["mappo", "dijkstra"],
                    "traffic_source": {
                        "policy": "mappo",
                        "scenario": "medium_load",
                    },
                    "traffic_manifest": {
                        "source_policy": "mappo",
                        "source_scenario": "medium_load",
                    },
                },
            )
            completion_path = root / "completion_manifest.json"
            with self.assertRaisesRegex(ValueError, "policies do not match"):
                runner.write_completion_manifest(
                    completion_path,
                    run_manifest_path=run_manifest_path,
                    results={"mappo": {}},
                    packet_outputs={"mappo": root / "clpkt_mappo.csv"},
                    packet_audits={"mappo": {"sha256": "0" * 64}},
                    summary_path=root / "closedloop_summary.csv",
                    summary_sha256="0" * 64,
                )
            self.assertFalse(completion_path.exists())

    def test_traffic_manifest_rejects_seed_and_generation_drift(self):
        cases = (
            (
                "seed_order",
                "episode,policy,workload_seed,generated\n"
                "0,mappo,102,1\n1,mappo,101,1\n",
                "workload_seed order",
            ),
            (
                "generated",
                "episode,policy,workload_seed,generated\n"
                "0,mappo,101,1\n1,mappo,102,2\n",
                "generated=2; expected 1",
            ),
            (
                "episode_order",
                "episode,policy,workload_seed,generated\n"
                "1,mappo,102,1\n0,mappo,101,1\n",
                "episodes must appear exactly",
            ),
            (
                "missing_policy",
                "episode,workload_seed,generated\n0,101,1\n1,102,1\n",
                "missing columns.*policy",
            ),
            (
                "unrelated_policy",
                "episode,policy,workload_seed,generated\n"
                "0,dijkstra,101,1\n1,dijkstra,102,1\n",
                "policy does not match source policy",
            ),
            (
                "wrong_scenario",
                "episode,policy,scenario,workload_seed,generated\n"
                "0,mappo,fault_links,101,1\n"
                "1,mappo,fault_links,102,1\n",
                "scenario does not match source scenario",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env_summary.csv"
            for name, contents, message in cases:
                with self.subTest(name=name):
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        runner.validate_traffic_manifest(
                            path,
                            workload_seeds=[101, 102],
                            initial_packets=1,
                            exogenous_packets_per_slot=0,
                            episode_slots=1,
                            source_policy="mappo",
                            source_scenario="medium_load",
                        )


class Ns3ProgramProvenanceTests(unittest.TestCase):
    def test_build_uses_frozen_source_and_records_actual_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen_source = Path(directory) / "ns3_closed_loop.cc"
            frozen_source.write_bytes(b"frozen C++ source\n")
            source_sha256 = runner.sha256_file(frozen_source)
            executable_sha256 = "b" * 64
            commands = []

            def fake_wsl_run(command, timeout=1800):
                del timeout
                commands.append(command)
                if "sha256sum" in command:
                    if runner.WSL_NS3_SCRATCH_SOURCE in command:
                        return f"{source_sha256}  {runner.WSL_NS3_SCRATCH_SOURCE}\n"
                    if runner.WSL_NS3_EXECUTABLE in command:
                        return (
                            f"{executable_sha256}  "
                            f"{runner.WSL_NS3_EXECUTABLE}\n"
                        )
                if "./ns3 build" in command:
                    return "build complete\n"
                if command.startswith("cp -- "):
                    return ""
                self.fail(f"unexpected WSL command: {command}")

            with mock.patch.object(runner, "wsl_run", side_effect=fake_wsl_run):
                program = runner.prepare_ns3_program(
                    frozen_source,
                    source_sha256,
                    skip_build=False,
                )

            copy_commands = [command for command in commands if command.startswith("cp -- ")]
            self.assertEqual(len(copy_commands), 1)
            self.assertIn(runner.to_wsl_path(frozen_source), copy_commands[0])
            self.assertNotIn(
                runner.to_wsl_path(runner.REPO / "src" / "ns3_closed_loop.cc"),
                copy_commands[0],
            )
            self.assertFalse(program["build_skipped"])
            self.assertEqual(
                program["frozen_source"],
                {
                    "path": str(frozen_source.resolve()),
                    "sha256": source_sha256,
                },
            )
            self.assertEqual(
                program["wsl_scratch_source"],
                {
                    "path": runner.WSL_NS3_SCRATCH_SOURCE,
                    "sha256": source_sha256,
                },
            )
            self.assertEqual(
                program["actual_executable"],
                {
                    "path": runner.WSL_NS3_EXECUTABLE,
                    "sha256": executable_sha256,
                },
            )

    def test_skip_build_rejects_source_drift_before_accepting_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen_source = Path(directory) / "ns3_closed_loop.cc"
            frozen_source.write_bytes(b"expected source\n")
            source_sha256 = runner.sha256_file(frozen_source)
            commands = []

            def fake_wsl_run(command, timeout=1800):
                del timeout
                commands.append(command)
                return f"{'f' * 64}  {runner.WSL_NS3_SCRATCH_SOURCE}\n"

            with mock.patch.object(runner, "wsl_run", side_effect=fake_wsl_run):
                with self.assertRaisesRegex(
                    ValueError,
                    "existing WSL scratch source.*SHA-256 changed",
                ):
                    runner.prepare_ns3_program(
                        frozen_source,
                        source_sha256,
                        skip_build=True,
                    )

            self.assertEqual(len(commands), 1)
            self.assertNotIn("./ns3 build", commands[0])
            self.assertNotIn("cp --", commands[0])
            self.assertNotIn(runner.WSL_NS3_EXECUTABLE, commands[0])

    def test_skip_build_rejects_a_missing_actual_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen_source = Path(directory) / "ns3_closed_loop.cc"
            frozen_source.write_bytes(b"expected source\n")
            source_sha256 = runner.sha256_file(frozen_source)

            def fake_wsl_run(command, timeout=1800):
                del timeout
                if runner.WSL_NS3_SCRATCH_SOURCE in command:
                    return f"{source_sha256}  {runner.WSL_NS3_SCRATCH_SOURCE}\n"
                if runner.WSL_NS3_EXECUTABLE in command:
                    raise RuntimeError("missing executable")
                self.fail(f"unexpected WSL command: {command}")

            with mock.patch.object(runner, "wsl_run", side_effect=fake_wsl_run):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "optimized executable.*missing or unreadable",
                ):
                    runner.prepare_ns3_program(
                        frozen_source,
                        source_sha256,
                        skip_build=True,
                    )

    def test_runtime_verification_rejects_binary_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen_source = Path(directory) / "ns3_closed_loop.cc"
            frozen_source.write_bytes(b"expected source\n")
            source_sha256 = runner.sha256_file(frozen_source)
            executable_sha256 = "b" * 64
            program = {
                "build_skipped": True,
                "frozen_source": {
                    "path": str(frozen_source.resolve()),
                    "sha256": source_sha256,
                },
                "wsl_scratch_source": {
                    "path": runner.WSL_NS3_SCRATCH_SOURCE,
                    "sha256": source_sha256,
                },
                "actual_executable": {
                    "path": runner.WSL_NS3_EXECUTABLE,
                    "sha256": executable_sha256,
                },
            }

            def fake_wsl_run(command, timeout=1800):
                del timeout
                if runner.WSL_NS3_SCRATCH_SOURCE in command:
                    return f"{source_sha256}  {runner.WSL_NS3_SCRATCH_SOURCE}\n"
                if runner.WSL_NS3_EXECUTABLE in command:
                    return f"{'c' * 64}  {runner.WSL_NS3_EXECUTABLE}\n"
                self.fail(f"unexpected WSL command: {command}")

            with mock.patch.object(runner, "wsl_run", side_effect=fake_wsl_run):
                with self.assertRaisesRegex(
                    ValueError,
                    "policy post-run ns-3 executable SHA-256 changed",
                ):
                    runner.verify_ns3_program_provenance(
                        program,
                        "policy post-run",
                    )


class Ns3SourceContractTests(unittest.TestCase):
    def cpp_source(self):
        return (ROOT / "src" / "ns3_closed_loop.cc").read_text(
            encoding="utf-8"
        )

    def test_cpp_counts_one_hop_and_records_the_sender(self):
        source = self.cpp_source()
        compact = " ".join(source.split())

        self.assertEqual(source.count("h.hops + 1"), 1)
        self.assertIn("h.hops = pr.pkt.hops;", compact)
        self.assertNotIn("h.hops = pr.pkt.hops + 1;", compact)
        self.assertIn("h.prevNode = (uint8_t)pr.node;", compact)
        self.assertIn("(uint8_t)hops, h.prevNode,", compact)
        self.assertNotIn("h.prevNode = pr.pkt.prevNode;", compact)

        on_rx = compact[
            compact.index("void OnRx(") : compact.index("static void AdmitInjections")
        ]
        self.assertLess(on_rx.index("g.delivered++;"), on_rx.index("if (DeadlineExceeded"))

    def test_cpp_counts_active_arrivals_before_every_terminal_branch(self):
        compact = " ".join(self.cpp_source().split())
        on_rx = compact[
            compact.index("void OnRx(") : compact.index("static void AdmitInjections")
        ]
        incoming = on_rx.index("g.incoming[nodeId]++;")

        self.assertEqual(on_rx.count("g.incoming[nodeId]++;"), 1)
        self.assertLess(on_rx.index("MarkBacklog(h.pktId);"), incoming)
        for branch in (
            "if (nodeId == h.finalDst)",
            "if (DeadlineExceeded",
            "if (hops >= g.maxHops)",
            "if (g.nodeQ[nodeId].size() >= g.nodeQCap)",
        ):
            with self.subTest(branch=branch):
                self.assertLess(incoming, on_rx.index(branch))

    def test_cpp_exports_one_explicit_terminal_reason_per_packet(self):
        compact = " ".join(self.cpp_source().split())

        self.assertIn(
            'delivered ? std::string("delivered") : g.dropReason.at(pid)',
            compact,
        )
        self.assertNotIn(
            'g.dropReason.count(pid) ? g.dropReason[pid] : "backlog"',
            compact,
        )

    def test_cpp_episode_boundary_resets_only_dynamic_data_plane_state(self):
        source = self.cpp_source()
        compact = " ".join(source.split())
        transition_start = compact.index("static void MarkBacklog")
        transition_end = compact.index("// ---- bridge socket helpers")
        transition = compact[transition_start:transition_end]

        for statement in (
            'g.dropReason[pktId] = "backlog";',
            "g.truncatedBacklog++;",
            "g.nodeQ.clear();",
            "g.incoming.clear();",
            "g.episodeLinkTx.clear();",
            "g.snapshotTx.clear();",
            "queue->Flush();",
            "g.txEpisodeByUid.clear();",
        ):
            with self.subTest(statement=statement):
                self.assertIn(statement, transition)

        for cumulative_state in (
            "g.linkTx.clear()",
            "g.delaysMs.clear()",
            "g.allSent.clear()",
        ):
            with self.subTest(cumulative_state=cumulative_state):
                self.assertNotIn(cumulative_state, transition)

        self.assertIn(
            "if ((int)h.episode != g.activeEpisode) { MarkBacklog(h.pktId); return; }",
            compact,
        )
        self.assertIn(",truncated_backlog=", source)

    def test_cpp_episode_clock_does_not_depend_on_injection_presence(self):
        source = self.cpp_source()
        compact = " ".join(source.split())
        slot_start = compact.index("void SlotBoundary(uint32_t gslot)")
        slot_end = compact.index("int main(int argc, char* argv[])")
        slot_body = compact[slot_start:slot_end]

        self.assertIn(
            "uint32_t episode = (gslot - 1) / g.episodeSlots;",
            compact,
        )
        self.assertIn(
            "return episode < g.numEpisodes ? (int)episode : -1;",
            compact,
        )
        self.assertIn(
            "int activeEp = EpisodeForGlobalSlot(gslot);",
            slot_body,
        )
        self.assertLess(
            slot_body.index("TransitionEpisode(activeEp)"),
            slot_body.index("AdmitInjections(gslot, true)"),
        )
        self.assertNotIn("uint32_t epStart = inj.episode", slot_body)
        self.assertIn("for (auto& [k, total] : g.episodeLinkTx)", slot_body)
        self.assertNotIn("for (auto& [k, total] : g.linkTx)", slot_body)

    def test_cpp_admission_phases_match_reset_and_step_end_semantics(self):
        source = self.cpp_source()
        compact = " ".join(source.split())
        slot_start = compact.index("void SlotBoundary(uint32_t gslot)")
        slot_end = compact.index("int main(int argc, char* argv[])")
        slot_body = compact[slot_start:slot_end]
        admit_start = compact.index("static void AdmitInjections")
        admit_end = compact.index("static void ExpireQueuedPackets")
        admit_body = compact[admit_start:admit_end]

        initial = slot_body.index("AdmitInjections(gslot, true)")
        report = slot_body.index("// ---- report ----")
        decisions = slot_body.index("// ---- decisions ----")
        expire = slot_body.index("ExpireQueuedPackets(gslot)")
        exogenous = slot_body.index("AdmitInjections(gslot, false)")
        self.assertLess(initial, report)
        self.assertLess(decisions, expire)
        self.assertLess(expire, exogenous)
        self.assertIn("bool initial;", source)
        self.assertLess(admit_body.index("g.sent++;"), admit_body.index("g.nodeQ[inj.src].size()"))

    def test_runner_binds_episode_count_and_exports_backlog_audit(self):
        source = (ROOT / "src" / "run_ns3_closed_loop.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('f"--num-episodes={len(seeds)}"', source)
        self.assertIn('f"--initial-packets={ini}"', source)
        self.assertIn("validate_packet_trace_contract(", source)
        self.assertIn('"truncated_backlog"', source)

    def test_runner_cannot_run_an_unbound_or_unverified_ns3_program(self):
        source = (ROOT / "src" / "run_ns3_closed_loop.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        policy_loop = next(
            node
            for node in ast.walk(main)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "policy"
        )
        verification_calls = sorted(
            (
                node
                for node in ast.walk(policy_loop)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "verify_ns3_program_provenance"
            ),
            key=lambda node: node.lineno,
        )
        process_calls = [
            node
            for node in ast.walk(policy_loop)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr == "run"
        ]

        self.assertEqual(len(verification_calls), 2)
        self.assertEqual(len(process_calls), 1)
        self.assertLess(verification_calls[0].lineno, process_calls[0].lineno)
        self.assertLess(process_calls[0].lineno, verification_calls[1].lineno)
        loop_source = ast.unparse(policy_loop)
        self.assertIn("WSL_NS3_EXECUTABLE", loop_source)
        self.assertNotIn("./ns3 run", loop_source)

        manifest_assignment = next(
            node
            for node in main.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "run_manifest"
                for target in node.targets
            )
        )
        manifest_keys = {
            key.value
            for key in manifest_assignment.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        self.assertIn("closed_loop_program", manifest_keys)

    def test_orchestrator_constructs_a_server_inside_the_policy_loop(self):
        source = (ROOT / "src" / "run_ns3_closed_loop.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ClosedLoopServer"
        ]
        enclosing_policy_loops = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "policy"
            and any(candidate is calls[0] for candidate in ast.walk(node))
        ]

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(enclosing_policy_loops), 1)
        loop = enclosing_policy_loops[0]
        self.assertIsInstance(loop.iter, ast.Name)
        self.assertEqual(loop.iter.id, "policies")
        self.assertLess(loop.lineno, calls[0].lineno)
        self.assertLessEqual(calls[0].lineno, loop.end_lineno)


if __name__ == "__main__":
    unittest.main()
