from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_source_runtime_equivalence_audit as audit


ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = (
    ROOT
    / "experiments"
    / "archive"
    / "source-snapshots"
    / "actor-score-hysteresis-c3d05253666b.zip"
)


def valid_artifact() -> dict:
    value = {
        "schema_version": audit.SCHEMA_VERSION,
        "audit_name": audit.AUDIT_NAME,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "new_evaluation_panel_accessed": False,
        "decision": {"all_pass": True},
    }
    value["source_runtime_equivalence_sha256"] = audit.sha256_json(value)
    return value


class SourceRuntimeEquivalenceAuditTests(unittest.TestCase):
    def test_archive_binds_exact_entry_set_hashes_and_lengths(self):
        record = audit.audit_archive(ARCHIVE)
        self.assertEqual(record["sha256"], audit.EXPECTED_ARCHIVE_SHA256)
        self.assertEqual(
            set(record["entries"]), set(audit.EXPECTED_ARCHIVE_ENTRIES)
        )
        for name, (expected_sha, expected_length) in (
            audit.EXPECTED_ARCHIVE_ENTRIES.items()
        ):
            with self.subTest(name=name):
                self.assertEqual(record["entries"][name]["sha256"], expected_sha)
                self.assertEqual(
                    record["entries"][name]["length"], expected_length
                )

    def test_method_contract_accepts_only_declared_relations(self):
        snapshot = """
class Example:
    def equal(self, value):
        return value

    def changed(self):
        return 1
"""
        current = """
class Example:
    def equal(self, value):
        return value

    def changed(self):
        return 2

    def added(self):
        return 3
"""
        result = audit._method_contract(
            snapshot,
            current,
            class_name="Example",
            expected_equal=("equal",),
            expected_changed=("changed",),
            expected_added=("added",),
        )
        self.assertEqual(result["equal_methods"], ["equal"])
        self.assertEqual(result["changed_methods"], ["changed"])
        self.assertEqual(result["added_methods"], ["added"])
        with self.assertRaisesRegex(ValueError, "added-method"):
            audit._method_contract(
                snapshot,
                current,
                class_name="Example",
                expected_equal=("equal",),
                expected_changed=("changed",),
                expected_added=(),
            )

    def test_artifact_validation_is_fail_closed(self):
        value = valid_artifact()
        expected = value["source_runtime_equivalence_sha256"]
        self.assertEqual(audit.validate_equivalence_artifact(value), expected)

        tampered = dict(value, paper_claim_allowed=True)
        with self.assertRaisesRegex(ValueError, "self-hash"):
            audit.validate_equivalence_artifact(tampered)

        claim_enabled = dict(value)
        claim_enabled["paper_claim_allowed"] = True
        claim_enabled["source_runtime_equivalence_sha256"] = audit.sha256_json(
            {
                key: item
                for key, item in claim_enabled.items()
                if key != "source_runtime_equivalence_sha256"
            }
        )
        with self.assertRaisesRegex(ValueError, "prohibit paper claims"):
            audit.validate_equivalence_artifact(claim_enabled)

    def test_load_and_replay_requires_complete_exact_artifact(self):
        artifact = valid_artifact()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "equivalence.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with mock.patch.object(
                audit, "build_equivalence_artifact", return_value=dict(artifact)
            ) as builder:
                replayed = audit.load_and_replay_equivalence_artifact(
                    path,
                    project_root=ROOT,
                    archive_path=ARCHIVE,
                    source_spec_path=Path("source.json"),
                    hysteresis_spec_path=Path("hysteresis.json"),
                )
            self.assertEqual(replayed, artifact)
            builder.assert_called_once()

            different = dict(artifact, extra="drift")
            with mock.patch.object(
                audit, "build_equivalence_artifact", return_value=different
            ), self.assertRaisesRegex(ValueError, "does not replay"):
                audit.load_and_replay_equivalence_artifact(
                    path,
                    project_root=ROOT,
                    archive_path=ARCHIVE,
                    source_spec_path=Path("source.json"),
                    hysteresis_spec_path=Path("hysteresis.json"),
                )

    def test_immutable_writer_reuses_match_and_rejects_drift(self):
        value = valid_artifact()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "equivalence.json"
            audit.ensure_immutable_json(path, value)
            first = path.read_bytes()
            audit.ensure_immutable_json(path, value)
            self.assertEqual(path.read_bytes(), first)
            with self.assertRaisesRegex(RuntimeError, "immutable artifact"):
                audit.ensure_immutable_json(path, dict(value, extra=True))

    def test_protocol_uses_only_exposed_workload_and_no_jobs(self):
        self.assertEqual(audit.WORKLOAD_SEED, 60001)
        self.assertEqual(
            audit.WORKLOAD_SEED_PROVENANCE,
            "replay_of_exposed_v7_validation_seed_60001",
        )
        self.assertEqual(audit.POLICY_KINDS, (
            "first_feasible",
            "checkpoint_beta_0p40",
        ))


if __name__ == "__main__":
    unittest.main()
