import tempfile
import unittest
from pathlib import Path

import numpy as np

import run_avoidable_switch_joint_sealed_test as sealed


class SealedContractTests(unittest.TestCase):
    def test_frozen_grid_size(self) -> None:
        mappo = len(sealed.formal.SCENARIOS) * len(sealed.formal.ARMS) * len(sealed.formal.POLICY_SEEDS) * len(sealed.TEST_SEEDS)
        classical = 2 * (8 + 8 + 1) * len(sealed.TEST_SEEDS)
        self.assertEqual(mappo + classical, 4100)
        self.assertEqual(sealed.TEST_SEEDS, tuple(range(78001, 78051)))

    def test_exact_sign_flip_minimum(self) -> None:
        self.assertEqual(sealed.exact_sign_flip(np.ones(8)), 2 / 256)

    def test_holm_is_monotone_in_rank(self) -> None:
        raw = [0.04, 0.01, 0.03, 0.02]
        adjusted = sealed.holm(raw)
        ranked = sorted(zip(raw, adjusted))
        self.assertEqual([value for _, value in ranked], sorted(value for _, value in ranked))

    def test_authorization_self_hash_detects_change(self) -> None:
        value = sealed.self_hashed({"a": 1}, "sha")
        sealed.validate_self_hash(value, "sha")
        value["a"] = 2
        with self.assertRaises(sealed.ContractError):
            sealed.validate_self_hash(value, "sha")

    def test_prepare_does_not_instantiate_test_rows(self) -> None:
        prereq = {"paths": sealed.paths()}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            body = sealed.authorization_body(prereq)
            self.assertEqual(body["test_access_count"], 0)
            self.assertFalse(body["sealed_test_instantiated"])
            self.assertFalse(any(output.iterdir()))


if __name__ == "__main__":
    unittest.main()
