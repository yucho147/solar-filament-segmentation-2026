"""Portable checks for the inference entry point; no model files are needed."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reinfer_10m", ROOT / "scripts/reinfer_10m.py")
INFER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INFER)


class InferenceChecks(unittest.TestCase):
    def test_seed_averaging(self):
        a = np.array([[0.2, 0.7]], dtype=np.float32)
        b = np.array([[0.6, 0.1]], dtype=np.float32)
        np.testing.assert_array_equal(INFER.mean_seed_maps(a, b), (a + b) / 2)
        with self.assertRaises(ValueError):
            INFER.mean_seed_maps(a, b.T)

    def test_cache_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            INFER.prepare_output(root, {"sample": "original"})
            with self.assertRaises(ValueError):
                INFER.prepare_output(root, {"sample": "different"})
            path = root / "probability.npy"
            INFER.save_probability(path, np.zeros((2, 2), dtype=np.float32))
            self.assertTrue(INFER.reusable_probability(path))
            np.save(path, np.ones((2, 2), dtype=np.float32))
            with self.assertRaises(ValueError):
                INFER.reusable_probability(path)

    def test_nonfinite_probability_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npy"
            np.save(path, np.array([[np.nan]], dtype=np.float32))
            with self.assertRaises(ValueError):
                INFER.probability(path)


if __name__ == "__main__":
    unittest.main()
