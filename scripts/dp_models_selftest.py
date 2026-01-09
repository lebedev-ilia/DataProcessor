#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
import unittest

# Ensure repo root is importable when running as a script (sys.path[0] = scripts/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dp_models.digests import sha256_file
from dp_models.manager import ModelManager, ModelManagerConfig
from dp_models.offline import network_guard
from dp_models.signatures import compute_model_signature


class TestDigests(unittest.TestCase):
    def test_sha256_file_stable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "a.bin")
            with open(p, "wb") as f:
                f.write(b"hello")
            d1 = sha256_file(p)
            d2 = sha256_file(p)
            self.assertEqual(d1, d2)


class TestSignature(unittest.TestCase):
    def test_signature_canonical_order(self) -> None:
        models_used = [
            {"model_name": "b", "model_version": "1", "weights_digest": "x", "runtime": "inprocess", "engine": "e", "precision": "fp16", "device": "cuda"},
            {"model_name": "a", "model_version": "2", "weights_digest": "y", "runtime": "inprocess", "engine": "e", "precision": "fp16", "device": "cuda"},
        ]
        s1 = compute_model_signature(models_used)
        s2 = compute_model_signature(list(reversed(models_used)))
        self.assertEqual(s1, s2)


class TestModelManagerPaths(unittest.TestCase):
    def test_forbid_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as models_root:
            with tempfile.TemporaryDirectory() as catalog_root:
                # write a malicious spec
                spec_path = os.path.join(catalog_root, "bad.json")
                with open(spec_path, "w", encoding="utf-8") as f:
                    f.write(
                        """
{
  "model_name": "bad",
  "model_version": "1",
  "role": "x",
  "runtime": "inprocess",
  "engine": "sentence-transformers",
  "precision": "fp16",
  "device_policy": "cpu",
  "local_artifacts": [{"path": "../escape", "kind": "dir"}],
  "weights_digest": "auto"
}
                        """.strip()
                    )
                mm = ModelManager(ModelManagerConfig(models_root=models_root, catalog_root=catalog_root))
                spec = mm.get_spec(model_name="bad")
                with self.assertRaises(Exception):
                    mm.resolve(spec)


def main() -> int:
    # Optional strict no-network guard for tests
    strict = os.environ.get("DP_MODELS_SELFTEST_NO_NETWORK", "1").strip() != "0"
    with network_guard(enabled=strict):
        suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
        res = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if res.wasSuccessful() else 2


if __name__ == "__main__":
    raise SystemExit(main())


