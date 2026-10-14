import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from c3_sandbox.config import load_config
from c3_sandbox.planning import build_plan
from c3_sandbox.evidence import PredictionStore, WorkspaceStore
from c3_sandbox.evidence._storage import digest, encode


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        (self.workspace / ".hidden").write_bytes(b"\x00\xff\x01")
        (self.workspace / "empty").mkdir()
        (self.workspace / ".hidden").chmod(0o640)
        self.store = WorkspaceStore(self.root / "store")

    def test_restore_complete_tree_and_modes(self):
        snapshot = self.store.capture(self.workspace)
        dest = self.root / "restored"
        report = self.store.restore(snapshot, dest)
        self.assertTrue(report.valid, report.mismatches)
        self.assertEqual((dest / ".hidden").read_bytes(), b"\x00\xff\x01")
        self.assertTrue((dest / "empty").is_dir())
        self.assertIn("filesystem-only", report.to_dict()["scope"])
        self.assertEqual((dest / ".hidden").stat().st_mode & 0o777, 0o640)
        self.assertFalse(self.store.restore(snapshot, dest).valid)

    def test_extra_missing_modified_modes(self):
        snapshot = self.store.capture(self.workspace)
        (self.workspace / "extra").write_text("unexpected")
        self.assertIn("unexpected: extra", self.store.verify(snapshot, self.workspace).mismatches)
        (self.workspace / "extra").unlink()
        (self.workspace / ".hidden").chmod(0o600)
        self.assertFalse(self.store.verify(snapshot, self.workspace).valid)
        (self.workspace / ".hidden").chmod(0o640)
        (self.workspace / ".hidden").write_bytes(b"changed")
        self.assertFalse(self.store.verify(snapshot, self.workspace).valid)
        (self.workspace / ".hidden").unlink()
        self.assertIn("missing: .hidden", self.store.verify(snapshot, self.workspace).mismatches)

    def test_symlink_and_special_file_rejected(self):
        (self.workspace / "link").symlink_to(".hidden")
        with self.assertRaises(ValueError):
            self.store.capture(self.workspace)
        (self.workspace / "link").unlink()
        os.mkfifo(self.workspace / "pipe")
        with self.assertRaises(ValueError):
            self.store.capture(self.workspace)

    def test_object_and_manifest_tampering_fail_closed(self):
        snapshot = self.store.capture(self.workspace)
        blob = next(self.store.objects.iterdir())
        blob.chmod(0o600)
        blob.write_bytes(b"corrupted")
        self.assertFalse(self.store.verify(snapshot, self.workspace).valid)
        dest = self.root / "dest"
        self.assertFalse(self.store.restore(snapshot, dest).valid)
        self.assertFalse(dest.exists())
        manifest = self.store.manifests / (snapshot + ".json")
        manifest.chmod(0o600)
        manifest.write_text("{}")
        self.assertFalse(self.store.verify(snapshot, self.workspace).valid)

    def test_hash_valid_path_traversal_manifest_rejected(self):
        unsafe = {"schema_version": 1, "scope": self.store.verify(self.store.capture(self.workspace), self.workspace).scope,
                  "entries": {".": {"kind": "directory", "mode": 0o755}, "../escaped": {"kind": "directory", "mode": 0o755}}}
        data = encode(unsafe)
        key = digest(data)
        (self.store.manifests / (key + ".json")).write_bytes(data)
        self.assertFalse(self.store.restore(key, self.root / "dest").valid)
        self.assertFalse((self.root / "escaped").exists())
        self.assertFalse(self.store.verify("../bad", self.workspace).valid)

    def test_restore_symlink_destination_rejected(self):
        snapshot = self.store.capture(self.workspace)
        dest = self.root / "symlink"
        dest.symlink_to(self.workspace, target_is_directory=True)
        self.assertFalse(self.store.restore(snapshot, dest).valid)

    def test_aliased_workspace_cannot_capture_its_store(self):
        nested_store = WorkspaceStore(self.workspace / "store")
        alias = self.workspace / ".." / self.workspace.name
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            nested_store.capture(alias)
        self.assertEqual(list(nested_store.objects.iterdir()), [])
        self.assertEqual(list(nested_store.manifests.iterdir()), [])
        with self.assertRaisesRegex(ValueError, "outside captured workspace"):
            nested_store.capture(self.workspace)

    def test_unsupported_backslash_filename_rejected_before_storage(self):
        (self.workspace / "bad\\name").write_bytes(b"data")
        with self.assertRaisesRegex(ValueError, "unsupported backslash"):
            self.store.capture(self.workspace)
        self.assertEqual(list(self.store.objects.iterdir()), [])
        self.assertEqual(list(self.store.manifests.iterdir()), [])

    def prediction(self):
        return {"case_id": "synthetic-1", "method_id": "fixture", "budget": 0,
                "candidate_universe": ["original", "target"], "probabilities": {"original": 0.2, "target": 0.8},
                "is_synthetic": True, "pre_confirmation": True}

    def test_predictions_roundtrip_immutable_and_tamper(self):
        store = PredictionStore(self.root / "predictions")
        record = self.prediction()
        key = store.freeze(record)
        self.assertEqual(store.freeze(record), key)
        self.assertEqual(store.read(key), record)
        changed = copy.deepcopy(record)
        changed["probabilities"]["target"] = 0.9
        self.assertNotEqual(store.freeze(changed), key)
        self.assertEqual(store.read(key), record)
        path = store.root / (key + ".json")
        path.chmod(0o600)
        path.write_text("{}")
        with self.assertRaises(ValueError):
            store.read(key)
        with self.assertRaises(ValueError):
            store.freeze(record)

    def test_invalid_predictions(self):
        store = PredictionStore(self.root / "predictions")
        variants = []
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.4"):
            record = self.prediction()
            record["probabilities"]["target"] = value
            variants.append(record)
        for key, value in (("probabilities", {"original": 0.2}), ("pre_confirmation", False), ("is_synthetic", None), ("candidate_universe", ["original", "original"]), ("budget", -1)):
            record = self.prediction()
            record[key] = value
            variants.append(record)
        for record in variants:
            with self.subTest(record=record), self.assertRaises(ValueError):
                store.freeze(record)

    def test_config_and_false_readiness(self):
        config = load_config(Path(__file__).parents[1] / "configs/local_serial.json")
        plan = build_plan(config)
        self.assertFalse(plan["ready_for_experiments"])
        self.assertEqual(plan["concurrency"], 1)
        self.assertTrue(plan["readiness_blockers"])
        declared = copy.deepcopy(config)
        declared.update(stage="p0", model={"model_id": "x", "version": "1", "budget": 1}, cases=[{"case_id": "1", "status": "frozen"}], probe_library_status="frozen", experiment_execution_enabled=True)
        for key in ("restore_acceptance_verified", "isolation_verified", "independent_scoring_verified", "confirmation_separation_verified"):
            declared[key] = True
        self.assertFalse(build_plan(declared)["ready_for_experiments"])
        for key, value in (("platform", "linux/amd64"), ("concurrency", 2), ("concurrency", True)):
            invalid = copy.deepcopy(config)
            invalid[key] = value
            with self.assertRaises(ValueError):
                build_plan(invalid)
        invalid = copy.deepcopy(config)
        invalid["resources"]["memory_mb"] = 100000
        with self.assertRaises(ValueError):
            build_plan(invalid)

    def test_duplicate_config_keys_rejected(self):
        path = self.root / "config.json"
        path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaises(ValueError):
            load_config(path)


if __name__ == "__main__":
    unittest.main()
