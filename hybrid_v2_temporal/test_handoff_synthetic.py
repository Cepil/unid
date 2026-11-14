"""Handoff allowlist, checksums and preservation tests using synthetic fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import torch
import model as M
import hybrid as H
from package_handoff import create_handoff
from verify_handoff import verify


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_root = Path(__file__).resolve().parent
        scratch = cls.source_root/"outputs"
        scratch.mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=scratch,prefix="synthetic-handoff-")
        cls.workspace = Path(cls.temporary.name)
        cls.root = cls.workspace/"source/hybrid_v2_temporal"
        cls.root.mkdir(parents=True)
        source_record = json.loads((cls.source_root/"PUBLIC09401_SOURCE.json").read_text(encoding="utf-8"))
        for name in source_record["core_source_sha256"]:
            shutil.copyfile(cls.source_root/name,cls.root/name)
        for name in ("PUBLIC09401_SOURCE.json","TEAM_HANDOFF.md"):
            shutil.copyfile(cls.source_root/name,cls.root/name)
        cls.model_dir = cls.root/"outputs/selected_temporal_tree"
        cls.model_dir.mkdir(parents=True)
        versions = H.package_versions()
        # Artificial weights and metadata exercise packaging, not real training/results.
        cls.bundle = {
            "format_version":3,"model_family":"hybrid_v2_temporal","recipe":"temporal_tree",
            "mlp_features":"current","tree_features":"all","epochs":16,"seeds":[42,43,44],
            "mixture_weights":[.5,.5],"class_labels":[1,2,3,4,5,6],"tree_file":"tree.joblib",
            "states":[M.Fusion(9).state_dict() for _ in range(3)],"encoder":M.encoder().state_dict(),
            "schemas":{"current":M.feature_names("current"),"all":M.feature_names("all")},
            "environment_versions":versions,
        }
        torch.save(cls.bundle,cls.model_dir/"model.pt")
        # Never deserialized: deliberately not a joblib model, only an opaque fixture.
        (cls.model_dir/"tree.joblib").write_bytes(b"invented tree fixture; no actual model")
        cls.weights_manifest = {"environment_versions":versions,"files":{
            name:hashlib.sha256((cls.model_dir/name).read_bytes()).hexdigest() for name in ("model.pt","tree.joblib")}}
        (cls.model_dir/"manifest.json").write_text(json.dumps(cls.weights_manifest),encoding="utf-8")
        validation = {"recipe":"temporal_tree","epochs":16,"validation_used_for_fitting":False,
                      "metrics":{"macro_f1":.751640},"environment_versions":versions}
        (cls.model_dir/"validation_metrics.json").write_text(json.dumps(validation),encoding="utf-8")
        cls.csv = cls.root/"outputs/submission_temporal_tree.csv"
        cls.csv.write_text("sample_id,label\ninvented_only,1\n",encoding="utf-8")
        (cls.root/"outputs/private.log").write_text("invented private log excluded",encoding="utf-8")
        cls.archive = cls.workspace/"synthetic_weights.tar.gz"
        create_handoff(cls.root,cls.model_dir,cls.csv,cls.archive)
        cls.receiver = cls.workspace/"receiver"
        cls.receiver.mkdir()
        with tarfile.open(cls.archive) as archive:
            archive.extractall(cls.receiver,filter="data")
        for name in source_record["core_source_sha256"]:
            shutil.copyfile(cls.source_root/name,cls.receiver/"hybrid_v2_temporal"/name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_archive_has_only_explicit_artifacts_and_no_csv_log_cache(self):
        with tarfile.open(self.archive) as archive:
            names = archive.getnames()
            self.assertFalse(any(name.endswith((".csv",".log",".npz",".npy")) or "/cache/" in name for name in names))
            self.assertIn("hybrid_v2_temporal/weights/model.pt",names)
            self.assertIn("hybrid_v2_temporal/weights/tree.joblib",names)
            card = json.load(archive.extractfile("hybrid_v2_temporal/MODEL_CARD.json"))
            self.assertEqual(card["submitted_csv_sha256"],hashlib.sha256(self.csv.read_bytes()).hexdigest())

    def test_recipient_verifies_hashes_source_and_submission(self):
        self.assertTrue(verify(self.receiver/"hybrid_v2_temporal",self.csv))

    def test_preserves_existing_archive(self):
        before = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        with self.assertRaises(FileExistsError):
            create_handoff(self.root,self.model_dir,self.csv,self.archive)
        self.assertEqual(before,hashlib.sha256(self.archive.read_bytes()).hexdigest())

    def test_changed_core_source_is_rejected(self):
        path = self.root/"test.py"
        original = path.read_bytes()
        try:
            path.write_bytes(original+b"\n# changed\n")
            with self.assertRaisesRegex(ValueError,"Core code differs"):
                create_handoff(self.root,self.model_dir,self.csv,self.workspace/"bad_source.tar.gz")
        finally:
            path.write_bytes(original)

    def test_corrupted_weight_is_rejected_without_deserialization(self):
        path = self.model_dir/"tree.joblib"
        original = path.read_bytes()
        try:
            path.write_bytes(original+b"changed")
            with self.assertRaisesRegex(ValueError,"checksum"):
                create_handoff(self.root,self.model_dir,self.csv,self.workspace/"bad_weight.tar.gz")
        finally:
            path.write_bytes(original)

    def test_wrong_recipe_is_rejected(self):
        wrong = {**self.bundle,"recipe":"temporal_both"}
        with patch("torch.load",return_value=wrong):
            with self.assertRaisesRegex(ValueError,"recipe"):
                create_handoff(self.root,self.model_dir,self.csv,self.workspace/"bad_recipe.tar.gz")

    def test_recipient_rejects_mismatched_library(self):
        with patch("importlib.metadata.version",return_value="0.0.synthetic"):
            with self.assertRaisesRegex(RuntimeError,"scikit-learn"):
                verify(self.receiver/"hybrid_v2_temporal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
