"""Hybrid fit/save/load/invariance and control parity; invented data only."""
import copy
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import hybrid as H
import model as M
from workflow import make_splits


def invented_data():
    rng = np.random.default_rng(10)
    records = []
    for experiment in range(6):
        for index in range(30):
            time = index*5+(index//7)*10
            values = rng.normal(size=5)
            records.append(dict(sample_id=f"invented_{experiment}_{index}", experiment_id=f"invented_{experiment}",
                                relative_time_s=time, meas_v=4.-index*.12, meas_a=values[1],
                                meas_t1=20.+time+values[2], meas_t2=25.+time*.7+values[3],
                                meas_p=values[4], label=index%6+1))
    frame = pd.DataFrame(records)
    frame.loc[0, "meas_t1"] = np.nan
    frame.loc[3, "meas_v"] = np.inf
    return frame,torch.from_numpy(rng.normal(size=(len(frame), 512)).astype(np.float32))


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        M.seed_all(42)
        cls.frame,cls.raw = invented_data()
        cls.train,cls.valid = cls.frame.iloc[:120],cls.frame.iloc[120:]
        cls.image_tr,cls.image_va = cls.raw[:120],cls.raw[120:]
        cls.bundles,cls.trees = H.fit_family(cls.train,cls.image_tr,torch.device("cpu"),
                                             epochs=1,tree_iterations=3,pca_components=4)

    def test_candidate_recipe_dimensions(self):
        expected = {"hybrid_control":("current","current"), "temporal_tree":("current","all"), "temporal_both":("all","all")}
        for recipe,(mlp,tree) in expected.items():
            bundle = self.bundles[recipe]
            self.assertEqual((bundle["mlp_features"], bundle["tree_features"]), (mlp,tree))
            self.assertEqual(bundle["states"][0]["sensor.0.weight"].shape[1],len(M.feature_names(mlp)))
            self.assertEqual(self.trees[recipe]["hgb"].n_features_in_,len(M.feature_names(tree))+4)

    def test_tree_only_candidate_reuses_control_neural_heads(self):
        for left,right in zip(self.bundles["hybrid_control"]["states"],self.bundles["temporal_tree"]["states"]):
            for key in left:
                torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)

    def test_all_candidates_normalized_and_serialization_stable(self):
        output_root = Path(__file__).resolve().parent/"outputs"
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root,prefix="invented-check-") as directory:
            directory = Path(directory)
            for recipe in H.RECIPES:
                expected = H.predict(self.valid,self.image_va,self.bundles[recipe],self.trees[recipe],torch.device("cpu"))
                self.assertEqual(expected.shape,(60,6))
                np.testing.assert_allclose(expected.sum(1),1.,atol=1e-6)
                torch.save(self.bundles[recipe],directory/"bundle.pt")
                joblib.dump(self.trees[recipe],directory/"tree.joblib")
                actual = H.predict(self.valid,self.image_va,
                    torch.load(directory/"bundle.pt",weights_only=True),joblib.load(directory/"tree.joblib"),torch.device("cpu"))
                np.testing.assert_allclose(actual,expected,rtol=0,atol=0)

    def test_shuffle_experiment_split_and_prefix_predictions(self):
        permutation = np.random.default_rng(3).permutation(len(self.valid))
        prefix = self.valid.relative_time_s.to_numpy() <= 70
        first_experiment = self.valid.experiment_id.to_numpy() == self.valid.experiment_id.iloc[0]
        for recipe in H.RECIPES:
            bundle,tree = self.bundles[recipe],self.trees[recipe]
            expected = H.predict(self.valid,self.image_va,bundle,tree,torch.device("cpu"))
            for positions in (permutation,np.flatnonzero(prefix),np.flatnonzero(first_experiment)):
                actual = H.predict(self.valid.iloc[positions],self.image_va[positions],bundle,tree,torch.device("cpu"))
                np.testing.assert_allclose(actual,expected[positions],atol=1e-6,rtol=1e-6)
                np.testing.assert_equal(actual.argmax(1),expected[positions].argmax(1))

    def test_future_value_changes_do_not_affect_prefix(self):
        modified = self.valid.copy()
        past = modified.relative_time_s.to_numpy() <= 70
        modified.loc[~past,["meas_v","meas_t1","meas_t2"]] = 1e6
        modified_images = self.image_va.clone()
        modified_images[~past] = 1e4
        for recipe in H.RECIPES:
            expected = H.predict(self.valid,self.image_va,self.bundles[recipe],self.trees[recipe],torch.device("cpu"))
            actual = H.predict(modified,modified_images,self.bundles[recipe],self.trees[recipe],torch.device("cpu"))
            np.testing.assert_allclose(actual[past],expected[past],atol=1e-6,rtol=1e-6)

    def test_inference_never_fits_preprocessing_or_pca(self):
        recipe = "temporal_tree"
        with patch.object(M,"fit_preprocessing",side_effect=AssertionError("Inference fit forbidden")), \
             patch.object(self.trees[recipe]["pca"],"fit",side_effect=AssertionError("Inference PCA fit forbidden")):
            H.predict(self.valid,self.image_va,self.bundles[recipe],self.trees[recipe],torch.device("cpu"))

    def test_labels_and_sample_identifiers_are_not_prediction_features(self):
        modified = self.valid.copy()
        modified["label"] = 1
        modified["sample_id"] = [f"unrelated_id_{i}" for i in range(len(modified))]
        for recipe in H.RECIPES:
            expected = H.predict(self.valid,self.image_va,self.bundles[recipe],self.trees[recipe],torch.device("cpu"))
            actual = H.predict(modified,self.image_va,self.bundles[recipe],self.trees[recipe],torch.device("cpu"))
            np.testing.assert_allclose(actual,expected,atol=0,rtol=0)

    def test_corrupted_schema_is_rejected(self):
        bundle = copy.deepcopy(self.bundles["temporal_tree"])
        bundle["schemas"]["all"] = bundle["schemas"]["all"][::-1]
        with self.assertRaises(ValueError):
            H.predict(self.valid,self.image_va,bundle,self.trees["temporal_tree"],torch.device("cpu"))

    def test_group_folds_preserve_experiments(self):
        for train,valid in make_splits(self.frame,folds=3):
            self.assertFalse(set(self.frame.iloc[train].experiment_id)&set(self.frame.iloc[valid].experiment_id))

    def test_recreating_original_hybrid_control_on_invented_data(self):
        reference_dir = Path(__file__).resolve().parent/"reference"
        sys.path.insert(0,str(reference_dir))
        spec = importlib.util.spec_from_file_location("hybrid_reference",reference_dir/"hybrid_reference.py")
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        bundle,trees = reference.fit(self.train,self.image_tr,torch.device("cpu"),epochs=1,tree_iterations=3,pca_components=4)
        expected = reference.predict_from_embeddings(self.valid,self.image_va,bundle,trees,torch.device("cpu"))["fusion_hybrid"]
        actual = H.predict(self.valid,self.image_va,self.bundles["hybrid_control"],self.trees["hybrid_control"],torch.device("cpu"))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=0)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        unittest.main(verbosity=2)
