"""Meaningful causal/invariance checks using invented data only, never competition files."""
import unittest
import numpy as np
import pandas as pd
from features import build_sensor_features, feature_names, fit_preprocessing, apply_preprocessing


def synthetic_frame():
    rows = []
    for experiment,offset in [("invented_A", 0.), ("invented_B", 20.)]:
        for time in [0., 5., 15., 40., 70., 100.]:
            rows.append({"sample_id":f"{experiment}_{time}", "experiment_id":experiment,
                         "relative_time_s":time, "meas_v":max(4.-time*.04, 0.), "meas_a":0.,
                         "meas_t1":10.+time*2+offset, "meas_t2":20.+time+offset, "meas_p":1.+time*.1})
    return pd.DataFrame(rows, index=np.arange(len(rows))*3+7)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_frame()

    def aligned(self, frame, feature_set="all"):
        result = build_sensor_features(frame, feature_set)
        result.index = frame.sample_id.to_numpy()
        return result

    def test_original_nine_columns_are_unchanged(self):
        f = self.frame
        actual = build_sensor_features(f, "current")
        expected = np.column_stack([f[["meas_v", "meas_a", "meas_t1", "meas_t2", "meas_p"]],
                                    f.meas_t2-f.meas_t1, (f.meas_t1+f.meas_t2)/2,
                                    np.maximum(f.meas_t1, f.meas_t2), f.meas_v*f.meas_a])
        np.testing.assert_equal(actual.to_numpy(), expected)
        self.assertEqual(list(actual.index), list(f.index))

    def test_sparse_lag_uses_actual_elapsed_seconds(self):
        actual = self.aligned(self.frame)
        row = actual.loc["invented_A_40.0"]
        self.assertAlmostEqual(row.meas_t1_delta30s, 70.)  # latest t <= 10 is t=5
        self.assertAlmostEqual(row.meas_t1_rate30s, 2.)   # 70 / (40-5), not 70/30

    def test_shuffle_preserves_every_feature(self):
        full = self.aligned(self.frame)
        shuffled = self.aligned(self.frame.sample(frac=1, random_state=3))
        pd.testing.assert_frame_equal(full, shuffled.reindex(full.index))

    def test_experiment_split_preserves_every_feature(self):
        full = self.aligned(self.frame)
        for _,group in self.frame.groupby("experiment_id"):
            pd.testing.assert_frame_equal(full.loc[group.sample_id], self.aligned(group))

    def test_removing_future_preserves_prefix(self):
        full = self.aligned(self.frame)
        prefix = self.frame.loc[self.frame.relative_time_s <= 40].copy()
        pd.testing.assert_frame_equal(full.loc[prefix.sample_id], self.aligned(prefix))

    def test_modifying_future_cannot_change_past(self):
        modified = self.frame.copy()
        modified.loc[modified.relative_time_s > 40, ["meas_v", "meas_t1", "meas_t2"]] = 1e9
        past = self.frame.relative_time_s <= 40
        pd.testing.assert_frame_equal(self.aligned(self.frame).loc[self.frame.loc[past].sample_id],
                                      self.aligned(modified).loc[self.frame.loc[past].sample_id])

    def test_tied_timestamps_have_no_order_dependence(self):
        tied = self.frame.iloc[[1]].copy()
        tied["sample_id"] = "invented_tie"
        tied["meas_t1"] = 100.
        tied["meas_v"] = 10.
        frame = pd.concat([self.frame, tied], ignore_index=True)
        full = self.aligned(frame)
        shuffled = self.aligned(frame.sample(frac=1, random_state=4))
        pd.testing.assert_frame_equal(full, shuffled.reindex(full.index))
        # Both t=5 rows participate in the peak, and lag uses their mean.
        self.assertAlmostEqual(full.loc["invented_A_5.0", "voltage_drop_from_past_peak"], 6.2)
        self.assertAlmostEqual(full.loc["invented_A_40.0", "meas_t1_delta30s"], 30.)

    def test_missing_history_is_not_filled_from_future(self):
        frame = self.frame.copy()
        frame.loc[frame.relative_time_s <= 15, "meas_t1"] = np.nan
        features = self.aligned(frame)
        self.assertTrue(np.isnan(features.loc["invented_A_15.0", "meas_t1_drop_from_past_peak"]))
        self.assertTrue(np.isnan(features.loc["invented_A_40.0", "meas_t1_delta30s"]))

    def test_zero_voltage_never_produces_infinity(self):
        frame = self.frame.copy()
        frame["meas_v"] = 0.
        features = build_sensor_features(frame)
        self.assertFalse(np.isinf(features.to_numpy()).any())
        self.assertTrue(features.voltage_drop_fraction.isna().all())

    def test_feature_sets_have_stable_dimensions(self):
        expected = {"current":9, "temperature":21, "voltage":11, "cooling":11, "all":25}
        for name,dim in expected.items():
            self.assertEqual(len(feature_names(name)), dim)
            self.assertEqual(build_sensor_features(self.frame, name).shape, (len(self.frame), dim))

    def test_preprocessing_is_fitted_only_on_given_training_rows(self):
        train = build_sensor_features(self.frame.iloc[:6])
        prep = fit_preprocessing(train)
        before = {k:v.copy() for k,v in prep.items()}
        valid = self.frame.iloc[6:].copy()
        valid[["meas_t1", "meas_t2"]] = 1e6
        transformed = apply_preprocessing(build_sensor_features(valid), prep)
        self.assertTrue(np.isfinite(transformed).all())
        for key,value in before.items():
            np.testing.assert_equal(prep[key], value)

    def test_transformed_prefix_and_permutation_are_invariant(self):
        features = self.aligned(self.frame)
        prep = fit_preprocessing(features)
        full = pd.DataFrame(apply_preprocessing(features, prep), index=features.index)
        prefix = self.frame[self.frame.relative_time_s <= 40].sample(frac=1, random_state=5)
        actual = pd.DataFrame(apply_preprocessing(self.aligned(prefix), prep), index=prefix.sample_id.to_numpy())
        pd.testing.assert_frame_equal(full.loc[prefix.sample_id], actual)

    def test_invalid_time_is_rejected(self):
        frame = self.frame.copy()
        frame.loc[frame.index[0], "relative_time_s"] = np.nan
        with self.assertRaises(ValueError):
            build_sensor_features(frame)


class NetworkTests(unittest.TestCase):
    def test_variable_sensor_size_forward_backward_and_checkpoint(self):
        import io
        import torch
        from network import Fusion
        torch.set_num_threads(2)
        for dim in (9, 25):
            net = Fusion(dim)
            images, sensors = torch.randn(4, 512), torch.randn(4, dim)
            logits = net(images, sensors)
            self.assertEqual(tuple(logits.shape), (4, 6))
            logits.square().mean().backward()
            self.assertIsNotNone(net.sensor[0].weight.grad)
            net.eval()
            saved = io.BytesIO()
            torch.save(net.state_dict(), saved)
            saved.seek(0)
            restored = Fusion(dim).eval()
            restored.load_state_dict(torch.load(saved, weights_only=True))
            torch.testing.assert_close(net(images, sensors), restored(images, sensors), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
