"""hybrid_cum: Hybrid v2 + organizer baseline v7 30 s window features + running-extreme features in the tree.

Components (both use the frozen ImageNet ResNet-18 512-d image embedding):
  1. Fusion MLP (seeds 42/43/44 averaged): image embedding + 9 current-row sensor features (Hybrid v2).
  2. HistGradientBoosting tree: 9 current-row sensor features + 15 window features + PCA-32 of the image.
     Window features come from the organizer baseline (baseline_v7.add_causal_features): for each of the
     5 sensors, the mean and max over (t-30 s, t] and the difference to the last row at or before t-30 s,
     within the same experiment only.
     Running-extreme features (7): since the experiment's first supplied observation up to time t
     (aggregated per timestamp): max of t1, t2, p; min of v; and t1-max(t1), t2-max(t2), v-min(v).
Final probability = 0.5 * MLP + 0.5 * tree. All imputation/scaling/image standardisation/PCA are fitted
on Train only and stored; inference only applies them.
"""
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import base_model as B
import baseline_v7 as BL

FORMAT_VERSION = 7
MODEL_FAMILY = "hybrid_cum"
SEEDS = (42, 43, 44)
EPOCHS = 16
PCA_COMPONENTS = 32
MIXTURE = (0.5, 0.5)
CURRENT_NAMES = B.SENSORS + ["t2_minus_t1", "t_mean", "t_max", "v_times_a"]
WINDOW_NAMES = [f"{s}_{k}30s" for s in BL.SENSORS for k in ("mean", "max", "delta")]
CUM_NAMES = ["t1_cummax", "t2_cummax", "p_cummax", "v_cummin", "t1_minus_cummax", "t2_minus_cummax", "v_minus_cummin"]
HGB_PARAMS = dict(learning_rate=.06, max_iter=250, max_leaf_nodes=15, min_samples_leaf=32,
                  l2_regularization=5.0, early_stopping=False, random_state=42)
PACKAGES = ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "pillow", "joblib")
BASELINE_SHA256 = hashlib.sha256(Path(BL.__file__).read_bytes()).hexdigest()


def package_versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def read_csv(path):
    frame = B.read_csv(path)
    if "experiment_id" not in frame or frame.experiment_id.isna().any():
        raise ValueError("Nonmissing experiment_id required")
    t = pd.to_numeric(frame["relative_time_s"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(t).all():
        raise ValueError("relative_time_s must be finite")
    return frame


def cumulative_features(frame):
    """Running extremes since the first supplied observation, per experiment, inclusive of time t.
    Aggregated per timestamp (max / min over same-time rows) so row order never matters; +-inf -> NaN first."""
    x = {c: pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float, copy=True)  # own copy: never edit the input
         for c in ("meas_t1", "meas_t2", "meas_p", "meas_v")}
    for c in x:
        x[c][~np.isfinite(x[c])] = np.nan
    times = pd.to_numeric(frame["relative_time_s"], errors="coerce").to_numpy(dtype=float)
    out = np.full((len(frame), 4), np.nan)
    for _, pos in frame.groupby(frame.experiment_id.to_numpy(), sort=False).indices.items():
        uniq, inv = np.unique(times[pos], return_inverse=True)
        for j, (col, kind) in enumerate((("meas_t1", "max"), ("meas_t2", "max"), ("meas_p", "max"), ("meas_v", "min"))):
            per_time = pd.Series(x[col][pos]).groupby(inv).agg(kind).reindex(range(len(uniq)))
            running = (per_time.cummax() if kind == "max" else per_time.cummin()).ffill().to_numpy(dtype=float)
            out[pos, j] = running[inv]
    cum = pd.DataFrame(out, index=frame.index, columns=CUM_NAMES[:4])
    cum["t1_minus_cummax"] = x["meas_t1"] - cum["t1_cummax"].to_numpy()
    cum["t2_minus_cummax"] = x["meas_t2"] - cum["t2_cummax"].to_numpy()
    cum["v_minus_cummin"] = x["meas_v"] - cum["v_cummin"].to_numpy()
    return cum


def tables(frame):
    """(current 9, tree table = current 9 + window 15 + running-extreme 7), frame row order, causal."""
    current = pd.DataFrame(B.numeric_features(frame), index=frame.index, columns=CURRENT_NAMES)
    base = frame[["sample_id", "experiment_id"]].copy()
    base["relative_time_s"] = pd.to_numeric(frame["relative_time_s"], errors="coerce")
    base[BL.SENSORS] = BL.numeric(frame, BL.SENSORS)
    windows = BL.add_causal_features(base, BL.SENSORS)
    if list(windows.columns) != WINDOW_NAMES:
        raise ValueError("Window feature schema mismatch")
    return current, pd.concat([current, windows, cumulative_features(frame)], axis=1)


def fit_prep(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
    median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
    filled = np.where(np.isnan(x), median, x)
    mean, std = filled.mean(0), filled.std(0)
    std[std < 1e-8] = 1.
    return {"median": median.tolist(), "mean": mean.tolist(), "std": std.tolist()}


def apply_prep(x, prep):
    median, mean, std = (np.asarray(prep[k], dtype=np.float64) for k in ("median", "mean", "std"))
    x = np.asarray(x, dtype=np.float64)
    if x.shape[1] != len(median):
        raise ValueError("Preprocessing width mismatch")
    x = np.where(np.isfinite(x), x, np.nan)
    return ((np.where(np.isnan(x), median, x) - mean) / std).clip(-20, 20).astype(np.float32)


def fit(frame, raw_images, device, epochs=EPOCHS, tree_iterations=None, pca_components=PCA_COMPONENTS):
    """Train-only fit; returns (bundle for model.pt, trees for tree.joblib)."""
    target = B.labels(frame)
    if set(target) != set(range(6)):
        raise ValueError("Training data must contain all six classes")
    if tuple(raw_images.shape) != (len(frame), 512) or not torch.isfinite(raw_images).all():
        raise ValueError("Invalid image embeddings")
    current, bs = tables(frame)
    prep_cur, prep_bs = fit_prep(current.to_numpy()), fit_prep(bs.to_numpy())
    sensors = apply_prep(current.to_numpy(), prep_cur)
    mean = raw_images.mean(0)
    std = raw_images.std(0, unbiased=False).clamp_min(1e-6)
    images = (raw_images - mean) / std
    states = []
    for seed in SEEDS:
        B.seed_all(seed)
        net = B.Fusion().to(device)
        loader = DataLoader(TensorDataset(images, torch.from_numpy(sensors), torch.from_numpy(target)),
                            batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs):
            net.train()
            for image, sensor, y in loader:
                optimizer.zero_grad(set_to_none=True)
                criterion(net(image.to(device), sensor.to(device)), y.to(device)).backward()
                optimizer.step()
        states.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        print(f"Fusion seed {seed}: trained {epochs} fixed epochs", flush=True)
        del net, optimizer
    pca = PCA(n_components=min(pca_components, len(frame) - 1, 512), svd_solver="randomized", random_state=42)
    reduced = pca.fit_transform(images.numpy())
    params = HGB_PARAMS if tree_iterations is None else {**HGB_PARAMS, "max_iter": tree_iterations}
    hgb = HistGradientBoostingClassifier(**params)
    hgb.fit(np.column_stack([apply_prep(bs.to_numpy(), prep_bs), reduced]), target)
    if not np.array_equal(hgb.classes_, np.arange(6)):
        raise ValueError("Unexpected class order")
    bundle = {"format_version": FORMAT_VERSION, "model_family": MODEL_FAMILY,
              "current_names": CURRENT_NAMES, "window_names": WINDOW_NAMES, "cum_names": CUM_NAMES,
              "prep_current": prep_cur, "prep_tree": prep_bs, "image_mean": mean, "image_std": std,
              "states": states, "seeds": list(SEEDS), "epochs": epochs, "hgb_params": params,
              "mixture": list(MIXTURE), "class_labels": [1, 2, 3, 4, 5, 6], "tree_file": "tree.joblib",
              "baseline_v7_sha256": BASELINE_SHA256, "environment_versions": package_versions()}
    return bundle, {"pca": pca, "hgb": hgb}


@torch.inference_mode()
def predict(frame, raw_images, bundle, trees, device, return_components=False):
    if bundle.get("format_version") != FORMAT_VERSION or bundle.get("model_family") != MODEL_FAMILY:
        raise ValueError("Checkpoint is not a hybrid_cum model")
    if (bundle["current_names"] != CURRENT_NAMES or bundle["window_names"] != WINDOW_NAMES
            or bundle.get("cum_names") != CUM_NAMES
            or bundle["class_labels"] != [1, 2, 3, 4, 5, 6] or tuple(bundle["seeds"]) != SEEDS
            or len(bundle["states"]) != len(SEEDS) or list(bundle["mixture"]) != list(MIXTURE)
            or bundle["baseline_v7_sha256"] != BASELINE_SHA256):
        raise ValueError("Checkpoint schema differs from this code")
    if tuple(raw_images.shape) != (len(frame), 512) or not torch.isfinite(raw_images).all():
        raise ValueError("Invalid inference image features")
    if not np.array_equal(trees["hgb"].classes_, np.arange(6)):
        raise ValueError("Tree class mapping differs")
    current, bs = tables(frame)
    sensors = apply_prep(current.to_numpy(), bundle["prep_current"])
    images = (raw_images - bundle["image_mean"]) / bundle["image_std"]
    seed_probs = []
    for state in bundle["states"]:
        net = B.Fusion().to(device)
        net.load_state_dict(state)
        net.eval()
        chunks = []
        for start in range(0, len(frame), 256):
            logits = net(images[start:start + 256].to(device), torch.from_numpy(sensors[start:start + 256]).to(device))
            chunks.append(logits.softmax(1).cpu().numpy())
        seed_probs.append(np.concatenate(chunks).astype(np.float64))
        del net
    mlp = np.mean(seed_probs, axis=0)
    reduced = trees["pca"].transform(images.numpy())
    tree = trees["hgb"].predict_proba(np.column_stack([apply_prep(bs.to_numpy(), bundle["prep_tree"]), reduced]))
    probability = MIXTURE[0] * mlp + MIXTURE[1] * tree
    for v in (mlp, tree, probability):
        if v.shape != (len(frame), 6) or not np.isfinite(v).all() or (v < 0).any() or not np.allclose(v.sum(1), 1, atol=1e-5):
            raise ValueError("Invalid class probabilities")
    return (probability, {"mlp": mlp, "tree": tree}) if return_components else probability


def metrics(target, probability):
    prediction = probability.argmax(1)
    return {"macro_f1": float(f1_score(target, prediction, labels=list(range(6)), average="macro", zero_division=0)),
            "class_f1": f1_score(target, prediction, labels=list(range(6)), average=None, zero_division=0).tolist(),
            "confusion_matrix": confusion_matrix(target, prediction, labels=list(range(6))).tolist()}
