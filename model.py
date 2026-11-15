"""hybrid_rewm: hybrid_epair with three additions (exp_t39 arms REW / REWM; the variant is chosen at training time).

  1. R: the Fusion MLP's 9 current-row sensor inputs are scaled with the Train median and inter-quartile range
     (non-finite -> median, IQR < 1e-8 -> 1) and clipped to +-20, instead of mean / standard deviation.
  2. EW: the tree and the pair auxiliaries also read the organizer baseline's window features with a 60 s window
     (mean, max, delta for the 5 sensors; 46 -> 61 sensor inputs).
  3. REWM only: a second, wide Fusion MLP (image embedding + the 61 tree sensor inputs with the tree preprocessing,
     seeds 42/43/44, 16 epochs) is mixed in before the pair step:
        base = 0.5 * MLP9 + 0.5 * tree;  mixed = 0.75 * base + 0.25 * (0.5 * MLP61 + 0.5 * tree)
     REW uses base directly. The pair auxiliaries then act on the top-2 of that probability as in hybrid_pair.

(hybrid_epair description follows.)
hybrid_epair: hybrid_pair whose tree and pair auxiliaries also read 15 cumulative event features (exp_t24 arm EPAIR).

For x in (meas_v, meas_t1, meas_t2, meas_p, thermal_t), same experiment, time <= t, per-timestamp mean, missing values
carried forward from the last observed value (leading gaps stay missing):
  dd_abs = max_{s<=t} (M_s - x_s), dd_rel = max_{s<=t} (M_s - x_s) / max(|M_s|, 1e-3), rise = max_{s<=t} (x_s - m_s),
with M / m the running max / min. No row-to-row time differences are used. The MLP is unchanged.

(hybrid_pair description follows.)
hybrid_pair: hybrid_cum + sensor-only adjacent-pair auxiliary classifiers (exp_t20 arm PAIR).

After the hybrid_cum probability p is formed, for the stage pairs (3,4) and (5,6): a binary HGB (same parameters as
the main tree) trained on the Train rows of those two stages, using only the 31 sensor features of the main tree
(no image PCA), estimates q = P(higher stage | one of the two). It is applied only to rows whose top-2 classes of p are
exactly that pair (stable sort: ties go to the lower class index); the pair's total probability is kept and its
inner ratio r = p_hi / (p_lo + p_hi) becomes 0.5 * r + 0.5 * q. Other classes are unchanged.

(Original hybrid_cum description follows.)
hybrid_cum: Hybrid v2 + organizer baseline v7 30 s window features + running-extreme features in the tree.

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

FORMAT_VERSION = 18
MODEL_FAMILY = "hybrid_rewm"
VARIANTS = ("REW", "REWM")
WIDE_MIX = 0.25
EVT_SIGNALS = ["meas_v", "meas_t1", "meas_t2", "meas_p", "thermal_t"]
EVT_NAMES = [f"{s}_{k}" for s in EVT_SIGNALS for k in ("dd_abs", "dd_rel", "rise")]
PAIRS = ((2, 3), (4, 5))  # 0-based: stages (3,4) and (5,6)
PAIR_ALPHA = 0.5
SEEDS = (42, 43, 44)
EPOCHS = 16
PCA_COMPONENTS = 32
MIXTURE = (0.5, 0.5)
CURRENT_NAMES = B.SENSORS + ["t2_minus_t1", "t_mean", "t_max", "v_times_a"]
WINDOW_NAMES = [f"{s}_{k}30s" for s in BL.SENSORS for k in ("mean", "max", "delta")]
W60_NAMES = [f"{s}_{k}60s" for s in BL.SENSORS for k in ("mean", "max", "delta")]
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
    if "thermal_t" not in frame.columns:
        raise ValueError("Input CSV lacks the observed column thermal_t")
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


def event_features(frame):
    x = frame[EVT_SIGNALS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    times = pd.to_numeric(frame["relative_time_s"], errors="coerce").to_numpy(dtype=float)
    out = np.full((len(frame), len(EVT_NAMES)), np.nan)
    for _, pos in frame.groupby(frame.experiment_id.to_numpy(), sort=False).indices.items():
        uniq, inv = np.unique(times[pos], return_inverse=True)
        for j, s in enumerate(EVT_SIGNALS):
            v = pd.Series(x[s].to_numpy()[pos]).groupby(inv).mean().reindex(range(len(uniq))).ffill().to_numpy(dtype=float)
            mx = pd.Series(v).cummax().to_numpy()
            mn = pd.Series(v).cummin().to_numpy()
            out[pos, 3 * j] = pd.Series(mx - v).cummax().to_numpy()[inv]
            out[pos, 3 * j + 1] = pd.Series((mx - v) / np.maximum(np.abs(mx), 1e-3)).cummax().to_numpy()[inv]
            out[pos, 3 * j + 2] = pd.Series(v - mn).cummax().to_numpy()[inv]
    return pd.DataFrame(out, index=frame.index, columns=EVT_NAMES)


def tables(frame):
    """(current 9, tree table = current 9 + 30 s window 15 + running-extreme 7 + event 15 + 60 s window 15), causal."""
    current = pd.DataFrame(B.numeric_features(frame), index=frame.index, columns=CURRENT_NAMES)
    base = frame[["sample_id", "experiment_id"]].copy()
    base["relative_time_s"] = pd.to_numeric(frame["relative_time_s"], errors="coerce")
    base[BL.SENSORS] = BL.numeric(frame, BL.SENSORS)
    windows = BL.add_causal_features(base, BL.SENSORS)
    w60 = BL.add_causal_features(base, BL.SENSORS, window_s=60.0)
    if list(windows.columns) != WINDOW_NAMES or list(w60.columns) != W60_NAMES:
        raise ValueError("Window feature schema mismatch")
    return current, pd.concat([current, windows, cumulative_features(frame), event_features(frame), w60], axis=1)


class FusionX(nn.Module):
    """Hybrid v2 Fusion layout with a configurable sensor width."""
    def __init__(self, sensor_dim):
        super().__init__()
        self.image = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.2))
        self.sensor = nn.Sequential(nn.Linear(sensor_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(160, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, 6))

    def forward(self, image, sensor):
        return self.head(torch.cat([self.image(image), self.sensor(sensor)], dim=1))


def fit_robust(x):
    x = np.asarray(x, dtype=np.float64)
    median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
    iqr = np.array([np.subtract(*np.quantile(c[np.isfinite(c)], [.75, .25])) if np.isfinite(c).any() else 1. for c in x.T])
    iqr[iqr < 1e-8] = 1.
    return {"median": median.tolist(), "iqr": iqr.tolist()}


def apply_robust(x, prep):
    median, iqr = (np.asarray(prep[k], dtype=np.float64) for k in ("median", "iqr"))
    x = np.asarray(x, dtype=np.float64)
    if x.shape[1] != len(median) or len(iqr) != len(median) or not (iqr > 0).all():
        raise ValueError("Preprocessing width mismatch")
    return ((np.where(np.isfinite(x), x, median) - median) / iqr).clip(-20, 20).astype(np.float32)


def train_mlps(make_net, images, sensors, target, device, epochs, tag):
    states = []
    for seed in SEEDS:
        B.seed_all(seed)
        net = make_net().to(device)
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
        print(f"{tag} seed {seed}: trained {epochs} fixed epochs", flush=True)
        del net, optimizer
    return states


@torch.inference_mode()
def mlp_probability(make_net, states, images, sensors, device):
    seed_probs = []
    for state in states:
        net = make_net().to(device)
        net.load_state_dict(state)
        net.eval()
        chunks = []
        for start in range(0, len(sensors), 256):
            logits = net(images[start:start + 256].to(device), torch.from_numpy(sensors[start:start + 256]).to(device))
            chunks.append(logits.softmax(1).cpu().numpy())
        seed_probs.append(np.concatenate(chunks).astype(np.float64))
        del net
    return np.mean(seed_probs, axis=0)


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


def apply_pairs(p, q_by_pair, alpha=PAIR_ALPHA):
    """Keep each pair's total; move its inner ratio toward the auxiliary estimate on rows whose top-2 is the pair."""
    out = p.copy()
    top2 = np.sort(np.argsort(-p, axis=1, kind="stable")[:, :2], axis=1)
    for (i, j), q in zip(PAIRS, q_by_pair):
        rows = (top2[:, 0] == i) & (top2[:, 1] == j)
        s = p[rows, i] + p[rows, j]
        r = p[rows, j] / np.maximum(s, 1e-12)
        r2 = (1 - alpha) * r + alpha * q[rows]
        out[rows, i], out[rows, j] = s * (1 - r2), s * r2
    return out


def fit(frame, raw_images, device, epochs=EPOCHS, tree_iterations=None, pca_components=PCA_COMPONENTS, variant="REWM"):
    """Train-only fit; returns (bundle for model.pt, trees for tree.joblib)."""
    if variant not in VARIANTS:
        raise ValueError("Unknown variant")
    target = B.labels(frame)
    if set(target) != set(range(6)):
        raise ValueError("Training data must contain all six classes")
    if tuple(raw_images.shape) != (len(frame), 512) or not torch.isfinite(raw_images).all():
        raise ValueError("Invalid image embeddings")
    current, bs = tables(frame)
    prep_cur, prep_bs = fit_robust(current.to_numpy()), fit_prep(bs.to_numpy())
    sensors = apply_robust(current.to_numpy(), prep_cur)
    mean = raw_images.mean(0)
    std = raw_images.std(0, unbiased=False).clamp_min(1e-6)
    images = (raw_images - mean) / std
    states = train_mlps(B.Fusion, images, sensors, target, device, epochs, "Fusion (robust 9 sensors)")
    pca = PCA(n_components=min(pca_components, len(frame) - 1, 512), svd_solver="randomized", random_state=42)
    reduced = pca.fit_transform(images.numpy())
    params = HGB_PARAMS if tree_iterations is None else {**HGB_PARAMS, "max_iter": tree_iterations}
    hgb = HistGradientBoostingClassifier(**params)
    hgb.fit(np.column_stack([apply_prep(bs.to_numpy(), prep_bs), reduced]), target)
    if not np.array_equal(hgb.classes_, np.arange(6)):
        raise ValueError("Unexpected class order")
    sensor_tree = apply_prep(bs.to_numpy(), prep_bs)
    pair_aux = []
    for i, j in PAIRS:
        m = (target == i) | (target == j)
        aux = HistGradientBoostingClassifier(**params).fit(sensor_tree[m], (target[m] == j).astype(int))  # sensors only
        if not np.array_equal(aux.classes_, [0, 1]):
            raise ValueError("Unexpected pair class order")
        pair_aux.append(aux)
    width = sensor_tree.shape[1]
    wide_states = (train_mlps(lambda: FusionX(width), images, sensor_tree, target, device, epochs, f"Wide fusion ({width} sensors)")
                   if variant == "REWM" else [])
    bundle = {"format_version": FORMAT_VERSION, "model_family": MODEL_FAMILY,
              "current_names": CURRENT_NAMES, "window_names": WINDOW_NAMES, "cum_names": CUM_NAMES,
              "prep_current": prep_cur, "prep_tree": prep_bs, "image_mean": mean, "image_std": std,
              "states": states, "seeds": list(SEEDS), "epochs": epochs, "hgb_params": params,
              "mixture": list(MIXTURE), "class_labels": [1, 2, 3, 4, 5, 6], "tree_file": "tree.joblib",
              "baseline_v7_sha256": BASELINE_SHA256, "environment_versions": package_versions(),
              "pairs": [list(x) for x in PAIRS], "pair_alpha": PAIR_ALPHA, "evt_names": EVT_NAMES,
              "w60_names": W60_NAMES, "variant": variant, "wide_states": wide_states, "wide_mix": WIDE_MIX,
              "wide_width": width}
    return bundle, {"pca": pca, "hgb": hgb, "pair_aux": pair_aux}


def predict(frame, raw_images, bundle, trees, device, return_components=False):
    if bundle.get("format_version") != FORMAT_VERSION or bundle.get("model_family") != MODEL_FAMILY:
        raise ValueError("Checkpoint is not a hybrid_rewm model")
    if (bundle["current_names"] != CURRENT_NAMES or bundle["window_names"] != WINDOW_NAMES
            or bundle.get("cum_names") != CUM_NAMES
            or bundle["class_labels"] != [1, 2, 3, 4, 5, 6] or tuple(bundle["seeds"]) != SEEDS
            or len(bundle["states"]) != len(SEEDS) or list(bundle["mixture"]) != list(MIXTURE)
            or bundle["baseline_v7_sha256"] != BASELINE_SHA256
            or bundle.get("pairs") != [list(x) for x in PAIRS] or bundle.get("pair_alpha") != PAIR_ALPHA
            or bundle.get("evt_names") != EVT_NAMES or bundle.get("w60_names") != W60_NAMES
            or bundle.get("variant") not in VARIANTS or bundle.get("wide_mix") != WIDE_MIX
            or len(bundle.get("wide_states", [])) != (len(SEEDS) if bundle.get("variant") == "REWM" else 0)
            or set(bundle["prep_current"]) != {"median", "iqr"}):
        raise ValueError("Checkpoint schema differs from this code")
    if len(trees.get("pair_aux", [])) != len(PAIRS) or any(not np.array_equal(a.classes_, [0, 1]) for a in trees["pair_aux"]):
        raise ValueError("Pair auxiliary classifiers missing or mis-ordered")
    if tuple(raw_images.shape) != (len(frame), 512) or not torch.isfinite(raw_images).all():
        raise ValueError("Invalid inference image features")
    if not np.array_equal(trees["hgb"].classes_, np.arange(6)):
        raise ValueError("Tree class mapping differs")
    current, bs = tables(frame)
    sensors = apply_robust(current.to_numpy(), bundle["prep_current"])
    images = (raw_images - bundle["image_mean"]) / bundle["image_std"]
    mlp = mlp_probability(B.Fusion, bundle["states"], images, sensors, device)
    reduced = trees["pca"].transform(images.numpy())
    sensor_tree = apply_prep(bs.to_numpy(), bundle["prep_tree"])
    tree = trees["hgb"].predict_proba(np.column_stack([sensor_tree, reduced]))
    cum = MIXTURE[0] * mlp + MIXTURE[1] * tree
    if bundle["variant"] == "REWM":
        if sensor_tree.shape[1] != bundle["wide_width"]:
            raise ValueError("Wide MLP width mismatch")
        wide = mlp_probability(lambda: FusionX(bundle["wide_width"]), bundle["wide_states"], images, sensor_tree, device)
        cum = (1 - WIDE_MIX) * cum + WIDE_MIX * (MIXTURE[0] * wide + MIXTURE[1] * tree)
    probability = apply_pairs(cum, [a.predict_proba(sensor_tree)[:, 1] for a in trees["pair_aux"]], bundle["pair_alpha"])
    for v in (mlp, tree, cum, probability):
        if v.shape != (len(frame), 6) or not np.isfinite(v).all() or (v < 0).any() or not np.allclose(v.sum(1), 1, atol=1e-5):
            raise ValueError("Invalid class probabilities")
    return (probability, {"mlp": mlp, "tree": tree, "cum": cum}) if return_components else probability


def metrics(target, probability):
    prediction = probability.argmax(1)
    return {"macro_f1": float(f1_score(target, prediction, labels=list(range(6)), average="macro", zero_division=0)),
            "class_f1": f1_score(target, prediction, labels=list(range(6)), average=None, zero_division=0).tolist(),
            "confusion_matrix": confusion_matrix(target, prediction, labels=list(range(6))).tolist()}
