"""Current-row and causal sensor features; no image or training dependencies."""
import numpy as np
import pandas as pd

SENSORS = ["meas_v", "meas_a", "meas_t1", "meas_t2", "meas_p"]
CURRENT_NAMES = SENSORS + ["t2_minus_t1", "temperature_mean", "temperature_max", "v_times_a"]
FEATURE_SETS = ("current", "temperature", "voltage", "cooling", "all")
WINDOWS = (10., 30., 60.)


def feature_names(feature_set):
    if feature_set not in FEATURE_SETS:
        raise ValueError("Unknown feature set")
    names = list(CURRENT_NAMES)
    if feature_set in ("temperature", "all"):
        names += [f"{sensor}_{kind}{int(window)}s"
                  for sensor in ("meas_t1", "meas_t2")
                  for window in WINDOWS for kind in ("delta", "rate")]
    if feature_set in ("voltage", "all"):
        names += ["voltage_drop_from_past_peak", "voltage_drop_fraction"]
    if feature_set in ("cooling", "all"):
        names += [f"{sensor}_drop_from_past_peak" for sensor in ("meas_t1", "meas_t2")]
    return names


def build_sensor_features(frame, feature_set="all"):
    """Return rows in exactly the input order; use only supplied causal history.

    Lag values: mean at the latest supplied timestamp <= t - window. A rate
    divides by the actual elapsed time, including when observations are sparse.
    Historical peaks include every observation at a timestamp <= current t.
    Same-time aggregation removes any dependence on the order of tied rows.
    Missing historical observations remain NaN for TRAIN-fitted imputation.
    """
    names = feature_names(feature_set)
    values = frame[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    v, a, t1, t2, _ = values.T
    with np.errstate(invalid="ignore", over="ignore"):
        current = np.column_stack([values, t2-t1, (t1+t2)/2, np.maximum(t1, t2), v*a])
    current[~np.isfinite(current)] = np.nan
    out = np.full((len(frame), len(names)), np.nan, dtype=float)
    out[:, :len(CURRENT_NAMES)] = current
    if feature_set == "current":
        return pd.DataFrame(out, index=frame.index, columns=names)
    required = {"experiment_id", "relative_time_s"}
    if not required.issubset(frame.columns) or frame.experiment_id.isna().any():
        raise ValueError("Temporal features require nonmissing experiment_id and relative_time_s")
    times = pd.to_numeric(frame.relative_time_s, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(times).all():
        raise ValueError("relative_time_s must be finite")
    work = pd.DataFrame(values, columns=SENSORS)
    work["experiment_id"] = frame.experiment_id.to_numpy()
    work["time"] = times
    destinations = {name:i for i,name in enumerate(names)}
    for _, positions in work.groupby("experiment_id", sort=False).indices.items():
        group = work.iloc[positions]
        # Means and maxima differ for tied timestamps. Keep each role explicit.
        grouped = group.groupby("time", sort=True)[["meas_v", "meas_t1", "meas_t2"]]
        means = grouped.mean()
        maxima = grouped.max()
        history_times = means.index.to_numpy(dtype=float)
        row_times = times[positions]
        row_timestamp = np.searchsorted(history_times, row_times)
        if feature_set in ("temperature", "all"):
            for sensor in ("meas_t1", "meas_t2"):
                actual = values[positions, SENSORS.index(sensor)]
                historical = means[sensor].to_numpy(dtype=float)
                for window in WINDOWS:
                    lag = np.searchsorted(history_times, row_times-window, side="right")-1
                    valid = lag >= 0
                    delta = np.full(len(positions), np.nan)
                    rate = np.full(len(positions), np.nan)
                    delta[valid] = actual[valid]-historical[lag[valid]]
                    rate[valid] = delta[valid]/(row_times[valid]-history_times[lag[valid]])
                    out[positions, destinations[f"{sensor}_delta{int(window)}s"]] = delta
                    out[positions, destinations[f"{sensor}_rate{int(window)}s"]] = rate
        if feature_set in ("voltage", "cooling", "all"):
            for sensor in ("meas_v", "meas_t1", "meas_t2"):
                if sensor == "meas_v" and feature_set not in ("voltage", "all"):
                    continue
                if sensor != "meas_v" and feature_set not in ("cooling", "all"):
                    continue
                # pandas cummax skips missing values; ffill retains earlier peaks.
                peaks = maxima[sensor].cummax().ffill().to_numpy(dtype=float)[row_timestamp]
                drop = peaks-values[positions, SENSORS.index(sensor)]
                if sensor == "meas_v":
                    out[positions, destinations["voltage_drop_from_past_peak"]] = drop
                    fraction = np.full(len(positions), np.nan)
                    valid = np.isfinite(peaks) & (np.abs(peaks) >= 1e-6)
                    fraction[valid] = drop[valid]/np.abs(peaks[valid])
                    out[positions, destinations["voltage_drop_fraction"]] = fraction
                else:
                    out[positions, destinations[f"{sensor}_drop_from_past_peak"]] = drop
    out[~np.isfinite(out)] = np.nan
    return pd.DataFrame(out, index=frame.index, columns=names)


def fit_preprocessing(features):
    x = features.to_numpy(dtype=float)
    median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
    imputed = np.where(np.isfinite(x), x, median)
    mean, std = imputed.mean(0), imputed.std(0)
    std[std < 1e-8] = 1.
    return {"median": median, "mean": mean, "std": std}


def apply_preprocessing(features, prep):
    x = features.to_numpy(dtype=float)
    median, mean, std = [np.asarray(prep[k], dtype=float) for k in ("median", "mean", "std")]
    if any(v.shape != (x.shape[1],) for v in (median, mean, std)) or (std <= 0).any():
        raise ValueError("Preprocessing dimensions or scales are invalid")
    imputed = np.where(np.isfinite(x), x, median)
    return ((imputed-mean)/std).clip(-20, 20).astype(np.float32)
