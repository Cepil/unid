"""Shared private cache, split validation and versioned artifact storage."""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import uuid
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
import model as M


def load_encoder(path):
    source = torch.load(path, map_location="cpu", weights_only=True)
    if "encoder" not in source:
        raise ValueError("Source checkpoint needs frozen ResNet-18 weights")
    encoder = M.encoder()
    encoder.load_state_dict(source["encoder"])
    return encoder, source["encoder"], hashlib.sha256(path.read_bytes()).hexdigest()


def embeddings(frame, csv, roots, cache, backbone, device, encoder_sha256, batch_size=32, workers=2):
    digest = hashlib.sha256(csv.read_bytes())
    digest.update(encoder_sha256.encode())
    digest.update(inspect.getsource(M.Images).encode())
    digest.update(inspect.getsource(M.extract).encode())
    for package in ("torch", "torchvision", "pillow"):
        digest.update(importlib.metadata.version(package).encode())
    paths = M.Images(frame, roots).paths
    for path in paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    fingerprint = digest.hexdigest()
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved.get("fingerprint") == fingerprint:
            value = saved["embeddings"]
            if tuple(value.shape) != (len(frame), 512) or not torch.isfinite(value).all():
                raise ValueError("Invalid private image cache")
            print("Using matching private frozen-image cache", flush=True)
            return value
    value = M.extract(frame, roots, backbone, device, batch_size, workers)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(cache.name+"."+uuid.uuid4().hex+".tmp")
    torch.save({"fingerprint":fingerprint, "embeddings":value}, temporary)
    temporary.replace(cache)
    return value


def make_splits(frame, folds=5, assignment=None):
    if "experiment_id" not in frame or frame.experiment_id.isna().any():
        raise ValueError("Nonmissing experiment IDs required")
    if assignment is None:
        splits = list(GroupKFold(n_splits=folds).split(frame, groups=frame.experiment_id))
    else:
        mapping = pd.read_csv(assignment, dtype={"sample_id":"string"})
        if not {"sample_id", "fold"}.issubset(mapping.columns) or mapping.sample_id.isna().any() or not mapping.sample_id.is_unique:
            raise ValueError("Invalid private fold mapping")
        if set(mapping.sample_id) != set(frame.sample_id):
            raise ValueError("Fold mapping must match Train IDs exactly")
        aligned = mapping.set_index("sample_id").loc[frame.sample_id, "fold"].to_numpy()
        if pd.isna(aligned).any() or len(pd.unique(aligned)) != folds:
            raise ValueError("Fold mapping differs from requested fold count")
        splits = [(np.flatnonzero(aligned != f), np.flatnonzero(aligned == f)) for f in pd.unique(aligned)]
    target = M.labels(frame)
    seen = np.zeros(len(frame), dtype=int)
    for tr,va in splits:
        if set(frame.iloc[tr].experiment_id) & set(frame.iloc[va].experiment_id):
            raise ValueError("An experiment spans multiple folds")
        if set(target[tr]) != set(range(6)):
            raise ValueError("Each fold Train must contain all six classes")
        seen[va] += 1
    if not np.all(seen == 1):
        raise ValueError("Each Train row must be held out exactly once")
    return splits


def save_candidate(directory, bundle, tree, encoder_state, source_sha256):
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Existing model artifacts are preserved; choose a new directory")
    directory.mkdir(parents=True, exist_ok=True)
    bundle = {**bundle, "encoder":encoder_state, "encoder_source_sha256":source_sha256}
    joblib.dump(tree, directory/"tree.joblib")
    torch.save(bundle, directory/"model.pt")
    description = {"model_family":bundle["model_family"], "recipe":bundle["recipe"],
                   "epochs":bundle["epochs"], "seeds":bundle["seeds"],
                   "environment_versions":bundle["environment_versions"], "encoder_source_sha256":source_sha256}
    description["files"] = {name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ("model.pt", "tree.joblib")}
    (directory/"manifest.json").write_text(json.dumps(description, indent=2), encoding="utf-8")
