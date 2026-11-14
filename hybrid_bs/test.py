"""Inference only: python test.py --input <CSV> --image_root <image folder> --output <CSV path>

Uses only the rows of --input (same-experiment rows at or before each row's time), the saved encoder,
MLPs, preprocessing, PCA and tree. No fitting, no downloads. Output keeps the input row order.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import base_model as B
import model as M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "weights/model.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid inference parameters")
    B.seed_all(42)
    saved = torch.load(args.model, map_location="cpu", weights_only=True)
    if saved.get("format_version") != M.FORMAT_VERSION or saved.get("model_family") != M.MODEL_FAMILY:
        raise ValueError("This test.py only runs hybrid_bs (format 6) checkpoints")
    tree_path = args.model.parent / "tree.joblib"
    if saved.get("tree_file") != "tree.joblib" or not tree_path.is_file():
        raise ValueError("The trained sibling tree.joblib is required next to model.pt")
    if args.output.resolve() in (args.model.resolve(), tree_path.resolve(), args.input.resolve()):
        raise ValueError("Output must not overwrite input or model artifacts")
    expected = saved["environment_versions"]["scikit-learn"]
    if importlib.metadata.version("scikit-learn") != expected:
        raise RuntimeError(f"Install the trained scikit-learn version {expected}; keep the GPU torch build unchanged.")
    manifest_path = args.model.parent / "manifest.json"
    if manifest_path.exists():
        files = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        for name, path in (("model.pt", args.model), ("tree.joblib", tree_path)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != files[name]:
                raise ValueError(f"{name} checksum differs from manifest.json")
    trees = joblib.load(tree_path)  # the team's own locally trained artifact
    frame = M.read_csv(args.input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = B.encoder(pretrained=False)
    backbone.load_state_dict(saved["encoder"])
    raw = B.extract(frame, [args.image_root, args.image_root.parent], backbone.to(device), device,
                    args.batch_size, args.workers)
    prediction = M.predict(frame, raw, saved, trees, device).argmax(1) + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": frame.sample_id, "label": prediction.astype(np.int64)}).to_csv(args.output, index=False)
    print("Submission saved in input row order.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
