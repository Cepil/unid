"""Inference only: local versioned checkpoint, sibling tree, supplied CSV/images."""
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
import hybrid as H
import model as M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent/"weights/model.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid inference parameters")
    M.seed_all()
    saved = torch.load(args.model, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 3 or saved.get("model_family") != "hybrid_v2_temporal":
        raise ValueError("Use the matching code for the original format-1 or format-2 checkpoints")
    filename = Path(saved["tree_file"])
    if filename.name != str(filename) or filename.name != "tree.joblib":
        raise ValueError("Tree must be the trained sibling tree.joblib")
    tree_path = args.model.parent/filename
    if args.output.resolve() in (args.model.resolve(), tree_path.resolve(), args.input.resolve()):
        raise ValueError("Output must not overwrite input or model artifacts")
    expected = saved["environment_versions"]["scikit-learn"]
    if importlib.metadata.version("scikit-learn") != expected:
        raise RuntimeError(f"Use the trained scikit-learn version: {expected}. Keep the GPU torch build unchanged.")
    manifest_path = args.model.parent/"manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for path in (args.model, tree_path):
            key = "model.pt" if path == args.model else "tree.joblib"
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][key]:
                raise ValueError("Model/tree artifact checksum mismatch")
    # Deserializes only the team's trusted, locally trained artifact.
    tree = joblib.load(tree_path)
    frame = M.read_csv(args.input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = M.encoder()
    backbone.load_state_dict(saved["encoder"])
    raw = M.extract(frame, [args.image_root, args.image_root.parent], backbone.to(device),
                    device, args.batch_size, args.workers)
    predictions = H.predict(frame, raw, saved, tree, device).argmax(1)+1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id":frame.sample_id, "label":predictions.astype(np.int64)}).to_csv(args.output, index=False)
    print("Submission saved in original input order.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
