"""Fit hybrid_rewm (variant REW or REWM) on provided Train only, score provided Validation once, save artifacts.

python train.py --data_dir ~/data/battery_data_light --output_dir weights [--variant REWM]
The encoder is torchvision ResNet-18 IMAGENET1K_V1 (downloaded once by torchvision), frozen. For an offline
machine pass --source_checkpoint <file with an 'encoder' state of the same frozen backbone>.
Existing output folders are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import torch
from threadpoolctl import threadpool_limits

import base_model as B
import model as M


def load_backbone(args):
    if args.source_checkpoint is None:
        return B.encoder(pretrained=True), "torchvision:ResNet18_Weights.IMAGENET1K_V1"
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=True)
    if "encoder" not in source:
        raise ValueError("Source checkpoint has no 'encoder' state")
    backbone = B.encoder(pretrained=False)
    backbone.load_state_dict(source["encoder"])
    return backbone, "checkpoint:" + hashlib.sha256(args.source_checkpoint.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_checkpoint", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--variant", choices=M.VARIANTS, default="REWM")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose a fresh --output_dir; existing artifacts are preserved")
    B.seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = {s: M.read_csv(args.data_dir / s / f"{s}.csv") for s in ("train", "validation")}
    train, valid = frames["train"], frames["validation"]
    if set(train.experiment_id) & set(valid.experiment_id) or set(train.sample_id) & set(valid.sample_id):
        raise ValueError("Train/Validation overlap")
    backbone, encoder_source = load_backbone(args)
    backbone = backbone.to(device)
    raw = {}
    for split, frame in frames.items():
        print(f"Extracting frozen image features: {split}", flush=True)
        raw[split] = B.extract(frame, [args.data_dir / split, args.data_dir], backbone, device,
                               args.batch_size, args.workers)
    encoder_state = {k: v.cpu() for k, v in backbone.state_dict().items()}
    del backbone
    bundle, trees = M.fit(train, raw["train"], device, variant=args.variant)  # Train only
    result = M.metrics(B.labels(valid), M.predict(valid, raw["validation"], bundle, trees, device))
    bundle = {**bundle, "encoder": encoder_state, "encoder_source": encoder_source}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(trees, args.output_dir / "tree.joblib")
    torch.save(bundle, args.output_dir / "model.pt")
    manifest = {"model_family": M.MODEL_FAMILY, "variant": args.variant, "format_version": M.FORMAT_VERSION, "trained_on": "provided Train only",
                "encoder_source": encoder_source, "baseline_v7_sha256": M.BASELINE_SHA256,
                "environment_versions": bundle["environment_versions"],
                "files": {n: hashlib.sha256((args.output_dir / n).read_bytes()).hexdigest()
                          for n in ("model.pt", "tree.joblib")}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "validation_metrics.json").write_text(json.dumps(
        {"evaluation": "provided Validation, scored once after Train-only fitting", "metrics": result,
         "environment_versions": bundle["environment_versions"]}, indent=2), encoding="utf-8")
    print(f"hybrid_rewm {args.variant}: Validation Macro F1={result['macro_f1']:.6f}", flush=True)
    print("Class F1 (1..6): " + ", ".join(f"{v:.4f}" for v in result["class_f1"]), flush=True)
    print(f"Saved model.pt, tree.joblib, manifest.json in {args.output_dir}. No Test rows loaded.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
