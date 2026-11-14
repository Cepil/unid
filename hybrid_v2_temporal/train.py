"""Fit a preselected Hybrid recipe on Train only; evaluate Validation after fitting."""
import argparse
import json
from pathlib import Path
import torch
from threadpoolctl import threadpool_limits
import hybrid as H
import model as M
from workflow import load_encoder, embeddings, save_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--source_checkpoint", type=Path, required=True)
    parser.add_argument("--recipe", choices=list(H.RECIPES), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=Path("cache"))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid training parameters")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose a fresh --output_dir; existing artifacts are preserved")
    M.seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = {split:M.read_csv(args.data_dir/split/f"{split}.csv") for split in ("train", "validation")}
    train,valid = frames["train"],frames["validation"]
    if any("experiment_id" not in f or f.experiment_id.isna().any() for f in frames.values()):
        raise ValueError("Nonmissing experiment IDs required")
    if set(train.experiment_id) & set(valid.experiment_id) or set(train.sample_id) & set(valid.sample_id):
        raise ValueError("Train/Validation overlap")
    backbone,encoder_state,source_sha = load_encoder(args.source_checkpoint)
    backbone.to(device)
    raw = {}
    for split,frame in frames.items():
        print("Preparing private frozen image features:", split, flush=True)
        raw[split] = embeddings(frame, args.data_dir/split/f"{split}.csv", [args.data_dir/split, args.data_dir],
            args.cache_dir/f"{split}_images.pt", backbone, device, source_sha, args.batch_size, args.workers)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    bundles,trees = H.fit_family(train, raw["train"], device, epochs=args.epochs, recipes=[args.recipe])
    bundle,tree = bundles[args.recipe],trees[args.recipe]
    predicted = H.predict(valid, raw["validation"], bundle, tree, device)
    result = H.metrics(M.labels(valid), predicted)
    save_candidate(args.output_dir, bundle, tree, encoder_state, source_sha)
    metadata = {"evaluation":"Provided Validation, evaluated after fixed-epoch Train-only fitting",
                "trained_on":"Train only", "recipe":args.recipe, "epochs":args.epochs,
                "validation_used_for_fitting":False, "metrics":result,
                "environment_versions":H.package_versions()}
    (args.output_dir/"validation_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{args.recipe}: Validation Macro F1={result['macro_f1']:.6f}", flush=True)
    print("Class F1 (1..6):", ", ".join(f"{value:.4f}" for value in result["class_f1"]), flush=True)
    print("Saved model.pt, tree.joblib and manifests. No Test rows loaded or submission generated.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
