"""Same-fold current Hybrid vs temporal tree vs temporal both; Train only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import hybrid as H
import model as M
from workflow import load_encoder, embeddings, make_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--source_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/paired_cv"))
    parser.add_argument("--cache_dir", type=Path, default=Path("cache"))
    parser.add_argument("--folds_csv", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.folds < 2 or args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid CV parameters")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose a fresh --output_dir")
    M.seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv = args.data_dir/"train/train.csv"
    frame = M.read_csv(csv)
    target = M.labels(frame)
    splits = make_splits(frame, args.folds, args.folds_csv)
    backbone,_,source_sha = load_encoder(args.source_checkpoint)
    raw = embeddings(frame, csv, [args.data_dir/"train", args.data_dir], args.cache_dir/"train_images.pt",
                     backbone.to(device), device, source_sha, args.batch_size, args.workers)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Features remain within experiments. Whole experiments are assigned to one fold.
    tables = {name:M.build_sensor_features(frame, name) for name in ("current", "all")}
    probabilities = {name:np.full((len(frame), 6), np.nan) for name in H.RECIPES}
    scores = {name:[] for name in H.RECIPES}
    assignments = np.full(len(frame), -1, dtype=int)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold,(tr,va) in enumerate(splits, 1):
        print(f"\nPaired fold {fold}/{len(splits)}", flush=True)
        train_tables = {name:table.iloc[tr] for name,table in tables.items()}
        valid_tables = {name:table.iloc[va] for name,table in tables.items()}
        bundles,trees = H.fit_family(frame.iloc[tr], raw[tr], device, train_tables, epochs=args.epochs)
        assignments[va] = fold-1
        for recipe in H.RECIPES:
            predicted = H.predict(frame.iloc[va], raw[va], bundles[recipe], trees[recipe], device, valid_tables)
            probabilities[recipe][va] = predicted
            score = H.metrics(target[va], predicted)["macro_f1"]
            scores[recipe].append(score)
            print(f"fold={fold} recipe={recipe} macro_f1={score:.6f}", flush=True)
        # Per-fold aggregate progress only; no data rows, IDs, or probabilities are logged.
        (args.output_dir/"progress.json").write_text(json.dumps({"completed_folds":fold, "fold_scores":scores}, indent=2), encoding="utf-8")
        del bundles,trees
    result = {"evaluation":"Train-only paired experiment GroupKFold", "epochs":args.epochs,
              "seeds":list(H.SEEDS), "mixture_weights":[.5, .5], "folds":len(splits),
              "environment_versions":H.package_versions(), "encoder_source_sha256":source_sha,
              "variants":{}, "selection_warning":"CV selects candidates; winning CV is not an unbiased final performance estimate"}
    reference = np.asarray(scores["hybrid_control"])
    print("\nSUMMARY: same-run paired GroupKFold (not Public)", flush=True)
    for recipe,values in probabilities.items():
        if not np.isfinite(values).all():
            raise ValueError("Incomplete OOF predictions")
        per_fold = np.asarray(scores[recipe])
        result["variants"][recipe] = {
            "fold_scores":scores[recipe], "fold_mean":float(per_fold.mean()),
            "fold_std":float(per_fold.std()), "worst_fold":float(per_fold.min()),
            "paired_deltas":(per_fold-reference).tolist(), "fold_wins":int((per_fold > reference).sum()),
            "oof":H.metrics(target, values),
        }
        entry = result["variants"][recipe]
        print(f"{recipe}: mean={entry['fold_mean']:.6f} OOF={entry['oof']['macro_f1']:.6f} "
              f"worst={entry['worst_fold']:.6f} wins={entry['fold_wins']}/{len(splits)}", flush=True)
    (args.output_dir/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Opaque private mapping is for exact same-fold follow-up only; never publish/upload it.
    mapping = frame[["sample_id", "experiment_id"]].copy()
    mapping["fold"] = assignments
    mapping.to_csv(args.output_dir/"fold_assignment.csv", index=False)
    print("CV finished. Metrics and private fold mapping saved locally. No model selected or submitted automatically.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
