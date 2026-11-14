"""Local prediction checks; never print competition rows, labels, images, or IDs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd


def submission(path):
    frame = pd.read_csv(path, dtype={"sample_id":"string"})
    if list(frame.columns) != ["sample_id", "label"] or frame.empty:
        raise ValueError("Invalid submission schema")
    if frame.isna().any().any() or not frame.sample_id.is_unique:
        raise ValueError("Missing values or duplicate IDs")
    if not pd.api.types.is_integer_dtype(frame.label) or not frame.label.isin(range(1, 7)).all():
        raise ValueError("Labels must be integer 1..6")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--final", type=Path, help="Optional original CSV to reproduce")
    parser.add_argument("--sample_submission", type=Path)
    parser.add_argument("--work_dir", type=Path, default=Path("outputs/verification"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype={"sample_id":"string", "experiment_id":"string"})
    if not {"experiment_id", "relative_time_s", "sample_id"}.issubset(frame.columns):
        raise ValueError("Missing input columns")
    if frame.experiment_id.isna().any() or frame.sample_id.isna().any() or not frame.sample_id.is_unique:
        raise ValueError("Invalid IDs")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve().parent/"test.py"
    with tempfile.TemporaryDirectory(dir=args.work_dir, prefix="check-") as tmp:
        tmp = Path(tmp)
        def run(name, part, batch=32):
            csv, output = tmp/f"{name}.csv", tmp/f"{name}_pred.csv"
            part.to_csv(csv, index=False)
            # Capture child output locally; only PASS/FAIL is returned to the console.
            with (tmp/f"{name}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run([sys.executable, str(runner), "--input", str(csv),
                    "--image_root", str(args.image_root.resolve()), "--output", str(output),
                    "--model", str(args.model.resolve()), "--workers", "0", "--batch_size", str(batch)],
                    stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError("Inference check failed; rerun test.py locally to inspect the error")
            pred = submission(output)
            if pred.sample_id.tolist() != part.sample_id.tolist():
                raise ValueError("Inference changed input order")
            return pred.set_index("sample_id").label
        reference = run("full", frame)
        if args.final is not None:
            final = submission(args.final).set_index("sample_id").label
            if set(final.index) != set(reference.index) or not np.array_equal(final.reindex(reference.index), reference):
                raise ValueError("Final CSV reproduction failed")
            print("PASS: final CSV reproduction", flush=True)
        if args.sample_submission is not None:
            ids = pd.read_csv(args.sample_submission, dtype={"sample_id":"string"}).sample_id
            if ids.isna().any() or not ids.is_unique or ids.tolist() != reference.index.tolist():
                raise ValueError("Full-input row order or IDs differ from submission template")
            print("PASS: template order and IDs", flush=True)
        cases = [("repeat", frame, 32), ("shuffled_and_batch17", frame.sample(frac=1, random_state=42), 17)]
        experiments = frame.experiment_id.drop_duplicates().tolist()
        for index in (0, 1):
            group = frame[frame.experiment_id.isin(experiments[index::2])]
            if len(group):
                cases.append((f"experiment_group_{index}", group, 32))
        times = pd.to_numeric(frame.relative_time_s, errors="raise")
        # Timestamp cut includes ALL tied rows. Never cut a same-time group in half.
        prefix = frame.groupby("experiment_id", sort=False).relative_time_s.transform(
            lambda s:float(np.quantile(pd.to_numeric(s, errors="raise"), .6)))
        cases.append(("past_60_percent", frame[times <= prefix], 32))
        for name,part,batch in cases:
            actual = run(name, part, batch)
            if not np.array_equal(actual, reference.reindex(actual.index)):
                raise ValueError(f"FAIL: {name} prediction invariance")
            print("PASS:", name, flush=True)
    print("PASS: local invariance checks; also run the organizer's official checker before submission.", flush=True)


if __name__ == "__main__":
    main()
