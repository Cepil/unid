"""Aggregate-only check: do same-experiment duplicate timestamps exist, and would the official
'future rows removed' check (first 60% of rows per experiment, by time) cut through a tied timestamp?
Prints counts only; no rows, IDs or values.

python check_ties.py --input ~/data/battery_data_light/test/test.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    args = p.parse_args()
    f = pd.read_csv(args.input, usecols=["sample_id", "experiment_id", "relative_time_s"],
                    dtype={"sample_id": "string", "experiment_id": "string"})
    dup_rows = int(f.duplicated(["experiment_id", "relative_time_s"], keep=False).sum())
    s = f.sort_values(["experiment_id", "relative_time_s"], kind="mergesort")
    rank = s.groupby("experiment_id").cumcount()
    size = s.groupby("experiment_id").sample_id.transform("size")
    s = s.assign(keep=rank < size * 0.6)
    per_time = s.groupby(["experiment_id", "relative_time_s"]).keep.agg(["min", "max"])
    split_groups = int((per_time["min"] != per_time["max"]).sum())
    print(f"experiments: {f.experiment_id.nunique()}, rows: {len(f)}")
    print(f"rows sharing a timestamp with another row of the same experiment: {dup_rows}")
    print(f"timestamps split by the official 60% cut: {split_groups}")
    print("OK: the 60% cut never splits a tied timestamp" if split_groups == 0 else
          "WARNING: the 60% cut splits tied timestamps; peak features at that time may change")


if __name__ == "__main__":
    main()
