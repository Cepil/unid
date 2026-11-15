"""End-to-end self test on INVENTED data only (random CSVs, random images, random encoder).

Runs train.py and test.py through their CLIs, then repeats the organizer-style consistency checks:
repeat, row shuffle, experiments split in two runs, first-60%-per-experiment (future removed), and
all-black images must change predictions. Also checks causality and order invariance of features.
Scores here are meaningless. python test_synthetic.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import base_model as B
import model as M

HERE = Path(__file__).resolve().parent


def make_split(root, name, n_exp, rows, rng, prefix, with_labels=True):
    (root / name / "images").mkdir(parents=True, exist_ok=True)
    recs = []
    for e in range(n_exp):
        t = np.cumsum(rng.uniform(3, 12, rows))  # irregular, strictly increasing times
        stage = np.minimum(5, (np.arange(rows) * 6) // rows)
        for i in range(rows):
            sid = f"{prefix}{e:02d}_{i:03d}"
            level = int(40 + 30 * stage[i] + rng.integers(0, 20))
            Image.fromarray(np.clip(rng.normal(level, 25, (24, 32, 3)), 0, 255).astype(np.uint8)).save(
                root / name / "images" / f"{sid}.png")
            rec = {"sample_id": sid, "image_path": f"images/{sid}.png", "experiment_id": f"{prefix}x{e}",
                   "relative_time_s": round(float(t[i]), 3),
                   "meas_v": 4.1 - .1 * stage[i] + rng.normal(0, .02), "meas_a": rng.normal(1, .1),
                   "meas_t1": 25 + 15 * stage[i] + rng.normal(0, 2), "meas_t2": 26 + 16 * stage[i] + rng.normal(0, 2),
                   "meas_p": 101 + rng.normal(0, .5), "thermal_t": 30 + 25 * stage[i] + rng.normal(0, 4)}
            if stage[i] >= 3:
                rec["meas_v"] *= 0.5  # voltage collapse so the drawdown features carry signal
            if with_labels:
                rec["label"] = int(stage[i]) + 1
            recs.append(rec)
    frame = pd.DataFrame(recs).sample(frac=1, random_state=1).reset_index(drop=True)  # scrambled file order
    frame.to_csv(root / name / f"{name}.csv", index=False)
    return frame


def run(args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    return r.stdout


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(prefix="hybrid_epair_selftest_") as tmp:
        tmp = Path(tmp)
        data = tmp / "battery_data_light"
        make_split(data, "train", 8, 36, rng, "tr")
        make_split(data, "validation", 3, 36, rng, "va")
        test = make_split(data, "test", 4, 30, rng, "te", with_labels=False)
        # feature causality and order invariance on the test frame
        full = M.tables(test)[1]
        s = test.sort_values(["experiment_id", "relative_time_s"], kind="mergesort")
        head = s[s.groupby("experiment_id").cumcount() < s.groupby("experiment_id").sample_id.transform("size") * .6]
        part = M.tables(test[test.sample_id.isin(head.sample_id)])[1]
        pd.testing.assert_frame_equal(part, full.loc[part.index])
        perm = rng.permutation(len(test))
        shuf = M.tables(test.iloc[perm].reset_index(drop=True))[1]
        np.testing.assert_allclose(shuf.to_numpy(), full.to_numpy()[perm], equal_nan=True)
        # random-weight encoder checkpoint (no download)
        torch.manual_seed(0)
        torch.save({"encoder": B.encoder(pretrained=False).state_dict()}, tmp / "source.pt")
        weights = HERE / "weights_selftest_tmp"
        if weights.exists():
            raise FileExistsError(f"{weights} exists; remove it first")
        try:
            print(run(["train.py", "--data_dir", data, "--source_checkpoint", tmp / "source.pt",
                       "--output_dir", weights, "--workers", "0"]).strip().splitlines()[-3])
            model = weights / "model.pt"

            def infer(name, frame, root=data / "test"):
                inp, out = tmp / f"{name}_in.csv", tmp / f"{name}_out.csv"
                frame.to_csv(inp, index=False)
                run(["test.py", "--input", inp, "--image_root", root, "--output", out, "--model", model,
                     "--workers", "0"])
                return pd.read_csv(out, dtype={"sample_id": "string"})

            base = infer("full", test)
            assert list(base.columns) == ["sample_id", "label"] and len(base) == len(test)
            assert base.sample_id.tolist() == test.sample_id.astype(str).tolist() and base.label.isin(range(1, 7)).all()

            def diff(a, b):
                m = a.merge(b, on="sample_id")
                return len(m), float((m.label_x != m.label_y).mean())

            n, d = diff(base, infer("repeat", test))
            assert d == 0, "repeat differs"
            n, d = diff(base, infer("shuffle", test.sample(frac=1, random_state=0)))
            assert d == 0 and n == len(test), "shuffle differs"
            exps = sorted(test.experiment_id.unique())
            split = pd.concat([infer(f"split{i}", test[test.experiment_id.isin(exps[i::2])]) for i in range(2)])
            n, d = diff(base, split)
            assert d == 0 and n == len(test), "experiment split differs"
            n, d = diff(base, infer("past", test[test.sample_id.isin(head.sample_id)]))
            assert d == 0, "future-removed (first 60%) differs"
            blank = tmp / "blank"
            for rel in test.image_path:
                (blank / rel).parent.mkdir(parents=True, exist_ok=True)
                with Image.open(data / "test" / rel) as im:
                    Image.new(im.mode, im.size).save(blank / rel)
            n, d = diff(base, infer("blank", test, root=blank))
            assert d > 0, "black images do not change predictions"
            print(f"black-image prediction change on invented data: {d:.1%}")
            # tampering with an artifact must be refused
            original = (weights / "tree.joblib").read_bytes()
            (weights / "tree.joblib").write_bytes(original + b"x")
            try:
                infer("tamper", test)
                raise AssertionError("tampered tree accepted")
            except RuntimeError:
                pass
            (weights / "tree.joblib").write_bytes(original)
        finally:
            for f in sorted(weights.glob("*")) if weights.exists() else []:
                f.unlink()
            if weights.exists():
                weights.rmdir()
    print("PASS: hybrid_epair CLI train/test, repeat, shuffle, experiment split, first-60% (future removed), "
          "black images change predictions, causal/order-invariant features, checksum guard")


if __name__ == "__main__":
    main()
