"""Full CLI/cache/artifact/verification checks using invented images and rows only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import torch
import model as M


def main():
    root = Path(__file__).resolve().parent
    output_root = root/"outputs"
    output_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root, prefix="invented-integration-") as directory:
        tmp = Path(directory)
        data = tmp/"invented_data"
        rng = np.random.default_rng(42)
        for split,n in [("train",4), ("validation",2), ("test",2)]:
            images = data/split/"images"
            images.mkdir(parents=True)
            rows = []
            for experiment in range(n):
                for step in range(6):
                    name = f"invented_{split}_{experiment}_{step}"
                    Image.fromarray(rng.integers(0,256,(40,48,3),dtype=np.uint8)).save(images/f"{name}.png")
                    row = dict(sample_id=name, experiment_id=f"invented_{split}_{experiment}", relative_time_s=step*15,
                               meas_v=4.-step*.6, meas_a=0., meas_t1=20.+step*12, meas_t2=25.+step*8,
                               meas_p=step+1., image_path=f"images/{name}.png")
                    if split != "test":
                        row["label"] = step+1
                    rows.append(row)
            pd.DataFrame(rows).to_csv(data/split/f"{split}.csv",index=False)
            if split == "test":
                pd.DataFrame({"sample_id":[r["sample_id"] for r in rows], "label":1}).to_csv(data/"sample_submission.csv",index=False)
        M.seed_all(1)
        source = tmp/"invented_source.pt"
        torch.save({"encoder":M.encoder().state_dict()},source)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

        def run(script,arguments,expected_failure=False):
            completed = subprocess.run([sys.executable,str(root/script),*map(str,arguments)],cwd=root,
                capture_output=True,text=True,encoding="utf-8",errors="replace")
            if expected_failure:
                if not completed.returncode or "FileExistsError" not in completed.stderr:
                    raise AssertionError("Artifact overwrite protection failed")
            elif completed.returncode:
                print(completed.stdout)
                print(completed.stderr)
                raise RuntimeError("Synthetic integration failed: "+script)
            print("PASS: invented-data",script,"overwrite protection" if expected_failure else "execution",flush=True)
            return completed.stdout

        cv_args = ["--data_dir",data,"--source_checkpoint",source,"--output_dir",tmp/"cv",
                   "--cache_dir",tmp/"cache","--folds","2","--epochs","1","--workers","0","--batch_size","8"]
        run("cv.py",cv_args)
        cv = json.loads((tmp/"cv/metrics.json").read_text(encoding="utf-8"))
        assert set(cv["variants"]) == {"hybrid_control","temporal_tree","temporal_both"}
        run("cv.py",cv_args,expected_failure=True)
        train_args = ["--data_dir",data,"--source_checkpoint",source,"--recipe","temporal_tree",
                      "--output_dir",tmp/"trained","--cache_dir",tmp/"cache","--epochs","1","--workers","0","--batch_size","8"]
        stdout = run("train.py",train_args)
        assert "Using matching private frozen-image cache" in stdout
        run("train.py",train_args,expected_failure=True)
        model = tmp/"trained/model.pt"
        saved = torch.load(model,map_location="cpu",weights_only=True)
        assert saved["format_version"] == 3 and saved["recipe"] == "temporal_tree"
        prediction = tmp/"submission.csv"
        run("test.py",["--input",data/"test/test.csv","--image_root",data/"test","--model",model,
                       "--output",prediction,"--workers","0","--batch_size","8"])
        run("verify.py",["--input",data/"test/test.csv","--image_root",data/"test","--model",model,
                         "--final",prediction,"--sample_submission",data/"sample_submission.csv","--work_dir",tmp/"checks"])
        manifest = json.loads((tmp/"trained/manifest.json").read_text(encoding="utf-8"))
        for name,digest in manifest["files"].items():
            assert hashlib.sha256((tmp/"trained"/name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        print("PASS: paired CV, cache reuse, versioned artifacts, inference checks and original preservation",flush=True)
    print("Invented data only; no competition performance was measured.",flush=True)


if __name__ == "__main__":
    main()
