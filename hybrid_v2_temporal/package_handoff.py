"""Package the existing Public-0.9401 candidate; no training or inference.

Weights and metadata only. The original submitted CSV is hashed, never included
or parsed. Dataset files, logs, caches and private fold mappings are excluded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import tarfile
import uuid


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda:source.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def regular_file(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError("Required artifact is missing or symlinked: "+path.name)


def create_handoff(root, model_dir, submitted_csv, output):
    root,model_dir,submitted_csv,output = [p.resolve() for p in (root,model_dir,submitted_csv,output)]
    checksum_file = output.with_name(output.name+".sha256")
    if output.exists() or checksum_file.exists():
        raise FileExistsError("Archive already exists; choose a new --output to preserve it")
    source_manifest = root/"PUBLIC09401_SOURCE.json"
    regular_file(source_manifest)
    source_record = json.loads(source_manifest.read_text(encoding="utf-8"))
    for name,expected in source_record["core_source_sha256"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Invalid source manifest path")
        path = root/Path(*relative.parts)
        regular_file(path)
        if digest(path) != expected:
            raise ValueError("Core code differs from the submitted candidate: "+name)
    names = ["model.pt","tree.joblib","manifest.json","validation_metrics.json"]
    for name in names:
        regular_file(model_dir/name)
    regular_file(submitted_csv)
    stored_manifest = json.loads((model_dir/"manifest.json").read_text(encoding="utf-8"))
    if set(stored_manifest["files"]) != {"model.pt","tree.joblib"}:
        raise ValueError("Unexpected trained artifact manifest")
    for name,expected in stored_manifest["files"].items():
        if digest(model_dir/name) != expected:
            raise ValueError("Checkpoint/tree checksum differs from the training manifest")
    # Safe CPU-only metadata inspection. No inference, labels or dataset rows.
    import torch
    bundle = torch.load(model_dir/"model.pt",map_location="cpu",weights_only=True)
    if bundle.get("format_version") != 3 or bundle.get("model_family") != "hybrid_v2_temporal":
        raise ValueError("Wrong model family or checkpoint format")
    expected = {"recipe":"temporal_tree","mlp_features":"current","tree_features":"all",
                "epochs":16,"seeds":[42,43,44],"mixture_weights":[.5,.5],
                "class_labels":[1,2,3,4,5,6],"tree_file":"tree.joblib"}
    for key,value in expected.items():
        if bundle.get(key) != value:
            raise ValueError("Checkpoint recipe differs from the Public candidate: "+key)
    if len(bundle.get("states",[])) != 3 or "encoder" not in bundle:
        raise ValueError("Missing inference weights")
    for state in bundle["states"]:
        for name,shape in {"sensor.0.weight":(32,9),"image.0.weight":(128,512),"head.3.weight":(6,64)}.items():
            if name not in state or tuple(state[name].shape) != shape:
                raise ValueError("Unexpected neural head dimensions")
    if "conv1.weight" not in bundle["encoder"] or tuple(bundle["encoder"]["conv1.weight"].shape) != (64,3,7,7):
        raise ValueError("Unexpected encoder dimensions")
    if len(bundle["schemas"]["current"]) != 9 or len(bundle["schemas"]["all"]) != 25:
        raise ValueError("Unexpected feature dimensions")
    validation = json.loads((model_dir/"validation_metrics.json").read_text(encoding="utf-8"))
    if validation.get("recipe") != "temporal_tree" or validation.get("epochs") != 16 or validation.get("validation_used_for_fitting") is not False:
        raise ValueError("Validation metadata differs from the reported training configuration")
    score = float(validation["metrics"]["macro_f1"])
    if abs(score-.751640) > .00000051:
        raise ValueError("Validation score does not match the reported Public candidate")
    versions = bundle["environment_versions"]
    if versions != validation["environment_versions"] or versions != stored_manifest["environment_versions"]:
        raise ValueError("Training environment metadata differs between artifacts")
    submission_sha = digest(submitted_csv)
    model_card = {
        "public_score_display":.9401,"public_score_precision":"Rounded 4-decimal leaderboard display; full precision not recorded",
        "leaderboard_rank_observed":1,"recipe":"temporal_tree","epochs":16,"seeds":[42,43,44],
        "provided_validation_macro_f1":score,"paired_cv_mean_reported":.878267,
        "paired_oof_macro_f1_reported":.880200,"paired_wins_reported":"5/5",
        "submitted_filename":submitted_csv.name,"submitted_csv_sha256":submission_sha,
        "submission_csv_included":False,"trained_on":"provided Train only",
        "private_score_known":False,"source_sha256":source_record["core_source_sha256"],
        "environment_versions":versions,"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "note":"Scores/selection refer to the user-reported candidate. Checksums identify files; they do not independently prove the uploaded DAKER file or score.",
    }
    environment = {"packaging_python":sys.version,"packaging_platform":platform.platform(),
                   "trained_package_versions":versions,"keep_organizer_gpu_torch_torchvision":True}
    # Pin fitted numerical libraries while preserving the target server's GPU build.
    requirements = "# Preserve the organizer-provided compatible GPU torch/torchvision pair.\ntorch\ntorchvision\n"
    for name in ("numpy","pandas","scikit-learn","pillow","joblib"):
        requirements += f"{name}=={versions[name]}\n"
    requirements += "threadpoolctl\n"
    generated = {
        "hybrid_v2_temporal/MODEL_CARD.json":json.dumps(model_card,indent=2).encode(),
        "hybrid_v2_temporal/HANDOFF_ENVIRONMENT.json":json.dumps(environment,indent=2).encode(),
        "hybrid_v2_temporal/requirements.reproduce.txt":requirements.encode(),
    }
    files = {"hybrid_v2_temporal/weights/"+name:model_dir/name for name in names}
    for name in ("TEAM_HANDOFF.md","PUBLIC09401_SOURCE.json"):
        regular_file(root/name)
        files["hybrid_v2_temporal/"+name] = root/name
    checksums = {name:digest(path) for name,path in files.items()}
    checksums.update({name:hashlib.sha256(payload).hexdigest() for name,payload in generated.items()})
    handoff_manifest = {"files":checksums,"source_sha256":source_record["core_source_sha256"],
                        "data_rows_included":False,"submission_csv_included":False,
                        "archive_layout":"Extract beside code; supplies hybrid_v2_temporal/weights/"}
    generated["hybrid_v2_temporal/handoff_manifest.json"] = json.dumps(handoff_manifest,indent=2).encode()
    generated["hybrid_v2_temporal/weights/SHA256SUMS"] = "".join(
        f"{checksums['hybrid_v2_temporal/weights/'+name]}  {name}\n" for name in names).encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    partial = output.with_name(output.name+"."+uuid.uuid4().hex+".partial")
    with tarfile.open(partial,"w:gz",compresslevel=3) as archive:
        for name,path in files.items():
            archive.add(path,arcname=name,recursive=False)
        for name,payload in generated.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info,io.BytesIO(payload))
    with tarfile.open(partial,"r:gz") as archive:
        expected_names = set(files)|set(generated)
        if {member.name for member in archive.getmembers()} != expected_names:
            raise RuntimeError("Unexpected archive entries")
        for name,expected_hash in checksums.items():
            value = hashlib.sha256(archive.extractfile(name).read()).hexdigest()
            if value != expected_hash:
                raise RuntimeError("Archive checksum verification failed")
    # Exclusive creation refuses an existing destination, including concurrent writes.
    # A failed copy leaves partial output for diagnosis rather than deleting user files.
    with output.open("xb") as destination, partial.open("rb") as source:
        shutil.copyfileobj(source,destination)
    partial.unlink()
    archive_sha = digest(output)
    with checksum_file.open("x",encoding="utf-8") as handle:
        handle.write(f"{archive_sha}  {output.name}\n")
    print("PASS: original source, recipe, trained artifact checksums and archive checksums")
    print("Archive:",output)
    print("Archive SHA256:",archive_sha)
    print("Submitted CSV SHA256:",submission_sha)
    print("Ready for private team transfer. No files were uploaded or sent.")
    return output


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",type=Path,default=root/"outputs/selected_temporal_tree")
    parser.add_argument("--submission",type=Path,default=root/"outputs/submission_temporal_tree.csv")
    parser.add_argument("--output",type=Path,default=root.parent/"temporal_tree_public09401_weights.tar.gz")
    args = parser.parse_args()
    create_handoff(root,args.model_dir,args.submission,args.output)


if __name__ == "__main__":
    main()
