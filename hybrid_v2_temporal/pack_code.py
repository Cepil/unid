"""Code-only archive, explicit allowlist; never package datasets, caches or weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

FILES = ["features.py", "network.py", "model.py", "hybrid.py", "workflow.py", "cv.py", "train.py", "test.py", "verify.py",
         "test_synthetic.py", "test_hybrid_synthetic.py", "test_integration_synthetic.py", "run_cv.sh", "requirements.txt",
         "README.md", "pack_code.py", "reference/base_model.py", "reference/hybrid_reference.py",
         "TEAM_HANDOFF.md", "PUBLIC09401_SOURCE.json", "package_handoff.py", "verify_handoff.py",
         "test_handoff_synthetic.py", ".gitignore"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/hybrid_v2_temporal_code.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.output.exists():
        raise FileExistsError("Choose a new archive path; existing file is preserved")
    for name in FILES:
        if not (root/name).is_file() or (root/name).is_symlink():
            raise ValueError("Missing or symlinked source")
    manifest = {"contains_competition_data":False, "contains_weights":False,
                "source_sha256":{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in FILES}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,"x",zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(root/name,"hybrid_v2_temporal/"+name)
        archive.writestr("hybrid_v2_temporal/package_manifest.json",json.dumps(manifest,indent=2))
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip():
            raise RuntimeError("Archive CRC failure")
        for name,digest in manifest["source_sha256"].items():
            if hashlib.sha256(archive.read("hybrid_v2_temporal/"+name)).hexdigest() != digest:
                raise RuntimeError("Source checksum mismatch")
    print(args.output, "bytes=",args.output.stat().st_size,"sha256=",hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
