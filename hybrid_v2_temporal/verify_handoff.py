"""Verify received code/weight checksums and optional reproduced CSV; no inference."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda:source.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def verify(root, reproduced_submission=None, check_environment=True):
    root = root.resolve()
    manifest = json.loads((root/"handoff_manifest.json").read_text(encoding="utf-8"))
    for name,expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe handoff path")
        path = root.parent/Path(*relative.parts)
        if not path.resolve().is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise ValueError("Missing, symlinked or out-of-scope handoff artifact")
        if digest(path) != expected:
            raise ValueError("Handoff artifact checksum mismatch: "+path.name)
    source = json.loads((root/"PUBLIC09401_SOURCE.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != source["core_source_sha256"]:
        raise ValueError("Handoff/source manifest mismatch")
    for name,expected in manifest["source_sha256"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe source path")
        path = root/Path(*relative.parts)
        if not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError("Source differs from submitted version: "+name)
    print("PASS: shared core source and every listed artifact checksum")
    card = json.loads((root/"MODEL_CARD.json").read_text(encoding="utf-8"))
    if reproduced_submission is not None:
        if not reproduced_submission.is_file() or reproduced_submission.is_symlink() or digest(reproduced_submission) != card["submitted_csv_sha256"]:
            raise ValueError("Reproduced submission differs from the original submitted file")
        print("PASS: reproduced CSV SHA256 equals original submitted CSV")
    if check_environment:
        expected = card["environment_versions"]["scikit-learn"]
        try:
            actual = importlib.metadata.version("scikit-learn")
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        if actual != expected:
            raise RuntimeError(f"Use the trained scikit-learn version {expected}; installed version is {actual}. See requirements.reproduce.txt. Preserve the GPU torch/torchvision pair.")
        print("PASS: scikit-learn version matches trained artifacts")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--reproduced_submission",type=Path)
    args = parser.parse_args()
    verify(args.root,args.reproduced_submission)


if __name__ == "__main__":
    main()
