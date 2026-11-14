"""Frozen v1 image encoder plus selectable causal sensor features."""
from pathlib import Path
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from features import SENSORS, FEATURE_SETS, build_sensor_features, feature_names
from features import fit_preprocessing, apply_preprocessing
from network import Fusion


def seed_all(seed=42):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def read_csv(path):
    frame = pd.read_csv(path, dtype={"sample_id": "string", "experiment_id": "string"})
    if not {"sample_id", "image_path", *SENSORS}.issubset(frame.columns):
        raise ValueError("Missing required CSV columns")
    if frame.empty or frame.sample_id.isna().any() or not frame.sample_id.is_unique:
        raise ValueError("Input needs unique, nonmissing sample IDs and at least one row")
    return frame


def labels(frame):
    values = pd.to_numeric(frame["label"], errors="raise").to_numpy()
    if not np.isin(values, np.arange(1, 7)).all():
        raise ValueError("Labels must be integers 1..6")
    return values.astype(np.int64)-1


class Images(Dataset):
    def __init__(self, frame, roots):
        self.paths = []
        for raw in frame.image_path:
            if pd.isna(raw):
                raise ValueError("Missing image path")
            path = next((Path(root)/str(raw) for root in roots if (Path(root)/str(raw)).is_file()), None)
            if path is None:
                raise FileNotFoundError("Image not found; check --image_root")
            self.paths.append(path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def encoder():
    backbone = resnet18(weights=None)
    backbone.fc = nn.Identity()
    return backbone.requires_grad_(False).eval()


@torch.inference_mode()
def extract(frame, roots, backbone, device, batch_size=32, workers=2):
    loader = DataLoader(Images(frame, roots), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    backbone.eval()
    chunks = []
    for step, images in enumerate(loader, 1):
        chunks.append(backbone(images.to(device)).cpu())
        if step % 100 == 0:
            print(f"Image extraction batch {step}/{len(loader)}", flush=True)
    return torch.cat(chunks)


@torch.inference_mode()
def predict(net, embeddings, sensors, device, batch_size=256):
    net.eval()
    predictions = []
    for start in range(0, len(sensors), batch_size):
        predictions.append(net(embeddings[start:start+batch_size].to(device),
                               sensors[start:start+batch_size].to(device)).argmax(1).cpu())
    return torch.cat(predictions).numpy()
