"""Current-row image/sensor fusion. No temporal context or Test fitting."""
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
from torchvision.models import resnet18, ResNet18_Weights

SENSORS = ['meas_v', 'meas_a', 'meas_t1', 'meas_t2', 'meas_p']

def seed_all(seed=42):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
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
    frame = pd.read_csv(path, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    required = ['sample_id', 'image_path'] + SENSORS
    if not set(required).issubset(frame.columns):
        raise ValueError('Missing required CSV columns')
    if frame.empty or frame.sample_id.isna().any() or not frame.sample_id.is_unique:
        raise ValueError('CSV must contain unique, nonmissing sample_id values')
    return frame

def numeric_features(frame):
    x = frame[SENSORS].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    v, a, t1, t2, p = x.T
    with np.errstate(invalid='ignore', over='ignore'):
        extra = np.column_stack([t2-t1, (t1+t2)/2, np.maximum(t1,t2), v*a])
    x = np.column_stack([x, extra])
    x[~np.isfinite(x)] = np.nan
    return x

def fit_preprocessing(frame):
    x = numeric_features(frame)
    median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in x.T])
    x = np.where(np.isnan(x), median, x)
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-8] = 1.0
    return {'median': median, 'mean': mean, 'std': std}

def apply_preprocessing(frame, prep):
    x = numeric_features(frame)
    x = np.where(np.isnan(x), prep['median'], x)
    return ((x-prep['mean'])/prep['std']).clip(-20,20).astype(np.float32)

class Images(Dataset):
    def __init__(self, frame, roots):
        self.paths = []
        for raw in frame.image_path:
            if pd.isna(raw):
                raise ValueError('Missing image_path')
            candidates = [Path(root)/str(raw) for root in roots]
            existing = next((p for p in candidates if p.is_file()), None)
            if existing is None:
                raise FileNotFoundError('Image path does not resolve; check --image_root. No basename fallback is used.')
            self.paths.append(existing)
        # Preserve the full frame; do not crop away edge regions.
        self.transform = transforms.Compose([
            transforms.Resize((224,224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406], [.229,.224,.225]),
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            return self.transform(image.convert('RGB'))

def encoder(pretrained=False):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Identity()
    return model.requires_grad_(False).eval()

@torch.inference_mode()
def extract(frame, roots, backbone, device, batch_size=32, workers=2):
    loader = DataLoader(Images(frame, roots), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type=='cuda')
    result = []
    backbone.eval()
    for step, images in enumerate(loader, 1):
        result.append(backbone(images.to(device)).cpu())
        if step % 100 == 0:
            print(f'Image batches: {step}/{len(loader)}', flush=True)
    return torch.cat(result)

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(nn.Linear(512,128), nn.ReLU(), nn.Dropout(.2))
        self.sensor = nn.Sequential(nn.Linear(9,32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(160,64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64,6))
    def forward(self, image, sensor):
        return self.head(torch.cat([self.image(image), self.sensor(sensor)], dim=1))

def labels(frame):
    values = pd.to_numeric(frame['label'], errors='raise').to_numpy()
    if not np.isin(values, np.arange(1,7)).all():
        raise ValueError('Labels must be integers 1..6')
    return values.astype(np.int64)-1
