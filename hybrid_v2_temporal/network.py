"""Same fusion architecture as v1, with a configurable sensor input size."""
import torch
from torch import nn


class Fusion(nn.Module):
    def __init__(self, sensor_dim=9):
        super().__init__()
        self.image = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.2))
        self.sensor = nn.Sequential(nn.Linear(sensor_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(160, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, 6))

    def forward(self, image, sensor):
        return self.head(torch.cat([self.image(image), self.sensor(sensor)], dim=1))
