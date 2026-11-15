"""Exact T37 MLP training recipe; training entry only."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import base_model as B
import base_epair as M

def train_mlp(images, sensors, y, device, epochs):
    """Same initial seeds, row order, batches, optimizer and epochs as canonical EPAIR."""
    states=[]
    for seed in M.SEEDS:
        B.seed_all(seed)
        net=B.Fusion().to(device)
        loader=DataLoader(TensorDataset(images, torch.from_numpy(sensors), torch.from_numpy(y)),
                          batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
        optimizer=torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion=nn.CrossEntropyLoss()
        for _ in range(epochs):
            net.train()
            for image, sensor, label in loader:
                optimizer.zero_grad(set_to_none=True)
                criterion(net(image.to(device),sensor.to(device)),label.to(device)).backward()
                optimizer.step()
        states.append({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
        del net, optimizer
    return states
