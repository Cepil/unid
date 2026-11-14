"""Extract frozen ImageNet features once; train a small fusion head."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
import model as M

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('weights/fusion.pt'))
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--class_weight', choices=['none','sqrt'], default='none')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError('epochs and patience must be positive')
    M.seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames = {s: M.read_csv(args.data_dir/s/f'{s}.csv') for s in ['train','validation']}
    tr, va = frames['train'], frames['validation']
    if 'experiment_id' not in tr or 'experiment_id' not in va:
        raise ValueError('experiment_id required for split validation')
    if tr.experiment_id.isna().any() or va.experiment_id.isna().any():
        raise ValueError('Missing experiment_id')
    if set(tr.experiment_id) & set(va.experiment_id):
        raise ValueError('Train and Validation experiments overlap')
    if set(tr.sample_id) & set(va.sample_id):
        raise ValueError('Train and Validation sample IDs overlap')
    prep = M.fit_preprocessing(tr)
    backbone = M.encoder(pretrained=True).to(device)
    embeddings = {}
    for s, frame in frames.items():
        print(f'Extracting {s} images', flush=True)
        embeddings[s] = M.extract(frame, [args.data_dir/s, args.data_dir], backbone,
                                   device, args.batch_size, args.workers)
    # Image normalization statistics are learned on Train only.
    image_mean = embeddings['train'].mean(0)
    image_std = embeddings['train'].std(0, unbiased=False).clamp_min(1e-6)
    embeddings = {s:(x-image_mean)/image_std for s,x in embeddings.items()}
    datasets = {s: TensorDataset(embeddings[s], torch.from_numpy(M.apply_preprocessing(f,prep)),
                                torch.from_numpy(M.labels(f))) for s,f in frames.items()}
    loader = DataLoader(datasets['train'], batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    net = M.Fusion().to(device)
    counts = np.bincount(M.labels(tr), minlength=6)
    if (counts==0).any():
        raise ValueError('Train is missing one or more classes')
    weight = None
    if args.class_weight == 'sqrt':
        w = np.sqrt(counts.sum()/(6*counts))
        w = np.clip(w,.33,3)
        weight = torch.tensor(w/w.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    best, stale, best_state, best_epoch = -1.0, 0, None, 0
    val_loader = DataLoader(datasets['validation'], batch_size=256, shuffle=False)
    for epoch in range(1,args.epochs+1):
        net.train()
        for image, sensor, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = criterion(net(image.to(device),sensor.to(device)), y.to(device))
            loss.backward()
            opt.step()
        net.eval()
        preds = []
        with torch.inference_mode():
            for image,sensor,_ in val_loader:
                preds.append(net(image.to(device),sensor.to(device)).argmax(1).cpu())
        pred = torch.cat(preds).numpy()
        score = f1_score(M.labels(va), pred, labels=list(range(6)), average='macro', zero_division=0)
        print(f'Epoch {epoch}: validation_macro_f1={score:.6f}', flush=True)
        if score > best:
            best, stale, best_epoch = score, 0, epoch
            best_state = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            print(classification_report(M.labels(va),pred,labels=list(range(6)),
                                        target_names=[str(i) for i in range(1,7)],zero_division=0), flush=True)
        else:
            stale += 1
        if stale >= args.patience:
            break
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'format_version':1,'encoder':backbone.cpu().state_dict(), 'fusion':best_state,
                'preprocessing':{k:v.tolist() for k,v in prep.items()},
                'image_mean':image_mean,'image_std':image_std,
                'best_epoch':best_epoch,'validation_macro_f1':best,'seed':args.seed,
                'class_weight':args.class_weight},args.output)
    print(f'Saved checkpoint; best epoch={best_epoch}, validation_macro_f1={best:.6f}',flush=True)

if __name__=='__main__':
    main()
