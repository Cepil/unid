"""Selected, predeclared hybrid: equal mean of 3-seed fusion and image/sensor HGB."""
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
import base_model as B

SEEDS=(42,43,44)

def get_embeddings(frame,csv,roots,cache,backbone,device,batch_size=32,workers=2):
    digest=hashlib.sha256(csv.read_bytes())
    digest.update(inspect.getsource(B.Images).encode())
    digest.update(inspect.getsource(B.encoder).encode())
    for package in ['torch','torchvision','pillow']:
        digest.update(importlib.metadata.version(package).encode())
    images=B.Images(frame,roots)
    for path in images.paths:
        stat=path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(f'{stat.st_size}:{stat.st_mtime_ns}'.encode())
    fingerprint=digest.hexdigest()
    if cache.exists():
        saved=torch.load(cache,map_location='cpu',weights_only=True)
        if saved['fingerprint']==fingerprint:
            value=saved['embeddings']
            if tuple(value.shape)!=(len(frame),512) or not torch.isfinite(value).all():
                raise ValueError('Invalid image feature cache')
            print('Using matching private image feature cache',flush=True)
            return value
    value=B.extract(frame,roots,backbone,device,batch_size,workers)
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache.with_suffix('.tmp')
    torch.save({'fingerprint':fingerprint,'embeddings':value},temporary)
    temporary.replace(cache)
    return value

def fit(train_frame,raw_images,device,epochs=16,tree_iterations=250,pca_components=32):
    """Train only: all imputation/scaling/PCA are fitted here, never at inference."""
    target=B.labels(train_frame)
    if set(target)!=set(range(6)):
        raise ValueError('Train must contain all six labels')
    prep=B.fit_preprocessing(train_frame)
    sensors=B.apply_preprocessing(train_frame,prep)
    mean=raw_images.mean(0)
    std=raw_images.std(0,unbiased=False).clamp_min(1e-6)
    images=(raw_images-mean)/std
    states=[]
    for seed in SEEDS:
        B.seed_all(seed)
        net=B.Fusion().to(device)
        # Same architecture, optimizer, batches, loss and fixed epochs as analyze_v2.py.
        loader=DataLoader(TensorDataset(images,torch.from_numpy(sensors),torch.from_numpy(target)),
                batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed))
        opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3)
        criterion=nn.CrossEntropyLoss()
        for epoch in range(epochs):
            net.train()
            for image,sensor,y in loader:
                opt.zero_grad(set_to_none=True)
                loss=criterion(net(image.to(device),sensor.to(device)),y.to(device))
                loss.backward()
                opt.step()
        states.append({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
        print(f'Fusion seed {seed}: trained {epochs} fixed epochs',flush=True)
        del net,opt
    pca=PCA(n_components=min(pca_components,len(train_frame)-1,512),
            svd_solver='randomized',random_state=42)
    reduced=pca.fit_transform(images.numpy())
    hgb=HistGradientBoostingClassifier(learning_rate=.06,max_iter=tree_iterations,
            max_leaf_nodes=15,min_samples_leaf=32,l2_regularization=5.0,
            early_stopping=False,random_state=42)
    hgb.fit(np.column_stack([sensors,reduced]),target)
    if not np.array_equal(hgb.classes_,np.arange(6)):
        raise ValueError('Unexpected classifier class order')
    bundle={'format_version':2,'states':states,'seeds':list(SEEDS),'epochs':epochs,
            'preprocessing':{k:v.tolist() for k,v in prep.items()},
            'image_mean':mean,'image_std':std,'tree_file':'tree.joblib',
            'mixture_weights':[.5,.5],'class_labels':[1,2,3,4,5,6]}
    return bundle,{'pca':pca,'hgb':hgb}

@torch.inference_mode()
def predict_from_embeddings(frame,raw_images,bundle,trees,device):
    if bundle['format_version']!=2 or bundle['class_labels']!=[1,2,3,4,5,6]:
        raise ValueError('Unsupported checkpoint format')
    prep={k:np.asarray(v) for k,v in bundle['preprocessing'].items()}
    sensors=B.apply_preprocessing(frame,prep)
    images=(raw_images-bundle['image_mean'])/bundle['image_std']
    if len(images)!=len(frame):
        raise ValueError('Image and sensor row counts differ')
    seed_probs=[]
    for state in bundle['states']:
        net=B.Fusion().to(device)
        net.load_state_dict(state)
        net.eval()
        values=[]
        for start in range(0,len(frame),256):
            logits=net(images[start:start+256].to(device),
                       torch.from_numpy(sensors[start:start+256]).to(device))
            values.append(logits.softmax(1).cpu().numpy())
        seed_probs.append(np.concatenate(values).astype(np.float64))
        del net
    mlp=np.mean(seed_probs,axis=0)
    reduced=trees['pca'].transform(images.numpy())
    hgb=trees['hgb'].predict_proba(np.column_stack([sensors,reduced]))
    w=bundle['mixture_weights']
    hybrid=w[0]*mlp+w[1]*hgb
    result={'fusion_seedavg':mlp,'fusion_hgb':hgb,'fusion_hybrid':hybrid}
    for value in result.values():
        if value.shape!=(len(frame),6) or not np.isfinite(value).all():
            raise ValueError('Invalid probabilities')
        if (value<0).any() or not np.allclose(value.sum(1),1,atol=1e-5):
            raise ValueError('Invalid probability normalization')
    return result
