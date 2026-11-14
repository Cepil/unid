"""Train-only paired experiments; no Public feedback optimization or Test access.

Run only on an allocated compute node. Caches and OOF artifacts stay private.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix
from threadpoolctl import threadpool_limits
import model as M
from cv import split_indices

NAMES=['v1_seed42','fusion_seedavg','sensor_hgb_diagnostic','fusion_hgb','fusion_hybrid']

def macro(y,pred):
    return float(f1_score(y,pred,labels=list(range(6)),average='macro',zero_division=0))

def get_embeddings(frame,csv,roots,cache,device,batch_size,workers):
    """Reuse only fixed ImageNet features; detect input/transform/image-file changes."""
    digest=hashlib.sha256(csv.read_bytes())
    digest.update(inspect.getsource(M.Images).encode())
    digest.update(inspect.getsource(M.encoder).encode())
    for package in ['torch','torchvision','pillow']:
        digest.update(importlib.metadata.version(package).encode())
    images=M.Images(frame,roots)
    for path in images.paths:
        stat=path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(f'{stat.st_size}:{stat.st_mtime_ns}'.encode())
    fingerprint=digest.hexdigest()
    if cache.exists():
        saved=torch.load(cache,map_location='cpu',weights_only=True)
        if saved['fingerprint']==fingerprint:
            values=saved['embeddings']
            if tuple(values.shape)!=(len(frame),512) or not torch.isfinite(values).all():
                raise ValueError('Invalid embedding cache; remove only this cache file and retry')
            print('Using matching private fixed-image feature cache',flush=True)
            return values
    print('Extracting Train ImageNet features; later runs reuse the local cache',flush=True)
    backbone=M.encoder(pretrained=True).to(device)
    values=M.extract(frame,roots,backbone,device,batch_size,workers)
    del backbone
    if device.type=='cuda':
        torch.cuda.empty_cache()
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache.with_suffix('.tmp')
    torch.save({'fingerprint':fingerprint,'embeddings':values},temporary)
    temporary.replace(cache)
    return values

def fit_predict_suite(train_frame,val_frame,raw_tr,raw_va,device,
                      epochs=16,seeds=(42,43,44),tree_iterations=250,pca_components=32):
    """Fit exclusively on training rows, then return held-out probability matrices."""
    y=M.labels(train_frame)
    if set(y)!=set(range(6)):
        raise ValueError('A training fold is missing a class; review experiment split')
    prep=M.fit_preprocessing(train_frame)
    sensor_tr=M.apply_preprocessing(train_frame,prep)
    sensor_va=M.apply_preprocessing(val_frame,prep)
    mean=raw_tr.mean(0)
    std=raw_tr.std(0,unbiased=False).clamp_min(1e-6)
    image_tr=(raw_tr-mean)/std
    image_va=(raw_va-mean)/std
    probabilities={}
    seed_probs=[]
    for seed in seeds:
        M.seed_all(seed)
        dataset=TensorDataset(image_tr,torch.from_numpy(sensor_tr),torch.from_numpy(y))
        loader=DataLoader(dataset,batch_size=256,shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
        net=M.Fusion().to(device)
        optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3)
        criterion=nn.CrossEntropyLoss()
        for epoch in range(epochs):
            net.train()
            for image,sensor,target in loader:
                optimizer.zero_grad(set_to_none=True)
                loss=criterion(net(image.to(device),sensor.to(device)),target.to(device))
                loss.backward()
                optimizer.step()
        net.eval()
        pred=[]
        with torch.inference_mode():
            for start in range(0,len(val_frame),256):
                logits=net(image_va[start:start+256].to(device),
                           torch.from_numpy(sensor_va[start:start+256]).to(device))
                pred.append(logits.softmax(1).cpu().numpy())
        probability=np.concatenate(pred).astype(np.float64)
        seed_probs.append(probability)
        if seed==42:
            probabilities['v1_seed42']=probability
        print(f'  Fusion seed {seed}: training complete',flush=True)
        del net,optimizer
    probabilities['fusion_seedavg']=np.mean(seed_probs,axis=0)
    if 'v1_seed42' not in probabilities:
        raise ValueError('Seed 42 is required for the paired v1 reference')

    # PCA and its input scaling are fit on fold Train, never on held-out rows.
    components=min(pca_components,len(train_frame)-1,image_tr.shape[1])
    pca=PCA(n_components=components,svd_solver='randomized',random_state=42)
    reduced_tr=pca.fit_transform(image_tr.numpy())
    reduced_va=pca.transform(image_va.numpy())
    for name,xtr,xva in [
        ('sensor_hgb_diagnostic',sensor_tr,sensor_va),
        ('fusion_hgb',np.column_stack([sensor_tr,reduced_tr]),
                      np.column_stack([sensor_va,reduced_va])),
    ]:
        classifier=HistGradientBoostingClassifier(learning_rate=.06,max_iter=tree_iterations,
                 max_leaf_nodes=15,min_samples_leaf=32,l2_regularization=5.0,
                 early_stopping=False,random_state=42)
        classifier.fit(xtr,y)
        if not np.array_equal(classifier.classes_,np.arange(6)):
            raise ValueError('Unexpected probability class order')
        probabilities[name]=classifier.predict_proba(xva)
        print(f'  {name}: training complete',flush=True)
    # One predeclared mixture; no fold-label or Public-score weight search.
    probabilities['fusion_hybrid']=.5*probabilities['fusion_seedavg']+.5*probabilities['fusion_hgb']
    for value in probabilities.values():
        if value.shape!=(len(val_frame),6) or not np.isfinite(value).all():
            raise ValueError('Invalid probabilities')
        if not np.allclose(value.sum(1),1,atol=1e-5) or (value<0).any():
            raise ValueError('Invalid probability normalization')
    return probabilities

def synthetic_check():
    """Exercise fitting, missing-value handling and output alignment without real data."""
    M.seed_all(42)
    rng=np.random.default_rng(42)
    frame=pd.DataFrame(rng.normal(size=(180,5)),columns=M.SENSORS)
    frame['label']=np.tile(np.arange(1,7),30)
    frame.loc[0,'meas_t1']=np.nan
    frame.loc[1,'meas_v']=np.inf
    images=torch.from_numpy(rng.normal(size=(180,512)).astype(np.float32))
    probs=fit_predict_suite(frame.iloc[:144],frame.iloc[144:],images[:144],images[144:],
            torch.device('cpu'),epochs=1,seeds=(42,43),tree_iterations=3,pca_components=4)
    assert set(probs)==set(NAMES)
    assert all(p.shape==(36,6) for p in probs.values())
    print('PASS: synthetic fitting, imputation, PCA and probability shapes',flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir',type=Path)
    p.add_argument('--cache_dir',type=Path)
    p.add_argument('--output_dir',type=Path,default=Path('outputs/analysis_v2'))
    p.add_argument('--folds_csv',type=Path,default=None)
    p.add_argument('--batch_size',type=int,default=32)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--synthetic_check',action='store_true')
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an allocated compute node; never run on the master')
    if args.synthetic_check:
        synthetic_check()
        return
    if args.data_dir is None or args.cache_dir is None:
        p.error('--data_dir and --cache_dir required')
    M.seed_all(42)
    csv=args.data_dir/'train/train.csv'
    frame=M.read_csv(csv)
    y=M.labels(frame)
    splits=split_indices(frame,5,args.folds_csv)
    assignments=np.full(len(frame),-1,dtype=np.int64)
    seen=np.zeros(len(frame),dtype=np.int64)
    for fold,(tr,va) in enumerate(splits):
        if set(frame.iloc[tr].experiment_id)&set(frame.iloc[va].experiment_id):
            raise ValueError('Experiment leakage')
        seen[va]+=1
        assignments[va]=fold
    if not np.all(seen==1):
        raise ValueError('Each Train row must be held out exactly once')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':
        raise RuntimeError('CUDA unavailable; stop and check the allocated GPU/environment')
    embeddings=get_embeddings(frame,csv,[args.data_dir/'train',args.data_dir],
            args.cache_dir/'train_resnet18.pt',device,args.batch_size,args.workers)
    probs={name:np.zeros((len(frame),6),dtype=np.float64) for name in NAMES}
    scores={name:[] for name in NAMES}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    mapping=frame[['sample_id','experiment_id']].copy()
    mapping['fold']=assignments
    mapping.to_csv(args.output_dir/'fold_assignment.csv',index=False)
    for fold,(tr,va) in enumerate(splits,1):
        print(f'\nFOLD {fold}/{len(splits)}',flush=True)
        outputs=fit_predict_suite(frame.iloc[tr],frame.iloc[va],embeddings[tr],embeddings[va],device)
        for name in NAMES:
            probs[name][va]=outputs[name]
            value=macro(y[va],outputs[name].argmax(1))
            scores[name].append(value)
            print(f'Fold {fold}: {name}={value:.6f}',flush=True)
        # Keep completed-fold probabilities locally in case allocation ends early.
        np.savez_compressed(args.output_dir/'oof_partial.npz',fold=assignments,
                            completed_folds=fold,**probs)
    summary={'epochs':16,'seeds':[42,43,44],'hybrid_weights':[.5,.5],
        'pca_components':32,'tree_iterations':250,'n_folds':len(splits),
        'data_used':'Train only; no provided Validation or Test', 'models':{}}
    print('\nSUMMARY: paired Train-only GroupKFold (Macro F1)',flush=True)
    reference=np.array(scores['v1_seed42'])
    for name in NAMES:
        prediction=probs[name].argmax(1)
        values=np.array(scores[name])
        entry={'fold_scores':scores[name],'fold_mean':float(values.mean()),
               'fold_std_population':float(values.std()),'worst_fold':float(values.min()),
               'oof_macro_f1':macro(y,prediction),
               'oof_class_f1':f1_score(y,prediction,labels=list(range(6)),average=None,zero_division=0).tolist(),
               'confusion_matrix':confusion_matrix(y,prediction,labels=list(range(6))).tolist(),
               'paired_deltas_vs_v1':(values-reference).tolist(),
               'fold_wins_vs_v1':int((values>reference).sum()),
               'eligible_as_final_multimodal':name!='sensor_hgb_diagnostic'}
        summary['models'][name]=entry
        print(f'{name}: mean={entry["fold_mean"]:.6f} std={entry["fold_std_population"]:.6f} '
              f'OOF={entry["oof_macro_f1"]:.6f} worst={entry["worst_fold"]:.6f} '
              f'wins_vs_v1={entry["fold_wins_vs_v1"]}/{len(splits)}',flush=True)
        print('  Class F1 (1..6): '+', '.join(f'{x:.4f}' for x in entry['oof_class_f1']),flush=True)
        print('  Paired fold deltas: '+', '.join(f'{x:+.6f}' for x in entry['paired_deltas_vs_v1']),flush=True)
    sensor=summary['models']['sensor_hgb_diagnostic']['oof_macro_f1']
    image_sensor=summary['models']['fusion_hgb']['oof_macro_f1']
    print(f'Image contribution within HGB family: OOF delta={image_sensor-sensor:+.6f}',flush=True)
    (args.output_dir/'metrics.json').write_text(json.dumps(summary,indent=2))
    print('Analysis complete. Existing fusion.pt/submission unchanged. No candidate selected or submitted automatically.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
