"""Paired Train-only experiment GroupKFold; fixed epochs, no fold-best selection."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
import model as M

def split_indices(frame, n_splits, assignment=None):
    if 'experiment_id' not in frame or frame.experiment_id.isna().any():
        raise ValueError('Nonmissing experiment_id is required')
    if assignment is None:
        if frame.experiment_id.nunique() < n_splits:
            raise ValueError('Too few experiments for requested fold count')
        return list(GroupKFold(n_splits=n_splits).split(frame, groups=frame.experiment_id))
    mapping = pd.read_csv(assignment, dtype={'sample_id':'string'})
    if not {'sample_id','fold'}.issubset(mapping.columns):
        raise ValueError('Fold CSV needs sample_id,fold columns')
    if mapping.sample_id.isna().any() or not mapping.sample_id.is_unique:
        raise ValueError('Fold CSV has missing or duplicate IDs')
    if set(mapping.sample_id) != set(frame.sample_id):
        raise ValueError('Fold CSV must match Train sample IDs exactly')
    aligned = frame[['sample_id','experiment_id']].merge(mapping[['sample_id','fold']],
                         on='sample_id',how='left',sort=False,validate='one_to_one')
    if aligned.fold.isna().any() or aligned.groupby('experiment_id').fold.nunique().max()!=1:
        raise ValueError('Every experiment must belong to exactly one fold')
    fold_ids = aligned.fold.to_numpy()
    values = pd.unique(fold_ids)
    if len(values)<2:
        raise ValueError('Need at least two folds')
    return [(np.flatnonzero(fold_ids!=v),np.flatnonzero(fold_ids==v)) for v in values]

def score(y,pred):
    return float(f1_score(y,pred,labels=list(range(6)),average='macro',zero_division=0))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir',type=Path,required=True)
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--folds_csv',type=Path,default=None)
    p.add_argument('--epochs',type=int,default=16)
    p.add_argument('--batch_size',type=int,default=32,help='Image extraction batch size')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output_dir',type=Path,default=Path('outputs/cv_v1'))
    p.add_argument('--baseline_model',type=Path,default=Path(__file__).resolve().parent/'baseline_reference.py')
    args=p.parse_args()
    if args.epochs<1 or args.folds<2:
        raise ValueError('Positive epochs and at least two folds required')
    M.seed_all(args.seed)
    frame=M.read_csv(args.data_dir/'train/train.csv')
    y=M.labels(frame)
    splits=split_indices(frame,args.folds,args.folds_csv)
    seen=np.zeros(len(frame),dtype=np.int64)
    assignments=np.full(len(frame),-1,dtype=np.int64)
    for fold,(tr,va) in enumerate(splits):
        if set(frame.iloc[tr].experiment_id)&set(frame.iloc[va].experiment_id):
            raise ValueError('Experiment leakage between folds')
        if set(y[tr])!=set(range(6)):
            raise ValueError('A training fold lacks one or more classes; review split design')
        seen[va]+=1
        assignments[va]=fold
    if not np.all(seen==1):
        raise ValueError('Each Train row must be held out once')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    private_mapping=frame[['sample_id','experiment_id']].copy()
    private_mapping['fold']=assignments
    private_mapping.to_csv(args.output_dir/'fold_assignment.csv',index=False)

    # Build paired official-baseline features; no learned preprocessing occurs here.
    spec=importlib.util.spec_from_file_location('official_baseline',args.baseline_model)
    baseline=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    roots=[args.data_dir/'train',args.data_dir]
    paths=baseline.resolve_image_paths(frame,roots)
    print('Extracting official baseline image statistics',flush=True)
    img=baseline.extract_image_features(frame,paths,n_jobs=2)
    base_x=baseline.build_features(frame,img)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Extracting frozen ImageNet features once for Train',flush=True)
    backbone=M.encoder(pretrained=True).to(device)
    raw_embeddings=M.extract(frame,roots,backbone,device,args.batch_size,args.workers)
    del backbone
    if device.type=='cuda':
        torch.cuda.empty_cache()
    names=['baseline_none','baseline_balanced','fusion_v1']
    predictions={name:np.full(len(frame),-1,dtype=np.int64) for name in names}
    fold_scores={name:[] for name in names}
    for fold,(tr,va) in enumerate(splits,1):
        for name,cw in [('baseline_none',None),('baseline_balanced','balanced')]:
            classifier=baseline.build_model(cw)
            classifier.fit(base_x.iloc[tr],y[tr]+1)
            pred=classifier.predict(base_x.iloc[va]).astype(np.int64)-1
            predictions[name][va]=pred
            fold_scores[name].append(score(y[va],pred))

        # All fitted sensor/embedding preprocessing is restricted to this fold's Train.
        prep=M.fit_preprocessing(frame.iloc[tr])
        sensor_tr=torch.from_numpy(M.apply_preprocessing(frame.iloc[tr],prep))
        sensor_va=torch.from_numpy(M.apply_preprocessing(frame.iloc[va],prep))
        mean=raw_embeddings[tr].mean(0)
        std=raw_embeddings[tr].std(0,unbiased=False).clamp_min(1e-6)
        embed_tr=(raw_embeddings[tr]-mean)/std
        embed_va=(raw_embeddings[va]-mean)/std
        M.seed_all(args.seed)
        loader=DataLoader(TensorDataset(embed_tr,sensor_tr,torch.from_numpy(y[tr])),
                batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
        net=M.Fusion().to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3)
        loss_fn=nn.CrossEntropyLoss()
        # Fixed in advance from the separate provided Validation run, not from fold labels.
        for epoch in range(1,args.epochs+1):
            net.train()
            for image,sensor,target in loader:
                opt.zero_grad(set_to_none=True)
                loss=loss_fn(net(image.to(device),sensor.to(device)),target.to(device))
                loss.backward()
                opt.step()
            if epoch%4==0:
                print(f'Fold {fold}/{len(splits)}: trained epoch {epoch}/{args.epochs}',flush=True)
        net.eval()
        pred=[]
        with torch.inference_mode():
            for start in range(0,len(va),256):
                pred.append(net(embed_va[start:start+256].to(device),
                                sensor_va[start:start+256].to(device)).argmax(1).cpu())
        pred=torch.cat(pred).numpy()
        predictions['fusion_v1'][va]=pred
        fold_scores['fusion_v1'].append(score(y[va],pred))
        print(f'Fold {fold}: '+', '.join(f'{n}={fold_scores[n][-1]:.6f}' for n in names),flush=True)
        del net,opt

    summary={'split':'Train-only experiment GroupKFold' if args.folds_csv is None else 'provided fold mapping',
             'n_folds':len(splits),'fusion_epochs':args.epochs,'seed':args.seed,
             'validation_used_for_fold_checkpoint_selection':False,'models':{}}
    print('\nSUMMARY (Macro F1; all six labels)',flush=True)
    for name in names:
        if not np.isin(predictions[name],np.arange(6)).all():
            raise ValueError('Missing out-of-fold prediction')
        scores=fold_scores[name]
        entry={'fold_scores':scores,'fold_mean':float(np.mean(scores)),
               'fold_std_population':float(np.std(scores)),'oof_macro_f1':score(y,predictions[name]),
               'oof_class_f1':f1_score(y,predictions[name],labels=list(range(6)),
                                     average=None,zero_division=0).tolist()}
        summary['models'][name]=entry
        print(f'{name}: mean={entry["fold_mean"]:.6f} std={entry["fold_std_population"]:.6f} '
              f'OOF={entry["oof_macro_f1"]:.6f}',flush=True)
    for name in names[:2]:
        delta=np.array(fold_scores['fusion_v1'])-np.array(fold_scores[name])
        print(f'Fusion minus {name}, paired fold deltas: '+', '.join(f'{d:+.6f}' for d in delta),flush=True)
    (args.output_dir/'metrics.json').write_text(json.dumps(summary,indent=2))
    print('CV complete. No Test or provided Validation rows were loaded. Existing checkpoint unchanged.',flush=True)

if __name__=='__main__':
    main()
