"""Fit fixed hybrid on provided Train, evaluate provided Validation, save artifacts."""
import argparse
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
import base_model as B
import model as H

def synthetic_check():
    B.seed_all(42)
    rng=np.random.default_rng(42)
    frame=pd.DataFrame(rng.normal(size=(180,5)),columns=B.SENSORS)
    frame['label']=np.tile(np.arange(1,7),30)
    frame.loc[0,'meas_t1']=np.nan
    frame.loc[1,'meas_v']=np.inf
    images=torch.from_numpy(rng.normal(size=(180,512)).astype(np.float32))
    bundle,trees=H.fit(frame.iloc[:144],images[:144],torch.device('cpu'),
                        epochs=1,tree_iterations=3,pca_components=4)
    val=frame.iloc[144:]
    full=H.predict_from_embeddings(val,images[144:],bundle,trees,torch.device('cpu'))
    with tempfile.TemporaryDirectory(prefix='hybrid-synthetic-') as temp:
        torch.save(bundle,Path(temp)/'hybrid.pt')
        joblib.dump(trees,Path(temp)/'tree.joblib')
        restored=H.predict_from_embeddings(val,images[144:],
                torch.load(Path(temp)/'hybrid.pt',map_location='cpu',weights_only=True),
                joblib.load(Path(temp)/'tree.joblib'),torch.device('cpu'))
        for name in full:
            np.testing.assert_allclose(full[name],restored[name],atol=1e-6,rtol=1e-6)
    reverse=H.predict_from_embeddings(val.iloc[::-1],images[144:].flip(0),bundle,trees,torch.device('cpu'))
    subset=H.predict_from_embeddings(val.iloc[::3],images[144:][::3],bundle,trees,torch.device('cpu'))
    repeat=H.predict_from_embeddings(val,images[144:],bundle,trees,torch.device('cpu'))
    for name in full:
        np.testing.assert_allclose(full[name],reverse[name][::-1],atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(full[name][::3],subset[name],atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(full[name],repeat[name],atol=1e-6,rtol=1e-6)
        assert np.array_equal(full[name].argmax(1),reverse[name][::-1].argmax(1))
        assert np.array_equal(full[name][::3].argmax(1),subset[name].argmax(1))
    print('PASS: synthetic hybrid fitting, artifact reload and repeated/subset/reversed inference',flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir',type=Path)
    p.add_argument('--cache_dir',type=Path)
    p.add_argument('--output',type=Path,default=Path('weights/hybrid.pt'))
    p.add_argument('--batch_size',type=int,default=32)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--synthetic_check',action='store_true')
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an allocated compute node')
    if args.synthetic_check:
        synthetic_check()
        return
    if args.data_dir is None or args.cache_dir is None:
        p.error('--data_dir and --cache_dir required')
    B.seed_all(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':
        raise RuntimeError('CUDA unavailable; check allocation/environment')
    frames={s:B.read_csv(args.data_dir/s/f'{s}.csv') for s in ['train','validation']}
    tr,va=frames['train'],frames['validation']
    for frame in frames.values():
        if 'experiment_id' not in frame or frame.experiment_id.isna().any():
            raise ValueError('Nonmissing experiment_id required')
    if set(tr.experiment_id)&set(va.experiment_id) or set(tr.sample_id)&set(va.sample_id):
        raise ValueError('Train/Validation overlap')
    backbone=B.encoder(pretrained=True).to(device)
    images={}
    for name,frame in frames.items():
        print(f'Preparing {name} image features',flush=True)
        images[name]=H.get_embeddings(frame,args.data_dir/name/f'{name}.csv',
                [args.data_dir/name,args.data_dir],args.cache_dir/f'{name}_resnet18.pt',
                backbone,device,args.batch_size,args.workers)
    bundle,trees=H.fit(tr,images['train'],device)
    probability=H.predict_from_embeddings(va,images['validation'],bundle,trees,device)
    metrics={}
    print('\nSUMMARY: provided Validation (not CV or Public)',flush=True)
    for name,values in probability.items():
        pred=values.argmax(1)
        score=float(f1_score(B.labels(va),pred,labels=list(range(6)),average='macro',zero_division=0))
        class_f1=f1_score(B.labels(va),pred,labels=list(range(6)),average=None,zero_division=0).tolist()
        metrics[name]={'macro_f1':score,'class_f1':class_f1}
        print(f'{name}: macro_f1={score:.6f}',flush=True)
        print('  Class F1 (1..6): '+', '.join(f'{v:.4f}' for v in class_f1),flush=True)
    bundle['encoder']=backbone.cpu().state_dict()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tree_path=args.output.parent/bundle['tree_file']
    joblib.dump(trees,tree_path)
    torch.save(bundle,args.output)
    versions={package:importlib.metadata.version(package) for package in
              ['torch','torchvision','numpy','pandas','scikit-learn','pillow','joblib']}
    metadata={'epochs':16,'seeds':list(H.SEEDS),'mixture_weights':[.5,.5],
              'train_split':'provided Train only','validation_used_for_training':False,
              'metrics':metrics,'environment_versions':versions}
    (args.output.parent/'training_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Saved weights/hybrid.pt and sibling tree.joblib; both are required for inference.',flush=True)
    print('Original fusion_v1 checkpoint and submission are unchanged.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
