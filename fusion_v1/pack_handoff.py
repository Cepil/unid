"""Package the existing Public-0.8504558014 run for private teammate transfer."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import zipfile

# Filled with the exact source hashes of the submitted version during packaging.
EXPECTED_CORE = {'model.py': 'c2a45ac7e0e128b6e3b386e89c42e7f0b112a7807953decca339a3ee3cb085ac', 'train.py': 'bd9d2fea14485d291da3d5178dbeeb3fa568f0ce7061aadd2e82a965a71375fd', 'test.py': '640c1114416014cd48d84e12df77ea1396c10709b61d4df48571d911fd9da480'}
PUBLIC_SCORE = '0.8504558014'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run_dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path,default=None)
    args=p.parse_args()
    root=args.run_dir.resolve()
    if not os.environ.get('SLURM_JOB_ID') or any(x in socket.gethostname().lower() for x in ['master','login']):
        raise RuntimeError('Run this on an allocated compute node, not a login/master node')
    required=['model.py','train.py','test.py','requirements.txt','weights/fusion.pt','fusion_v1_submission.csv']
    for name in required:
        file=root/name
        if not file.is_file() or file.is_symlink():
            raise FileNotFoundError(f'Missing or symlinked required artifact: {name}')
    for name,expected in EXPECTED_CORE.items():
        if digest(root/name)!=expected:
            raise RuntimeError(f'{name} changed from the Public submission source. Preserve original version before packaging.')

    # Inspect metadata locally only. No preprocessing values or data rows are printed.
    import torch
    bundle=torch.load(root/'weights/fusion.pt',map_location='cpu',weights_only=True)
    keys={'encoder','fusion','preprocessing','image_mean','image_std','best_epoch','validation_macro_f1','seed','class_weight'}
    if not keys.issubset(bundle) or bundle['best_epoch']!=16 or abs(bundle['validation_macro_f1']-.778443)>1e-6:
        raise RuntimeError('Checkpoint metadata does not match the reported Public candidate')
    if bundle['seed']!=42 or bundle['class_weight']!='none':
        raise RuntimeError('Unexpected checkpoint training configuration')
    if any(len(bundle['preprocessing'][k])!=9 for k in ['median','mean','std']):
        raise RuntimeError('Unexpected preprocessing dimensions')
    del bundle

    optional=['README.md','README_PUBLIC.md','cv.py','baseline_reference.py','verify_submission.py',
              'logs/fusion_v1.log','logs/groupkfold_v1.log',
              'outputs/cv_v1/metrics.json','outputs/cv_v1/fold_assignment.csv']
    selected=required+[n for n in optional if (root/n).is_file() and not (root/n).is_symlink()]
    output=(args.output or root.parent/'fusion_v1_public_0.8504558014.zip').resolve()
    if output.exists():
        raise FileExistsError('Archive already exists; choose a new --output path to preserve it')
    output.parent.mkdir(parents=True,exist_ok=True)
    packages={}
    for dist in metadata.distributions():
        name=dist.metadata.get('Name')
        if name:
            try:
                packages[name]=metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
    env={'python':sys.version,'platform':platform.platform(),
         'torch':torch.__version__,'torch_cuda':torch.version.cuda,
         'packages':dict(sorted(packages.items(),key=lambda pair:pair[0].lower()))}
    manifest={'public_score_reported_by_user':PUBLIC_SCORE,'submitted_csv':'fusion_v1_submission.csv',
              'checkpoint':'weights/fusion.pt','best_epoch':16,'validation_macro_f1':.778443,
              'trained_on':'Train only; provided Validation selected checkpoint',
              'ensemble':False,'preprocessing':'Embedded in weights/fusion.pt',
              'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'files':{n:{'sha256':digest(root/n),'bytes':(root/n).stat().st_size} for n in selected},
              'absent_optional_files':[n for n in optional if n not in selected],
              'not_created':['Validation predictions/probabilities','confusion matrix','fold checkpoints','feature cache'],
              'verification_reported_by_user':'Full CSV schema/IDs/template order and selected subset/repeat/reverse checks passed',
              'note':'Compare the CSV SHA256 with the downloaded file that was actually uploaded to DAKER.'}
    notes='''# Public 0.8504558014 — team handoff

Original submission: fusion_v1_submission.csv. Actual checkpoint: weights/fusion.pt.
Train only was fitted; provided Validation selected epoch 16 (Macro F1 0.778443).
No ensemble, no Train+Validation refit, and no CV checkpoint is used for inference.
The .pt contains the frozen ResNet-18, fusion head, sensor medians/means/stds,
image embedding means/stds, seed and selection metadata. No separate scaler needed.
No notebook or feature cache was used. Existing validation predictions/probabilities
and confusion matrix were not saved and are not regenerated by this packaging step.
CV code, metrics, fold assignments and training logs are included only when present.
Manifest SHA256 identifies every original artifact; verify the CSV against the
downloaded original file that was actually submitted.

Inference, from this directory:
python test.py --input <test.csv> --image_root <test image root> --output <submission.csv>
Default model path is weights/fusion.pt; inference performs no fitting/download.

Training command used:
python -u train.py --data_dir <battery_data_light> --batch_size 32 --workers 2 --epochs 60 --output weights/fusion.pt
Seed 42, class_weight none, AdamW lr=.001 weight_decay=.001, head batch_size 256,
patience 12. It stopped at epoch 28 and saved best epoch 16, not the last model.

ResNet18_Weights.IMAGENET1K_V1, torchvision 0.20.1; original pretrained weights:
https://download.pytorch.org/models/resnet18-f37072fd.pth
These encoder weights are already inside fusion.pt, so the original download is
not required for inference. Full package versions are in environment.json and
requirements.environment.txt. They describe the training environment; do not
blindly replace the organizer-provided RTX 5080 GPU PyTorch with this cu121 build.
Validate inference in the destination environment before final submission.

This ZIP is for private team transfer only. It intentionally contains the actual
submission CSV, fold mapping and data-derived logs; do not publish it on GitHub.
Original datasets, credentials, virtual environments and caches are excluded.
'''
    # Explicit allowlist; no recursive traversal of data/environment/cache directories.
    partial=output.with_name(output.name+'.partial')
    try:
        with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for name in selected:
                z.write(root/name,'fusion_v1/'+name)
            z.writestr('fusion_v1/HANDOFF.md',notes)
            z.writestr('fusion_v1/manifest.json',json.dumps(manifest,indent=2))
            z.writestr('fusion_v1/environment.json',json.dumps(env,indent=2))
            z.writestr('fusion_v1/requirements.environment.txt',''.join(f'{k}=={v}\n' for k,v in env['packages'].items()))
        with zipfile.ZipFile(partial) as z:
            bad=z.testzip()
            if bad:
                raise RuntimeError('Archive CRC check failed')
        partial.rename(output)
    except Exception:
        # Keep partial output for diagnosis; never remove existing user artifacts.
        raise
    print('Archive:',output)
    print('Bytes:',output.stat().st_size)
    print('Archive SHA256:',digest(output))
    print('Submitted CSV SHA256:',manifest['files']['fusion_v1_submission.csv']['sha256'])
    print('Checkpoint SHA256:',manifest['files']['weights/fusion.pt']['sha256'])
    print('Missing optional files:',', '.join(manifest['absent_optional_files']) or 'none')
    print('READY for private team transfer; not uploaded or sent.')

if __name__=='__main__':
    main()
